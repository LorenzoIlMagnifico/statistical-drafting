import torch
import torch.nn as nn
import torch.nn.functional as F


class DraftNet(nn.Module):
    def __init__(self, cardnames):
        """
        Simple MLP network to predict draft picks.

        Args:
            cardnames (List[str]): Names of cards in the set.
            dropout (float): Dropout rate for regularization.
        """
        super(DraftNet, self).__init__()

        # Customize to given set.
        self.cardnames = cardnames
        # Input: pool (N) + missing (N) + position (2)
        hidden_dims = [len(self.cardnames) * 2 + 2, 400, 400]

        # Network layers. 
        self.dropout_layer = nn.Dropout(0.6)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in hidden_dims[1:])
        self.output_layer = nn.Linear(hidden_dims[-1], len(cardnames))


    def forward(self, x, pack, position, missing):
        # Concatenate pool, missing-card vector, and draft position
        x = torch.cat([x, missing, position], dim=-1)

        # Hidden layers
        for layer, norm in zip(self.hidden_layers, self.norms):
            x = layer(x)
            x = F.gelu(x)
            x = self.dropout_layer(x)
            x = norm(x) # Apply BatchNorm1d (ensure correct shape: [batch_size, num_features])

        # Output layer
        x = self.output_layer(x)

        # Require cards to be in pack. To get pick order, use a ones vector for the pack.
        x = x * pack
        return x


class EmbeddingDraftNet(nn.Module):
    def __init__(self, cardnames, embed_dim=64, hidden_dims=(256, 128), card_features=None):
        """
        Embedding-based draft pick model with optional per-card static features.

        Each card is represented as a concatenation of a learned embed_dim-dimensional
        vector and (optionally) a fixed feature vector loaded from set data.
        Pool and missing-card context are encoded as the mean of those full card
        representations.  A pointwise scoring MLP then scores every card simultaneously.

        Args:
            cardnames (List[str]): Names of cards in the set.
            embed_dim (int): Dimensionality of learned card embeddings.
            hidden_dims (tuple): Hidden layer sizes for the scoring MLP.
            card_features (torch.Tensor or None): (N, F) float tensor of static
                per-card features (e.g. loaded via load_card_features()).  When
                provided they are registered as a non-learned buffer and concatenated
                with the learned embedding to form the full card representation.
        """
        super(EmbeddingDraftNet, self).__init__()
        self.cardnames = cardnames
        self.embed_dim = embed_dim
        N = len(cardnames)

        # One shared embedding table for all cards.
        self.card_embed = nn.Embedding(N, embed_dim)

        # Optional fixed card features (registered as buffer so .to(device) moves them).
        if card_features is not None:
            self.register_buffer("card_features", card_features.float())
            self.n_feat = card_features.shape[-1]
        else:
            self.register_buffer("card_features", None)
            self.n_feat = 0

        # Full per-card representation size.
        card_rep_dim = embed_dim + self.n_feat

        # Scoring head: [card_rep | pool_rep | miss_rep | position(2)] -> 1
        input_dim = card_rep_dim * 3 + 2
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(0.3)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.score_head = nn.Sequential(*layers)

    def _card_reps(self):
        """
        Return full card representations (N, embed_dim + n_feat).
        Concatenates learned embedding with fixed features when available.
        """
        w = self.card_embed.weight                          # (N, D)
        if self.n_feat > 0:
            return torch.cat([w, self.card_features], dim=-1)  # (N, D+F)
        return w

    def _pool_rep(self, indicator, card_reps):
        """
        Mean-pool card representations weighted by a binary indicator.

        Args:
            indicator  (B, N): float — 1 where card is present, 0 elsewhere.
            card_reps  (N, D+F): full card representation matrix.
        Returns:
            (B, D+F) mean representation; zeros for empty sets.
        """
        summed = indicator @ card_reps                          # (B, D+F)
        counts = indicator.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return summed / counts

    def forward(self, x, pack, position, missing):
        """
        Args:
            x        (B, N): pool indicator (float).
            pack     (B, N): pack indicator (float) — mask for available cards.
            position (B, 2): normalised [pack_number/2, pick_number/14].
            missing  (B, N): missing-card indicator (float).
        Returns:
            (B, N) scores, zeroed for cards not in pack.
        """
        B, N = x.shape

        card_reps = self._card_reps()           # (N, D+F)
        R = card_reps.shape[-1]

        # Context vectors: mean of cards in pool / missing (B, D+F).
        pool_rep = self._pool_rep(x, card_reps)
        miss_rep = self._pool_rep(missing, card_reps)

        # Broadcast everything to (B, N, *).
        all_reps  = card_reps.unsqueeze(0).expand(B, N, R)    # (B, N, D+F)
        pool_exp  = pool_rep.unsqueeze(1).expand(B, N, R)     # (B, N, D+F)
        miss_exp  = miss_rep.unsqueeze(1).expand(B, N, R)     # (B, N, D+F)
        pos_exp   = position.unsqueeze(1).expand(B, N, 2)     # (B, N, 2)

        # Score every card in one batched pass: (B, N, 3*(D+F)+2) -> (B, N).
        combined = torch.cat([all_reps, pool_exp, miss_exp, pos_exp], dim=-1)
        scores   = self.score_head(combined).squeeze(-1)

        # Zero out cards not in the pack.
        return scores * pack
