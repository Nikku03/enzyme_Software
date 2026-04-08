from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:
    def winner_v2_feature_dim(
        embedding_dim: int,
        *,
        use_existing_candidate_features: bool = True,
        use_score_gap_features: bool = True,
        use_rank_features: bool = True,
        use_pairwise_features: bool = True,
        use_graph_local_features: bool = True,
        use_3d_local_features: bool = True,
    ) -> int:
        dim = int(embedding_dim)
        if use_existing_candidate_features:
            dim += 1
        if use_score_gap_features:
            dim += 1
        if use_rank_features:
            dim += 1
        if use_pairwise_features:
            dim += int(embedding_dim) * 3
        if use_graph_local_features:
            dim += 1
        if use_3d_local_features:
            dim += 1
        return dim


    def winner_v2_1_feature_dim(
        embedding_dim: int,
        *,
        use_existing_candidate_features: bool = True,
        use_score_gap_features: bool = True,
        use_rank_features: bool = True,
        use_pairwise_features: bool = True,
        use_graph_local_features: bool = True,
        use_3d_local_features: bool = True,
        use_top2_gap_features: bool = True,
        use_normalized_score_features: bool = True,
        use_shortlist_context_features: bool = True,
    ) -> int:
        dim = winner_v2_feature_dim(
            embedding_dim,
            use_existing_candidate_features=use_existing_candidate_features,
            use_score_gap_features=use_score_gap_features,
            use_rank_features=use_rank_features,
            use_pairwise_features=use_pairwise_features,
            use_graph_local_features=use_graph_local_features,
            use_3d_local_features=use_3d_local_features,
        )
        if use_top2_gap_features:
            dim += 1
        if use_normalized_score_features:
            dim += 2
        if use_shortlist_context_features:
            dim += 7
        return dim


    def winner_v2_2_feature_dim(
        embedding_dim: int,
        *,
        use_existing_candidate_features: bool = True,
        use_score_gap_features: bool = True,
        use_rank_features: bool = True,
        use_normalized_score_features: bool = True,
        use_pairwise_features: bool = False,
        use_graph_local_features: bool = False,
        use_3d_local_features: bool = False,
        use_extra_candidate_features: bool = False,
    ) -> int:
        dim = int(embedding_dim)
        if use_existing_candidate_features:
            dim += 1
        if use_score_gap_features:
            dim += 1
        if use_rank_features:
            dim += 1
        if use_normalized_score_features:
            dim += 1
        if use_pairwise_features:
            dim += int(embedding_dim) * 3
        if use_graph_local_features:
            dim += 1
        if use_3d_local_features:
            dim += 1
        if use_extra_candidate_features:
            dim += 2
        return dim


    class ShortlistHead(nn.Module):
        """Scalar shortlist scorer over per-atom embeddings."""

        def __init__(
            self,
            embedding_dim: int,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.hidden_dim = max(1, int(hidden_dim or self.embedding_dim))
            self.net = nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, atom_features):
            if int(atom_features.size(-1)) != int(self.embedding_dim):
                raise ValueError(
                    f"Expected shortlist atom features with dim {self.embedding_dim}, "
                    f"got {int(atom_features.size(-1))}"
                )
            return self.net(atom_features)


    class WinnerHead(nn.Module):
        """Shortlist-local winner scorer over candidate feature rows."""

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.hidden_dim = max(1, int(hidden_dim or self.feature_dim))
            self.net = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, candidate_features):
            if int(candidate_features.size(-1)) != int(self.feature_dim):
                raise ValueError(
                    f"Expected winner candidate features with dim {self.feature_dim}, "
                    f"got {int(candidate_features.size(-1))}"
                )
            return self.net(candidate_features)


    class WinnerHeadV2(nn.Module):
        """Winner scorer over frozen-shortlist candidate features."""

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.hidden_dim = max(1, int(hidden_dim or self.feature_dim))
            self.net = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, candidate_features):
            if int(candidate_features.size(-1)) != int(self.feature_dim):
                raise ValueError(
                    f"Expected winner v2 candidate features with dim {self.feature_dim}, "
                    f"got {int(candidate_features.size(-1))}"
                )
            return self.net(candidate_features)


    class WinnerHeadV2_1(nn.Module):
        """Winner scorer over frozen-shortlist candidate features with richer context."""

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.hidden_dim = max(1, int(hidden_dim or self.feature_dim))
            self.net = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, candidate_features):
            if int(candidate_features.size(-1)) != int(self.feature_dim):
                raise ValueError(
                    f"Expected winner v2.1 candidate features with dim {self.feature_dim}, "
                    f"got {int(candidate_features.size(-1))}"
                )
            return self.net(candidate_features)


    class WinnerHeadV2_2(nn.Module):
        """Winner scorer over frozen-shortlist candidate features with controlled complexity."""

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.hidden_dim = max(1, int(hidden_dim or self.feature_dim))
            self.net = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, candidate_features):
            if int(candidate_features.size(-1)) != int(self.feature_dim):
                raise ValueError(
                    f"Expected winner v2.2 candidate features with dim {self.feature_dim}, "
                    f"got {int(candidate_features.size(-1))}"
                )
            return self.net(candidate_features)
else:  # pragma: no cover
    def winner_v2_feature_dim(*args, **kwargs):
        require_torch()

    def winner_v2_1_feature_dim(*args, **kwargs):
        require_torch()

    def winner_v2_2_feature_dim(*args, **kwargs):
        require_torch()

    class ShortlistHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class WinnerHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class WinnerHeadV2:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class WinnerHeadV2_1:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class WinnerHeadV2_2:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
