from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


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


    def winner_v2_3_feature_dim(
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
        return winner_v2_2_feature_dim(
            embedding_dim,
            use_existing_candidate_features=use_existing_candidate_features,
            use_score_gap_features=use_score_gap_features,
            use_rank_features=use_rank_features,
            use_normalized_score_features=use_normalized_score_features,
            use_pairwise_features=use_pairwise_features,
            use_graph_local_features=use_graph_local_features,
            use_3d_local_features=use_3d_local_features,
            use_extra_candidate_features=use_extra_candidate_features,
        )


    def winner_v2_context_feature_dim(
        embedding_dim: int,
        *,
        use_relative_top_candidate_features: bool = True,
        use_local_competition_features: bool = True,
        use_geometry_proxy_features: bool = True,
    ) -> int:
        dim = winner_v2_feature_dim(
            embedding_dim,
            use_existing_candidate_features=True,
            use_score_gap_features=True,
            use_rank_features=True,
            use_pairwise_features=True,
            use_graph_local_features=True,
            use_3d_local_features=True,
        )
        if use_relative_top_candidate_features:
            dim += 1
        if use_local_competition_features:
            dim += 4
        if use_geometry_proxy_features:
            dim += 1
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


    class WinnerHeadV2_3(nn.Module):
        """Winner scorer over frozen-shortlist candidate features with compact defaults."""

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
                    f"Expected winner v2.3 candidate features with dim {self.feature_dim}, "
                    f"got {int(candidate_features.size(-1))}"
                )
            return self.net(candidate_features)


    class WinnerHeadV2Context(nn.Module):
        """Winner scorer with compact source-context augmentation over v2-style candidate features."""

        def __init__(
            self,
            candidate_feature_dim: int,
            *,
            source_vocab_size: int,
            source_embedding_dim: int = 8,
            use_source_features: bool = True,
            use_source_bias: bool = False,
            use_hard_source_indicator: bool = True,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.candidate_feature_dim = int(candidate_feature_dim)
            self.use_source_features = bool(use_source_features)
            self.use_source_bias = bool(use_source_bias)
            self.use_hard_source_indicator = bool(use_hard_source_indicator)
            self.source_vocab_size = max(1, int(source_vocab_size))
            self.source_embedding_dim = max(1, int(source_embedding_dim))
            self.source_embedding = (
                nn.Embedding(self.source_vocab_size, self.source_embedding_dim)
                if self.use_source_features
                else None
            )
            self.source_bias = (
                nn.Embedding(self.source_vocab_size, 1)
                if self.use_source_bias
                else None
            )
            total_dim = self.candidate_feature_dim
            if self.use_source_features:
                total_dim += self.source_embedding_dim
            if self.use_hard_source_indicator:
                total_dim += 1
            self.feature_dim = int(total_dim)
            self.hidden_dim = max(1, int(hidden_dim or self.feature_dim))
            self.net = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, candidate_features, *, source_indices=None, hard_source_indicator=None):
            if int(candidate_features.size(-1)) != int(self.candidate_feature_dim):
                raise ValueError(
                    f"Expected winner context candidate features with dim {self.candidate_feature_dim}, "
                    f"got {int(candidate_features.size(-1))}"
                )
            parts = [candidate_features]
            if self.use_source_features:
                if source_indices is None:
                    raise ValueError("source_indices are required when winner context source features are enabled")
                parts.append(self.source_embedding(source_indices.view(-1).long()))
            if self.use_hard_source_indicator:
                if hard_source_indicator is None:
                    raise ValueError(
                        "hard_source_indicator is required when winner context hard-source indicator is enabled"
                    )
                parts.append(hard_source_indicator.view(-1, 1).to(dtype=candidate_features.dtype))
            combined = torch.cat(parts, dim=-1)
            if int(combined.size(-1)) != int(self.feature_dim):
                raise ValueError(
                    f"Expected combined winner context features with dim {self.feature_dim}, "
                    f"got {int(combined.size(-1))}"
                )
            logits = self.net(combined)
            if self.use_source_bias:
                if source_indices is None:
                    raise ValueError("source_indices are required when winner context source bias is enabled")
                logits = logits + self.source_bias(source_indices.view(-1).long())
            return logits
else:  # pragma: no cover
    def winner_v2_feature_dim(*args, **kwargs):
        require_torch()

    def winner_v2_1_feature_dim(*args, **kwargs):
        require_torch()

    def winner_v2_2_feature_dim(*args, **kwargs):
        require_torch()

    def winner_v2_3_feature_dim(*args, **kwargs):
        require_torch()

    def winner_v2_context_feature_dim(*args, **kwargs):
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

    class WinnerHeadV2_3:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class WinnerHeadV2Context:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
