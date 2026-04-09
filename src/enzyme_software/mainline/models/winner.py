from __future__ import annotations

from enzyme_software.liquid_nn_v2.model.two_head_shortlist_winner import WinnerHead, winner_v2_feature_dim


def small_local_winner_feature_dim(embedding_dim: int) -> int:
    return int(
        winner_v2_feature_dim(
            int(embedding_dim),
            use_existing_candidate_features=True,
            use_score_gap_features=True,
            use_rank_features=True,
            use_pairwise_features=False,
            use_graph_local_features=True,
            use_3d_local_features=True,
        )
    )


def build_small_local_winner_head(
    embedding_dim: int,
    *,
    hidden_dim: int | None = None,
    dropout: float = 0.1,
) -> WinnerHead:
    return WinnerHead(
        small_local_winner_feature_dim(int(embedding_dim)),
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
