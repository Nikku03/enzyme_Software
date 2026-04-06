from __future__ import annotations

from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:
    CYP3A4_STATES: List[Dict[str, object]] = [
        {
            "name": "closed_restrictive",
            "heme_center": [-0.35, 0.20, -0.10],
            "oxo_axis": [-0.08, 0.16, 0.98],
            "gate_openness": 0.25,
            "pocket_radius": 6.50,
            "phe_filter_strength": 0.90,
            "channel_bias": [0.82, 0.18, 0.10],
            "allosteric_pressure": 0.00,
        },
        {
            "name": "open_productive",
            "heme_center": [0.00, -0.15, 0.12],
            "oxo_axis": [0.12, -0.05, 0.99],
            "gate_openness": 0.75,
            "pocket_radius": 8.00,
            "phe_filter_strength": 0.50,
            "channel_bias": [0.90, 0.60, 0.40],
            "allosteric_pressure": 0.20,
        },
        {
            "name": "expanded_stacked",
            "heme_center": [0.28, 0.18, 0.25],
            "oxo_axis": [0.26, 0.10, 0.96],
            "gate_openness": 0.90,
            "pocket_radius": 9.20,
            "phe_filter_strength": 0.35,
            "channel_bias": [0.68, 0.84, 0.72],
            "allosteric_pressure": 0.80,
        },
    ]


    def get_cyp3a4_states(*, device=None, dtype=None) -> List[Dict[str, torch.Tensor | float | str]]:
        states: List[Dict[str, torch.Tensor | float | str]] = []
        for state in CYP3A4_STATES:
            axis = torch.as_tensor(state["oxo_axis"], device=device, dtype=dtype or torch.float32)
            axis = axis / axis.norm().clamp_min(1.0e-6)
            states.append(
                {
                    "name": str(state["name"]),
                    "heme_center": torch.as_tensor(state["heme_center"], device=device, dtype=dtype or torch.float32),
                    "oxo_axis": axis,
                    "gate_openness": float(state["gate_openness"]),
                    "pocket_radius": float(state["pocket_radius"]),
                    "phe_filter_strength": float(state["phe_filter_strength"]),
                    "channel_bias": torch.as_tensor(state["channel_bias"], device=device, dtype=dtype or torch.float32),
                    "allosteric_pressure": float(state["allosteric_pressure"]),
                }
            )
        return states


    def _safe_atom_coords(atom_coords: Optional[torch.Tensor], rows: int, *, device, dtype) -> torch.Tensor:
        if atom_coords is None:
            return torch.zeros(rows, 3, device=device, dtype=dtype)
        coords = atom_coords.to(device=device, dtype=dtype)
        if coords.ndim != 2 or int(coords.size(-1)) != 3 or int(coords.size(0)) != int(rows):
            return torch.zeros(rows, 3, device=device, dtype=dtype)
        return torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


    def _optional_feature(value, rows: int, width: int, *, device, dtype) -> torch.Tensor:
        if width <= 0:
            return torch.zeros(rows, 0, device=device, dtype=dtype)
        if value is None:
            return torch.zeros(rows, width, device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype)
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        if int(out.size(0)) != int(rows):
            return torch.zeros(rows, width, device=device, dtype=dtype)
        if int(out.size(-1)) == int(width):
            return out
        if int(out.size(-1)) > int(width):
            return out[:, :width]
        return torch.nn.functional.pad(out, (0, int(width) - int(out.size(-1))))


    def _molecule_masks(batch_index: torch.Tensor, num_molecules: int) -> List[torch.Tensor]:
        return [(batch_index == idx) for idx in range(num_molecules)]


    def _masked_edge_feature_means(
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        atom_mask: torch.Tensor,
        *,
        width: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if edge_index is None or edge_attr is None or not getattr(edge_index, "numel", lambda: 0)() or not getattr(edge_attr, "numel", lambda: 0)():
            return torch.zeros(width, device=device, dtype=dtype)
        edge_index_t = edge_index.to(device=device, dtype=torch.long)
        edge_attr_t = edge_attr.to(device=device, dtype=dtype)
        if edge_attr_t.ndim == 1:
            edge_attr_t = edge_attr_t.unsqueeze(-1)
        edge_width = min(width, int(edge_attr_t.size(-1)))
        src = edge_index_t[0]
        dst = edge_index_t[1]
        mask = atom_mask[src] & atom_mask[dst]
        if not bool(mask.any()):
            return torch.zeros(width, device=device, dtype=dtype)
        mean = edge_attr_t[mask, :edge_width].mean(dim=0)
        if edge_width == width:
            return mean
        return torch.nn.functional.pad(mean, (0, width - edge_width))


    def _reaction_direction_candidates(
        atom_coords: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        batch_index: torch.Tensor,
        *,
        max_directions: int = 4,
    ) -> torch.Tensor:
        rows = int(atom_coords.size(0))
        device = atom_coords.device
        dtype = atom_coords.dtype
        direction = torch.zeros(rows, max_directions, 3, device=device, dtype=dtype)
        if edge_index is not None and getattr(edge_index, "numel", lambda: 0)():
            edge_arr = edge_index.to(device=device, dtype=torch.long)
            for atom_idx in range(rows):
                nbr_mask = edge_arr[0] == atom_idx
                nbrs = edge_arr[1, nbr_mask]
                nbrs = nbrs[(nbrs >= 0) & (nbrs < rows)]
                if nbrs.numel() > 0:
                    limit = min(max_directions - 1, int(nbrs.numel()))
                    for slot, nbr_idx in enumerate(nbrs[:limit]):
                        direction[atom_idx, slot] = atom_coords[atom_idx] - atom_coords[int(nbr_idx)]
                    nbr_centroid = atom_coords[nbrs].mean(dim=0)
                    direction[atom_idx, max_directions - 1] = atom_coords[atom_idx] - nbr_centroid
        for mol_idx in range(int(batch_index.max().item()) + 1 if batch_index.numel() else 0):
            mol_mask = batch_index == mol_idx
            if not bool(mol_mask.any()):
                continue
            mol_centroid = atom_coords[mol_mask].mean(dim=0, keepdim=True)
            missing = mol_mask & (direction.norm(dim=-1).amax(dim=-1) <= 1.0e-6)
            if bool(missing.any()):
                direction[missing, 0, :] = atom_coords[missing] - mol_centroid
        direction_norm = direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return direction / direction_norm


    def score_molecule_against_states(
        atom_coords: torch.Tensor,
        batch_index: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        phase5_atom_features: Optional[torch.Tensor],
        local_chem_features: Optional[torch.Tensor],
        states: List[Dict[str, torch.Tensor | float | str]],
        *,
        state_temperature: float,
        min_state_weight: float,
    ) -> Dict[str, torch.Tensor]:
        rows = int(atom_coords.size(0))
        device = atom_coords.device
        dtype = atom_coords.dtype
        num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        z = torch.zeros(num_molecules, 7, device=device, dtype=dtype)
        phase5 = _optional_feature(phase5_atom_features, rows, 18, device=device, dtype=dtype)
        local_chem = _optional_feature(local_chem_features, rows, 11, device=device, dtype=dtype)
        access_score = phase5[:, 8:9] if phase5.numel() else torch.zeros(rows, 1, device=device, dtype=dtype)
        crowding = local_chem[:, 4:5] if local_chem.numel() else torch.zeros(rows, 1, device=device, dtype=dtype)
        charge_delta = local_chem[:, 5:6] if local_chem.numel() else torch.zeros(rows, 1, device=device, dtype=dtype)
        for mol_idx, mol_mask in enumerate(_molecule_masks(batch_index, num_molecules)):
            if not bool(mol_mask.any()):
                continue
            mol_coords = atom_coords[mol_mask]
            num_atoms = float(mol_mask.sum().item())
            bbox_span = (mol_coords.max(dim=0).values - mol_coords.min(dim=0).values).norm()
            topk = min(6, int(mol_coords.size(0)))
            local_access = access_score[mol_mask].view(-1)
            exposed_order = torch.argsort(local_access, descending=True)[:topk]
            exposed_span = (
                (mol_coords[exposed_order].max(dim=0).values - mol_coords[exposed_order].min(dim=0).values).norm()
                if topk > 1
                else torch.tensor(0.0, device=device, dtype=dtype)
            )
            mean_access = access_score[mol_mask].mean()
            mean_crowding = crowding[mol_mask].mean().clamp(0.0, 1.0)
            max_charge_delta = charge_delta[mol_mask].abs().max().clamp(0.0, 2.0) / 2.0
            edge_means = _masked_edge_feature_means(
                edge_index,
                edge_attr,
                mol_mask,
                width=7,
                device=device,
                dtype=dtype,
            )
            aromatic_frac = edge_means[3].clamp(0.0, 1.0)
            rotatable_frac = edge_means[6].clamp(0.0, 1.0)
            z[mol_idx] = torch.stack(
                [
                    torch.log1p(torch.tensor(num_atoms, device=device, dtype=dtype)) / 4.0,
                    bbox_span / 12.0,
                    exposed_span / 10.0,
                    rotatable_frac.clamp(0.0, 1.0),
                    aromatic_frac.clamp(0.0, 1.0),
                    mean_access.clamp(0.0, 1.0),
                    (0.55 * max_charge_delta + 0.45 * (1.0 - mean_crowding)).clamp(0.0, 1.0),
                ]
            )
        coeff = torch.as_tensor(
            [
                [-1.10, -1.05, -0.75, -0.10, 0.05, -0.15, -0.05],
                [0.05, 0.30, 0.18, 0.18, 0.02, 0.32, 0.08],
                [0.95, 1.05, 0.92, 0.28, 0.16, 0.20, 0.24],
            ],
            device=device,
            dtype=dtype,
        )
        bias = torch.as_tensor([0.55, 0.30, -0.30], device=device, dtype=dtype)
        logits = z @ coeff.t() + bias
        weights = torch.softmax(logits / max(float(state_temperature), 1.0e-3), dim=-1)
        floor = max(0.0, min(float(min_state_weight), 0.30))
        if floor > 0.0:
            weights = weights.clamp_min(floor)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return {
            "molecule_features": z,
            "state_logits": logits,
            "state_weights": weights,
        }


    def compute_proximity_score(
        atom_coords: torch.Tensor,
        state_heme_center: torch.Tensor,
        *,
        distance_center: float,
        distance_sigma: float,
    ) -> torch.Tensor:
        distance = (atom_coords - state_heme_center.view(1, 3)).norm(dim=-1, keepdim=True)
        return torch.exp(-torch.square(distance - float(distance_center)) / (2.0 * (float(distance_sigma) ** 2))).clamp(0.0, 1.0)


    def compute_orientation_score(
        reaction_direction: torch.Tensor,
        state_axis: torch.Tensor,
        *,
        alpha: float,
    ) -> torch.Tensor:
        cosine = torch.matmul(reaction_direction, state_axis.view(3, 1)).squeeze(-1).clamp(-1.0, 1.0)
        best = torch.relu(cosine).amax(dim=1, keepdim=True)
        return best.pow(float(alpha)).clamp(0.0, 1.0)


    def compute_access_score(
        atom_coords: torch.Tensor,
        state_heme_center: torch.Tensor,
        phase5_atom_features: Optional[torch.Tensor],
        local_chem_features: Optional[torch.Tensor],
        state: Dict[str, torch.Tensor | float | str],
        *,
        path_lambda: float,
        crowding_lambda: float,
        radial_lambda: float,
        filter_lambda: float,
    ) -> Dict[str, torch.Tensor]:
        rows = int(atom_coords.size(0))
        device = atom_coords.device
        dtype = atom_coords.dtype
        phase5 = _optional_feature(phase5_atom_features, rows, 18, device=device, dtype=dtype)
        local_chem = _optional_feature(local_chem_features, rows, 11, device=device, dtype=dtype)
        rel = atom_coords - state_heme_center.view(1, 3)
        dist = rel.norm(dim=-1, keepdim=True)
        unit = rel / dist.clamp_min(1.0e-6)
        pocket_radius = max(1.0, float(state["pocket_radius"]))
        gate = float(state["gate_openness"])
        phe_filter = float(state["phe_filter_strength"])
        allosteric = float(state["allosteric_pressure"])
        channel_bias = state["channel_bias"]
        solvent_pref = torch.sigmoid(unit[:, 2:3])
        path_2a_2e = torch.sigmoid(unit[:, 0:1])
        path_2b_2c = torch.sigmoid(unit[:, 1:2])
        channel_pref = (
            float(channel_bias[0].item()) * solvent_pref
            + float(channel_bias[1].item()) * path_2a_2e
            + float(channel_bias[2].item()) * path_2b_2c
        ) / max(float(channel_bias.sum().item()), 1.0e-6)
        blockage = phase5[:, 9:10] if phase5.numel() else torch.zeros(rows, 1, device=device, dtype=dtype)
        crowding = local_chem[:, 4:5] if local_chem.numel() else torch.zeros(rows, 1, device=device, dtype=dtype)
        radial_overflow = torch.relu((dist - pocket_radius) / pocket_radius)
        path_cost = (
            (dist / pocket_radius)
            + 0.75 * radial_overflow
            + 0.35 * (1.0 - channel_pref)
            + 0.25 * allosteric * (dist / pocket_radius)
        )
        filter_penalty = (phe_filter * (0.60 * blockage + 0.40 * crowding)).clamp(0.0, 1.5)
        normalized_distance = (dist / pocket_radius).clamp(0.0, 3.0)
        channel_multiplier = channel_pref.clamp(0.20, 1.25)
        score = torch.exp(
            -(float(path_lambda) * normalized_distance)
            - (float(radial_lambda) * radial_overflow.clamp(0.0, 2.0))
            - (float(filter_lambda) * filter_penalty)
            - (float(crowding_lambda) * crowding.clamp(0.0, 2.0))
        ) * gate * channel_multiplier
        return {
            "score": score.clamp(0.0, 1.0),
            "path_cost": path_cost,
            "crowding": crowding,
            "channel_pref": channel_pref,
            "radial_overflow": radial_overflow,
            "filter_penalty": filter_penalty,
            "normalized_distance": normalized_distance,
        }


    def compute_electronic_score(
        local_chem_features: Optional[torch.Tensor],
        bde_values: Optional[torch.Tensor],
        *,
        beta_bde: float = 0.55,
        beta_charge: float = 0.20,
        beta_etn: float = 0.25,
    ) -> torch.Tensor:
        if local_chem_features is None:
            require_torch()
        rows = int(local_chem_features.size(0))
        device = local_chem_features.device
        dtype = local_chem_features.dtype
        local_chem = _optional_feature(local_chem_features, rows, 11, device=device, dtype=dtype)
        bde = _optional_feature(bde_values, rows, 1, device=device, dtype=dtype)
        bde_score = ((430.0 - bde) / 120.0).clamp(-2.0, 2.0)
        charge_delta = local_chem[:, 5:6].clamp(-2.0, 2.0)
        etn = ((2.0 * local_chem[:, 7:8]) - 1.0).clamp(-1.0, 1.0)
        return (beta_bde * bde_score) + (beta_charge * charge_delta) + (beta_etn * etn)


    def build_state_conditioned_features(
        atom_coords: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        batch_index: torch.Tensor,
        phase5_atom_features: Optional[torch.Tensor],
        local_chem_features: Optional[torch.Tensor],
        bde_values: Optional[torch.Tensor],
        states: List[Dict[str, torch.Tensor | float | str]],
        state_weights: torch.Tensor,
        *,
        distance_center: float,
        distance_sigma: float,
        orientation_alpha: float,
        access_path_lambda: float,
        access_crowding_lambda: float,
        access_radial_lambda: float,
        access_filter_lambda: float,
    ) -> Dict[str, torch.Tensor]:
        rows = int(atom_coords.size(0))
        reaction_dir = _reaction_direction_candidates(atom_coords, edge_index, batch_index)
        proximity_parts = []
        orientation_parts = []
        access_parts = []
        path_cost_parts = []
        channel_pref_parts = []
        radial_overflow_parts = []
        filter_penalty_parts = []
        normalized_distance_parts = []
        electronic_raw = compute_electronic_score(local_chem_features, bde_values)
        electronic_effective_parts = []
        for state in states:
            proximity = compute_proximity_score(
                atom_coords,
                state["heme_center"],
                distance_center=distance_center,
                distance_sigma=distance_sigma,
            )
            orientation = compute_orientation_score(
                reaction_dir,
                state["oxo_axis"],
                alpha=orientation_alpha,
            )
            access_payload = compute_access_score(
                atom_coords,
                state["heme_center"],
                phase5_atom_features,
                local_chem_features,
                state,
                path_lambda=access_path_lambda,
                crowding_lambda=access_crowding_lambda,
                radial_lambda=access_radial_lambda,
                filter_lambda=access_filter_lambda,
            )
            proximity_parts.append(proximity)
            orientation_parts.append(orientation)
            access_parts.append(access_payload["score"])
            path_cost_parts.append(access_payload["path_cost"])
            channel_pref_parts.append(access_payload["channel_pref"])
            radial_overflow_parts.append(access_payload["radial_overflow"])
            filter_penalty_parts.append(access_payload["filter_penalty"])
            normalized_distance_parts.append(access_payload["normalized_distance"])
            electronic_effective_parts.append(electronic_raw * access_payload["score"])
        proximity = torch.cat(proximity_parts, dim=1)
        orientation = torch.cat(orientation_parts, dim=1)
        access = torch.cat(access_parts, dim=1)
        path_cost = torch.cat(path_cost_parts, dim=1)
        channel_pref = torch.cat(channel_pref_parts, dim=1)
        radial_overflow = torch.cat(radial_overflow_parts, dim=1)
        filter_penalty = torch.cat(filter_penalty_parts, dim=1)
        normalized_distance = torch.cat(normalized_distance_parts, dim=1)
        electronic = torch.cat(electronic_effective_parts, dim=1)
        state_weight_atoms = state_weights[batch_index]
        weighted_gate = (state_weight_atoms * torch.as_tensor([float(s["gate_openness"]) for s in states], device=atom_coords.device, dtype=atom_coords.dtype).view(1, -1)).sum(dim=1, keepdim=True)
        weighted_pocket_radius = (
            state_weight_atoms
            * torch.as_tensor([float(s["pocket_radius"]) for s in states], device=atom_coords.device, dtype=atom_coords.dtype).view(1, -1)
        ).sum(dim=1, keepdim=True)
        return {
            "proximity": proximity,
            "orientation": orientation,
            "access": access,
            "path_cost": path_cost,
            "electronic": electronic,
            "electronic_raw": electronic_raw,
            "reaction_direction": reaction_dir,
            "weighted_gate": weighted_gate,
            "weighted_pocket_radius": weighted_pocket_radius,
            "channel_pref": channel_pref,
            "radial_overflow": radial_overflow,
            "filter_penalty": filter_penalty,
            "normalized_distance": normalized_distance,
        }


    def rescore_candidates(
        frozen_scores: torch.Tensor,
        state_features: Dict[str, torch.Tensor],
        state_weights: torch.Tensor,
        batch_index: torch.Tensor,
        *,
        proximity_weight: float,
        orientation_weight: float,
        access_weight: float,
        electronic_weight: float,
        learned_weight: float,
        aggregation_temperature: float,
    ) -> Dict[str, torch.Tensor]:
        learned = frozen_scores.view(-1, 1)
        mechanistic = (
            float(proximity_weight) * state_features["proximity"]
            + float(orientation_weight) * state_features["orientation"]
            + float(access_weight) * state_features["access"]
            + float(electronic_weight) * state_features["electronic"]
        )
        gate = torch.sigmoid(
            1.35 * state_features["proximity"]
            + 0.95 * state_features["orientation"]
            + 1.55 * state_features["access"]
            - 1.10 * state_features["filter_penalty"]
        )
        state_scores = (float(learned_weight) * learned) + (gate * mechanistic)
        log_weights = torch.log(state_weights.clamp_min(1.0e-6))[batch_index]
        agg_temp = max(float(aggregation_temperature), 1.0e-3)
        final_scores = agg_temp * torch.logsumexp(log_weights + (state_scores / agg_temp), dim=1, keepdim=True)
        return {
            "state_scores": state_scores,
            "mechanistic": mechanistic,
            "gate": gate,
            "final_scores": final_scores,
        }

else:  # pragma: no cover
    CYP3A4_STATES = []

    def get_cyp3a4_states(*args, **kwargs):
        require_torch()

    def score_molecule_against_states(*args, **kwargs):
        require_torch()

    def build_state_conditioned_features(*args, **kwargs):
        require_torch()

    def compute_proximity_score(*args, **kwargs):
        require_torch()

    def compute_orientation_score(*args, **kwargs):
        require_torch()

    def compute_access_score(*args, **kwargs):
        require_torch()

    def compute_electronic_score(*args, **kwargs):
        require_torch()

    def rescore_candidates(*args, **kwargs):
        require_torch()
