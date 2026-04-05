from __future__ import annotations

import numpy as np


def _solve_etn_yield(hamiltonian: np.ndarray, rho0: np.ndarray) -> np.ndarray:
    n = int(hamiltonian.shape[0])
    ident = np.eye(n, dtype=np.complex128)
    lhs = (-1.0j * np.kron(ident, hamiltonian)) + (1.0j * np.kron(np.conjugate(hamiltonian), ident))
    lhs = lhs + (1.0e-4 * np.eye(n * n, dtype=np.complex128))
    rhs = -rho0.reshape(-1)
    try:
        vec_x = np.linalg.solve(lhs, rhs)
        x = vec_x.reshape(n, n)
        x = 0.5 * (x + np.conjugate(x.T))
        return np.real(np.diag(x)).astype(np.float32)
    except np.linalg.LinAlgError:
        return np.real(np.diag(rho0)).astype(np.float32)


def compute_etn_prior_scores(
    atom_coords: np.ndarray | None,
    updated_charges: np.ndarray,
    field_score: np.ndarray,
    access_proxy: np.ndarray,
    crowding: np.ndarray,
    bde_values: np.ndarray,
    edge_index: np.ndarray | None,
    *,
    max_sites: int = 32,
) -> dict[str, np.ndarray]:
    charges = np.asarray(updated_charges, dtype=np.float32).reshape(-1)
    field = np.asarray(field_score, dtype=np.float32).reshape(-1)
    access = np.asarray(access_proxy, dtype=np.float32).reshape(-1)
    crowd = np.asarray(crowding, dtype=np.float32).reshape(-1)
    bde = np.asarray(bde_values, dtype=np.float32).reshape(-1)
    num_atoms = int(charges.size)
    zeros = np.zeros((num_atoms, 1), dtype=np.float32)
    if num_atoms == 0:
        return {"prior_score": zeros, "energy": zeros, "gamma": zeros, "yield": zeros}

    bde_norm = np.clip((bde - 250.0) / 250.0, 0.0, 1.5)
    energy = (0.65 * bde_norm) + (0.20 * charges) + (0.25 * field)
    gamma = np.log1p(np.exp((0.90 * access) + (0.60 * field) - (0.70 * crowd))).astype(np.float32)
    gamma = np.clip(gamma, 1.0e-4, 3.0)

    if atom_coords is None or num_atoms <= 1:
        prior = 1.0 / (1.0 + np.exp(energy - gamma))
        return {
            "prior_score": prior.reshape(-1, 1).astype(np.float32),
            "energy": energy.reshape(-1, 1).astype(np.float32),
            "gamma": gamma.reshape(-1, 1).astype(np.float32),
            "yield": prior.reshape(-1, 1).astype(np.float32),
        }

    coords = np.asarray(atom_coords, dtype=np.float32).reshape(num_atoms, 3)
    ranking = np.argsort(-(field + access - crowd))[: min(max_sites, num_atoms)]
    sub_coords = coords[ranking]
    sub_energy = energy[ranking]
    sub_gamma = gamma[ranking]
    deltas = sub_coords[:, None, :] - sub_coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1).astype(np.float32)
    near = (distances < 3.5).astype(np.float32)
    np.fill_diagonal(near, 0.0)
    coupling = 0.18 * np.exp(-distances / 2.5) * near
    if edge_index is not None and getattr(edge_index, "size", 0):
        edges = np.asarray(edge_index, dtype=np.int64)
        local_lookup = {int(atom_idx): local_idx for local_idx, atom_idx in enumerate(ranking.tolist())}
        for src, dst in edges.T.tolist():
            i = local_lookup.get(int(src))
            j = local_lookup.get(int(dst))
            if i is not None and j is not None:
                coupling[i, j] = max(float(coupling[i, j]), 0.25)

    h = np.diag(sub_energy.astype(np.complex128) - (1.0j * sub_gamma.astype(np.complex128))) + coupling.astype(np.complex128)
    seed = np.exp(-(sub_energy - sub_gamma))
    seed = seed / np.clip(seed.sum(), 1.0e-6, None)
    rho0 = np.diag(seed.astype(np.complex128))
    diag_x = _solve_etn_yield(h, rho0)
    yields = 2.0 * sub_gamma * np.clip(diag_x, 0.0, None)
    yields = yields / np.clip(yields.max(), 1.0e-6, None)

    full_yield = np.zeros((num_atoms,), dtype=np.float32)
    full_yield[ranking] = yields.astype(np.float32)
    prior = np.clip(0.65 * full_yield + 0.35 * (1.0 / (1.0 + np.exp(energy - gamma))), 0.0, 1.0)
    return {
        "prior_score": prior.reshape(-1, 1).astype(np.float32),
        "energy": energy.reshape(-1, 1).astype(np.float32),
        "gamma": gamma.reshape(-1, 1).astype(np.float32),
        "yield": full_yield.reshape(-1, 1).astype(np.float32),
    }
