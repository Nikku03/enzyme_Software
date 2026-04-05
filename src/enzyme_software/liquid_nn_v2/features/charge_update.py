from __future__ import annotations

import numpy as np


_CHI0 = {
    1: 2.20,
    6: 2.55,
    7: 3.04,
    8: 3.44,
    9: 3.98,
    15: 2.19,
    16: 2.58,
    17: 3.16,
    35: 2.96,
    53: 2.66,
}

_HARDNESS = {
    1: 12.0,
    6: 10.0,
    7: 12.5,
    8: 13.5,
    9: 15.0,
    15: 8.5,
    16: 8.0,
    17: 10.0,
    35: 8.0,
    53: 6.5,
}


def update_local_charges_eem(
    atom_coords: np.ndarray | None,
    atomic_numbers: np.ndarray,
    initial_charges: np.ndarray,
    *,
    ridge: float = 1.0e-3,
    damping_delta: float = 0.25,
) -> np.ndarray:
    atom_z = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    q0 = np.asarray(initial_charges, dtype=np.float32).reshape(-1)
    num_atoms = int(atom_z.size)
    if atom_coords is None or num_atoms <= 1:
        return q0.reshape(-1, 1).astype(np.float32)

    coords = np.asarray(atom_coords, dtype=np.float32).reshape(num_atoms, 3)
    chi = np.asarray([_CHI0.get(int(z), 2.5) for z in atom_z], dtype=np.float32)
    eta = np.asarray([_HARDNESS.get(int(z), 9.0) for z in atom_z], dtype=np.float32)
    total_charge = float(q0.sum())

    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1).astype(np.float32)
    coulomb = 1.0 / np.sqrt(np.square(distances) + (float(damping_delta) ** 2))
    np.fill_diagonal(coulomb, 0.0)

    a = np.zeros((num_atoms + 1, num_atoms + 1), dtype=np.float64)
    a[:num_atoms, :num_atoms] = coulomb
    a[np.arange(num_atoms), np.arange(num_atoms)] = (2.0 * eta).astype(np.float64) + float(ridge)
    a[:num_atoms, -1] = -1.0
    a[-1, :num_atoms] = 1.0

    b = np.zeros((num_atoms + 1,), dtype=np.float64)
    b[:num_atoms] = -chi.astype(np.float64)
    b[-1] = total_charge

    try:
        solution = np.linalg.solve(a, b)
        charges = solution[:num_atoms]
    except np.linalg.LinAlgError:
        charges = q0.astype(np.float64)
    charges = np.clip(charges, -2.5, 2.5).astype(np.float32)
    return charges.reshape(-1, 1)
