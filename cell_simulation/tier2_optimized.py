"""
Tier 2 optimized: closer to real deployment speeds.

Two adjustments from tier3_cascade.py:
1. Add a LINEAR surrogate (Ridge regression) — will be fast and still accurate
   when dynamics are near-linear around operating point.
2. Add BATCHED inference — the real use case is "1000 knockouts" or 
   "1000 environmental perturbations," not one at a time.
"""

import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from tier3_cascade import (
    Tier3MechanisticODE,
    Tier2DynamicSurrogate,
    generate_training_data,
)


class Tier2Linear:
    """Ridge regression surrogate. Vectorized inference — ultra fast."""
    
    GENE_ORDER = Tier2DynamicSurrogate.GENE_ORDER
    
    def __init__(self, alpha=1.0):
        self.W = None
        self.b = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.alpha = alpha
        self.trained = False
    
    def fit(self, X, Y):
        Xs = self.x_scaler.fit_transform(X)
        Ys = self.y_scaler.fit_transform(Y)
        # Ridge: W = (X^T X + αI)^-1 X^T Y
        Xaug = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        A = Xaug.T @ Xaug + self.alpha * np.eye(Xaug.shape[1])
        A[-1, -1] = 1e-12  # don't regularize bias
        B = Xaug.T @ Ys
        W_aug = np.linalg.solve(A, B)
        self.W = W_aug[:-1]  # (d_in, d_out)
        self.b = W_aug[-1]   # (d_out,)
        self.trained = True
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Batched: X is (N, 17), returns (N, 10)."""
        Xs = self.x_scaler.transform(X)
        Ys_pred = Xs @ self.W + self.b
        return self.y_scaler.inverse_transform(Ys_pred)
    
    @staticmethod
    def build_queries(init_states: np.ndarray, gene_ids: list) -> np.ndarray:
        """init_states: (N, 10), gene_ids: list of N gene strings or None."""
        N = len(gene_ids)
        X = np.zeros((N, 17), dtype=np.float32)
        X[:, :10] = init_states
        for i, gid in enumerate(gene_ids):
            if gid and gid in Tier2Linear.GENE_ORDER:
                X[i, 10 + Tier2Linear.GENE_ORDER.index(gid)] = 1.0
        return X


def main():
    print("=" * 70)
    print("TIER 2 OPTIMIZED: Linear surrogate + batched inference")
    print("=" * 70)
    
    # Generate data once (same as tier3_cascade.py)
    print("\n[1] Generating training data...")
    X_train, Y_train, train_gen_time = generate_training_data(n_samples=1000, seed=0)
    
    print("\n[2] Generating held-out test set...")
    X_test, Y_test, test_gen_time = generate_training_data(n_samples=500, seed=42)
    
    # Fit linear surrogate
    print("\n[3] Training Ridge surrogate...")
    t0 = time.perf_counter()
    t2_lin = Tier2Linear(alpha=1.0)
    t2_lin.fit(X_train, Y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time*1000:.1f} ms")
    
    # Accuracy
    Y_pred = t2_lin.predict_batch(X_test)
    r2_per_met = [r2_score(Y_test[:, j], Y_pred[:, j]) for j in range(10)]
    met_names = ['glc', 'G6P', 'F6P', 'FBP', 'PEP', 'pyr', 'lac', 'ATP', 'ADP', 'NAD']
    
    atp_init_test = X_test[:, 7]
    viable_true = Y_test[:, 7] > 0.3 * atp_init_test
    viable_pred = Y_pred[:, 7] > 0.3 * atp_init_test
    via_acc = (viable_true == viable_pred).mean()
    
    print("\n[4] Accuracy (Ridge linear surrogate):")
    for name, r2 in zip(met_names, r2_per_met):
        print(f"    {name:5s}: R² = {r2:+.3f}")
    print(f"  Viability accuracy: {via_acc*100:.1f}%")
    
    # Timing: batched inference
    print("\n[5] Batched inference timing:")
    for N in [100, 1000, 10000, 100000]:
        # Synthesize queries
        rng = np.random.default_rng(7)
        y0_base = Tier3MechanisticODE.initial_state()
        inits = y0_base * rng.uniform(0.7, 1.3, size=(N, 10))
        gids = [rng.choice(Tier2Linear.GENE_ORDER + [None]) for _ in range(N)]
        X_q = Tier2Linear.build_queries(inits, gids)
        
        # Warm up
        _ = t2_lin.predict_batch(X_q[:10])
        
        t0 = time.perf_counter()
        _ = t2_lin.predict_batch(X_q)
        t2_time = time.perf_counter() - t0
        
        print(f"  N={N:>7d}:  {t2_time*1000:>8.2f} ms  ({t2_time*1e6/N:>8.2f} µs/query)")
    
    # Tier 3 equivalent
    print("\n[6] Tier 3 equivalent speed (measured on 20 runs):")
    sim = Tier3MechanisticODE()
    sim.simulate(t_final=60.0, n_points=20)  # warm up
    t3_times = []
    for _ in range(20):
        sim.reset()
        r = sim.simulate(t_final=60.0, n_points=20)
        t3_times.append(r['time_ms'])
    t3_per = np.mean(t3_times)
    print(f"  Tier 3: {t3_per:.1f} ms/query")
    
    # Speedup
    # Batched inference at N=100k: let's recompute
    rng = np.random.default_rng(7)
    y0_base = Tier3MechanisticODE.initial_state()
    inits = y0_base * rng.uniform(0.7, 1.3, size=(100000, 10))
    gids = [rng.choice(Tier2Linear.GENE_ORDER + [None]) for _ in range(100000)]
    X_q = Tier2Linear.build_queries(inits, gids)
    t0 = time.perf_counter()
    _ = t2_lin.predict_batch(X_q)
    t2_big = time.perf_counter() - t0
    
    t2_per_us = t2_big * 1e6 / 100000
    speedup_per_query = (t3_per * 1000) / t2_per_us  # both in µs
    
    print(f"\n[7] Comparison:")
    print(f"  Tier 3:  {t3_per*1000:>10.0f} µs/query (mechanistic BDF ODE)")
    print(f"  Tier 2:  {t2_per_us:>10.2f} µs/query (batched linear)")
    print(f"\n  Per-query speedup: {speedup_per_query:.0f}x")
    
    # Amortized with 1k training runs
    fixed_cost = train_gen_time + train_time  # mostly training data generation
    # For N queries:
    # Tier 3 only:  N * t3_per (ms)
    # Cascade:     fixed_cost * 1000 + N * t2_per_us / 1000  (ms)
    for N in [1000, 10000, 100000, 1000000]:
        t3_total_ms = N * t3_per
        cascade_total_ms = fixed_cost * 1000 + N * t2_per_us / 1000
        ratio = t3_total_ms / cascade_total_ms
        print(f"\n  N={N:>7d}:  Tier 3={t3_total_ms/1000:>8.0f}s  "
              f"cascade={cascade_total_ms/1000:>8.0f}s  "
              f"speedup={ratio:>6.0f}x")


if __name__ == '__main__':
    main()
