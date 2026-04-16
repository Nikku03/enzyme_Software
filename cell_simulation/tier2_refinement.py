"""
Tier 2: Neural refinement of FBA
=================================

Problem being solved:
  FBA gives 69.5% balanced accuracy, with very poor specificity (50%).
  The 13 errors (9 FN + 4 FP) have patterns:
    - 4 FN have biomass ~98% (non-metabolic essentials FBA can't see)
    - 5 FN have biomass ~30-40% (FBA overestimates redundancy)
    - 4 FP have biomass=0 (FBA underestimates — missing reactions/alternate paths)
  
  Tier 2's job: learn a correction from (FBA features) -> (true essentiality)
  using gradient-boosted trees. Validation MUST be leave-one-out cross-validation
  because we only have 90 labeled genes — any other split leaks.

Honest expectation: this lifts balanced accuracy from 69.5% to ~75-80%.
Anyone claiming >85% on 90 genes with LOO-CV is doing it wrong.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from typing import Tuple

from tier1_fba import Tier1FBA, GENE_ESSENTIALITY


def evaluate_model(model_factory, feats: np.ndarray, y_true: np.ndarray,
                   name: str) -> dict:
    """
    Leave-one-out CV. model_factory is a callable that returns a fresh model.
    This is the ONLY honest validation method for a dataset of N=90.
    """
    loo = LeaveOneOut()
    y_pred = np.zeros_like(y_true)
    
    for train_idx, test_idx in loo.split(feats):
        model = model_factory()
        model.fit(feats[train_idx], y_true[train_idx])
        y_pred[test_idx] = model.predict(feats[test_idx])
    
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    
    acc = (tp + tn) / len(y_true)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    bal = 0.5 * (sens + spec)
    
    return {
        'name': name,
        'accuracy': acc,
        'balanced_accuracy': bal,
        'sensitivity': sens,
        'specificity': spec,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def fmt(r):
    return (f"  {r['name']:30s}  "
            f"acc={r['accuracy']*100:5.1f}%  "
            f"bal={r['balanced_accuracy']*100:5.1f}%  "
            f"sens={r['sensitivity']*100:5.1f}%  "
            f"spec={r['specificity']*100:5.1f}%  "
            f"TP={r['tp']:2d} FP={r['fp']:2d} TN={r['tn']:2d} FN={r['fn']:2d}")


def main():
    print("=" * 70)
    print("TIER 2: Neural refinement with LOO-CV")
    print("=" * 70)
    
    # Build Tier 1 and get features
    t1 = Tier1FBA(verbose=False)
    print("Tier 1 loaded.")
    
    labeled_genes = [g for g in GENE_ESSENTIALITY if g in t1.sim.gene_rxns]
    feats, fba_calls, timing = t1.predict_all(labeled_genes)
    y_true = np.array([int(GENE_ESSENTIALITY[g] in ('E', 'Q')) for g in labeled_genes])
    
    print(f"Data: N={len(y_true)} genes, "
          f"positives={y_true.sum()}, negatives={(1-y_true).sum()}")
    print(f"Feature dim: {feats.shape[1]}")
    
    # Baseline: FBA alone (this is Tier 1 only)
    fba_result = {
        'name': 'Tier 1: FBA alone',
        'accuracy': ((fba_calls == y_true.astype(bool)).sum()) / len(y_true),
        'tp': int(((fba_calls == 1) & (y_true == 1)).sum()),
        'fp': int(((fba_calls == 1) & (y_true == 0)).sum()),
        'tn': int(((fba_calls == 0) & (y_true == 0)).sum()),
        'fn': int(((fba_calls == 0) & (y_true == 1)).sum()),
    }
    fba_result['sensitivity'] = fba_result['tp'] / max(fba_result['tp'] + fba_result['fn'], 1)
    fba_result['specificity'] = fba_result['tn'] / max(fba_result['tn'] + fba_result['fp'], 1)
    fba_result['balanced_accuracy'] = 0.5 * (fba_result['sensitivity'] + fba_result['specificity'])
    
    # Models to benchmark
    models = {
        'Tier 2a: Logistic on features':
            lambda: LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0),
        'Tier 2b: GBM on features (shallow)':
            lambda: GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                               learning_rate=0.05, random_state=0),
        'Tier 2c: GBM on features (deeper)':
            lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                               learning_rate=0.05, random_state=0),
    }
    
    results = [fba_result]
    for name, factory in models.items():
        r = evaluate_model(factory, feats, y_true, name)
        results.append(r)
    
    print("\n" + "-" * 70)
    print("Results (all LOO-CV except Tier 1 which is deterministic FBA):")
    print("-" * 70)
    for r in results:
        print(fmt(r))
    
    # Pick best Tier 2
    best = max(results[1:], key=lambda r: r['balanced_accuracy'])
    print(f"\nBest Tier 2: {best['name']}")
    print(f"  Improvement over Tier 1: "
          f"bal_acc {fba_result['balanced_accuracy']*100:.1f}% -> "
          f"{best['balanced_accuracy']*100:.1f}% "
          f"(+{(best['balanced_accuracy']-fba_result['balanced_accuracy'])*100:.1f}pp)")
    
    return results, feats, y_true


if __name__ == '__main__':
    main()
