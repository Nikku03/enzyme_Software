"""
Iterative Colab revisit loop for the hybrid full-xTB council model.

Cycle (up to --iterations times):
  1. Deep-analyze the last --recent-reports episode logs using LNN, wave, and
     analogical diagnostics extracted from each JSONL.
  2. Aggregate findings across all logs to identify systemic problems.
  3. Build a priority-ordered strategy list from the evidence.
  4. Train for 40-60 epochs (with early stopping) using the top strategy.
  5. Analyze the new result and compare against the rolling baseline.
  6. Promote baseline if improved; if stuck for MAX_STUCK_STREAK consecutive
     attempts, switch to alternate exploration paths.

Usage — from Colab cell:
    exec(open('/content/enzyme_Software/scripts/colab_revisit_hybrid_lnn.py').read())

Usage — from CLI:
    python scripts/colab_revisit_hybrid_lnn.py [--iterations 5] [--recent-reports 5]
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, variance
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Number of consecutive no-improvements before switching to alternate paths
MAX_STUCK_STREAK = 3


# ── lazy loader for the episode-log analyzer ─────────────────────────────────

_ANALYZE_FN = None


def _get_analyze_fn():
    global _ANALYZE_FN
    if _ANALYZE_FN is not None:
        return _ANALYZE_FN
    path = ROOT / "scripts" / "analyze_hybrid_episode_log.py"
    if not path.exists():
        raise FileNotFoundError(f"Analyzer not found: {path}")
    spec   = importlib.util.spec_from_file_location("_hybrid_ep_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ANALYZE_FN = module.analyze
    return _ANALYZE_FN


# ── environment helpers ───────────────────────────────────────────────────────

def _ensure_rdkit() -> None:
    try:
        from rdkit import Chem  # noqa: F401
        return
    except Exception:
        pass
    print("RDKit not found. Installing rdkit-pypi...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "rdkit-pypi"], cwd=str(ROOT)
    )


def _env_default(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


# ── deep log diagnostics ──────────────────────────────────────────────────────

def _extract_log_diagnostics(log_path: Path) -> dict[str, Any]:
    """
    Parse a hybrid episode JSONL and return a structured dict of training
    signals used to drive strategy selection.

    Extracted signals
    -----------------
    LNN voter
        lnn_voter_acc    per-val-episode argmax accuracy
        lnn_conf_mean    mean LNN confidence at top-1 atom

    Wave voter
        wave_voter_acc   same for wave
        wave_vote_degrading  True if wave_vote_loss rises in 2nd half of training
        wave_vote_loss_trend list of per-epoch means

    Analogical voter
        ana_voter_acc    same for analogical
        ana_conf_mean    mean analogical confidence (dead if < 0.05)
        ana_dead         bool
        comp_wt_frozen   True if competition_weight_mean ≈ 1/N (no differentiation)
        ana_cyp_dominant True if CYP loss dominates site loss
        precedent_nonzero  True if the precedent logbook was ever active

    Council / board
        board_lnn_var    variance of per-epoch mean LNN board weight
        council_voter_acc argmax of council_logit accuracy on val

    Overall
        best_val_top1 / best_val_top3
        overfitting_gap  mean(train_top1) - mean(val_top1) over last 3 epochs
        n_epochs
    """
    # --- accumulators ---
    ep_train_top1: dict[int, list[int]] = defaultdict(list)
    ep_val_top1:   dict[int, list[int]] = defaultdict(list)
    ep_val_top3:   dict[int, list[int]] = defaultdict(list)
    board_epoch:   dict[int, list[list[float]]] = defaultdict(list)   # [lnn,wave,ana]
    ana_conf_all:  list[float] = []
    voter_acc:     dict[str, list[int]] = defaultdict(list)           # val only
    lnn_conf_all:  list[float] = []

    step_wave_loss: dict[int, list[float]] = defaultdict(list)
    step_ana_cyp:   dict[int, list[float]] = defaultdict(list)
    step_ana_site:  dict[int, list[float]] = defaultdict(list)
    step_comp_wt:   dict[int, list[float]] = defaultdict(list)

    precedent_nonzero = False
    n_epochs = 0

    def _scalar(v: Any) -> float | None:
        """Unwrap a possibly-nested single-element list to a scalar float."""
        if v is None:
            return None
        if isinstance(v, (int, float, bool)):
            return float(v)
        if isinstance(v, list):
            if len(v) == 0:
                return None
            inner = v[0]
            if isinstance(inner, (int, float, bool)):
                return float(inner)
        return None

    def _argmax_nested(arr: list, n: int) -> int | None:
        """Return argmax index for a list that may be [[s],[s],...] or [s,s,...]."""
        if not arr or len(arr) != n:
            return None
        try:
            scores = [_scalar(x) for x in arr]
            if any(s is None for s in scores):
                return None
            return max(range(n), key=lambda i: scores[i])
        except Exception:
            return None

    with open(log_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue

            rtype = rec.get("record_type")
            epoch = int(rec.get("epoch") or 0)
            split = str(rec.get("split") or "")
            n_epochs = max(n_epochs, epoch)

            # ── episode records ──────────────────────────────────────────────
            if rtype == "episode":
                votes   = rec.get("votes")      or {}
                outcome = rec.get("outcome")    or {}
                ana     = rec.get("analogical") or {}
                inp     = rec.get("input")      or {}

                n_atoms    = int(inp.get("num_atoms") or 0)
                true_sites = outcome.get("true_site_atoms") or []
                top1       = bool(outcome.get("top1_hit"))
                top3       = bool(outcome.get("top3_hit"))

                if split == "train":
                    ep_train_top1[epoch].append(int(top1))
                elif split == "val":
                    ep_val_top1[epoch].append(int(top1))
                    ep_val_top3[epoch].append(int(top3))

                    # Per-voter accuracy: argmax of each voter's logit array.
                    # Vote arrays are [[scalar], [scalar], ...] (nested), so we
                    # use _argmax_nested to handle both flat and nested formats.
                    if true_sites and n_atoms > 0:
                        for voter_key, vote_field in [
                            ("lnn",     "lnn_vote"),
                            ("wave",    "wave_vote"),
                            ("ana",     "analogical_vote"),
                            ("council", "council_logit"),
                        ]:
                            arr = votes.get(vote_field)
                            if arr and isinstance(arr, list):
                                best = _argmax_nested(arr, n_atoms)
                                if best is not None:
                                    voter_acc[voter_key].append(int(best in true_sites))

                    # Board weights (per-atom [lnn, wave, ana])
                    bw = votes.get("board_weights")
                    if bw and isinstance(bw, list):
                        for w in bw:
                            if isinstance(w, list) and len(w) == 3:
                                try:
                                    board_epoch[epoch].append([float(w[0]), float(w[1]), float(w[2])])
                                except (TypeError, ValueError):
                                    pass

                    # Bridge-level analogical confidence (real metric for "dead" detection).
                    # It lives in episode.analogical.bridge_metrics (same as step diagnostics).
                    bm = ana.get("bridge_metrics") or {}
                    ac = bm.get("analogical_confidence_mean")
                    if ac is not None:
                        try:
                            ana_conf_all.append(float(ac))
                        except (TypeError, ValueError):
                            pass

                    # Precedent logbook active: check precedent_logbook_size > 0
                    if not precedent_nonzero:
                        if float(bm.get("precedent_logbook_size") or 0) > 0:
                            precedent_nonzero = True
                        elif float(bm.get("precedent_positive_support_mean") or 0) > 0:
                            precedent_nonzero = True

                    # LNN confidence (vote-head confidence, from votes.lnn_conf)
                    lnn_conf_arr = votes.get("lnn_conf")
                    if lnn_conf_arr and isinstance(lnn_conf_arr, list):
                        for v in lnn_conf_arr:
                            s = _scalar(v)
                            if s is not None:
                                lnn_conf_all.append(s)

            # ── step records ─────────────────────────────────────────────────
            elif rtype == "step":
                stats   = rec.get("stats")        or {}
                diag    = rec.get("diagnostics")  or {}
                diag_nb = diag.get("nexus_bridge") or {}
                diag_som= diag.get("som")          or {}

                if split == "train":
                    wvl = stats.get("nexus_wave_vote_loss")
                    if wvl is not None:
                        try:
                            step_wave_loss[epoch].append(float(wvl))
                        except (TypeError, ValueError):
                            pass

                    # CYP / site from nexus_bridge diagnostics
                    for src_key, dest in [
                        ("analogical_cyp_loss",  step_ana_cyp),
                        ("analogical_site_loss",  step_ana_site),
                    ]:
                        v = diag_nb.get(src_key)
                        if v is not None:
                            try:
                                dest[epoch].append(float(v))
                            except (TypeError, ValueError):
                                pass

                    # Bridge-level confidence (train)
                    ac = diag_nb.get("analogical_confidence_mean")
                    if ac is not None:
                        try:
                            ana_conf_all.append(float(ac))
                        except (TypeError, ValueError):
                            pass

                # Competition weight is in diagnostics.som (both train and val steps)
                cwt = diag_som.get("competition_weight_mean")
                if cwt is not None:
                    try:
                        step_comp_wt[epoch].append(float(cwt))
                    except (TypeError, ValueError):
                        pass

    epochs = sorted(set(ep_train_top1) | set(ep_val_top1))
    if not epochs:
        return {"valid": False, "n_epochs": 0}

    # --- overfitting gap (last 3 epochs) ---
    last3       = epochs[-3:] if len(epochs) >= 3 else epochs
    train_means = [mean(ep_train_top1[e]) for e in last3 if ep_train_top1[e]]
    val_means   = [mean(ep_val_top1[e])   for e in last3 if ep_val_top1[e]]
    overfitting_gap = (mean(train_means) - mean(val_means)) if (train_means and val_means) else 0.0

    # --- best val ---
    best_val_top1 = max(
        (mean(ep_val_top1[e]) for e in epochs if ep_val_top1[e]), default=0.0
    )
    best_val_top3 = max(
        (mean(ep_val_top3[e]) for e in epochs if ep_val_top3[e]), default=0.0
    )

    # --- board stability: variance of per-epoch mean LNN weight ---
    lnn_epoch_means = [
        mean(w[0] for w in board_epoch[e])
        for e in epochs if board_epoch[e]
    ]
    board_lnn_var = variance(lnn_epoch_means) if len(lnn_epoch_means) >= 2 else 0.0

    # --- wave voter degradation: wave_vote_loss rising in 2nd half ---
    wvl_series = [mean(step_wave_loss[e]) for e in epochs if step_wave_loss[e]]
    wave_vote_degrading = False
    if len(wvl_series) >= 4:
        half = len(wvl_series) // 2
        wave_vote_degrading = mean(wvl_series[half:]) > mean(wvl_series[:half]) + 0.03

    # --- analogical confidence ---
    ana_conf_mean = mean(ana_conf_all) if ana_conf_all else 0.0
    ana_dead      = ana_conf_mean < 0.05

    # --- competition weight frozen (≈ 1/N ≈ 0.0335) ---
    all_cwt       = [v for e in epochs for v in step_comp_wt[e]]
    comp_wt_mean  = mean(all_cwt) if all_cwt else 0.0335
    # < 0.07 means near-random atom weighting (no real differentiation)
    comp_wt_frozen = comp_wt_mean < 0.07

    # --- analogical CYP dominance ---
    cyp_means  = [mean(step_ana_cyp[e])  for e in epochs if step_ana_cyp[e]]
    site_means = [mean(step_ana_site[e]) for e in epochs if step_ana_site[e]]
    # CYP dominant if CYP loss stays high (>0.5) even in later epochs
    ana_cyp_dominant = (mean(cyp_means[-3:]) > 0.5) if len(cyp_means) >= 3 else (
        mean(cyp_means) > 0.5 if cyp_means else False
    )

    # --- per-voter accuracy (over all val episodes) ---
    voter_accs = {
        voter: mean(vals) if vals else 0.0
        for voter, vals in voter_acc.items()
    }

    return {
        "valid":                True,
        "n_epochs":             n_epochs,
        "best_val_top1":        best_val_top1,
        "best_val_top3":        best_val_top3,
        "overfitting_gap":      overfitting_gap,
        "board_lnn_var":        board_lnn_var,
        "wave_vote_degrading":  wave_vote_degrading,
        "wave_vote_loss_series": wvl_series,
        "ana_conf_mean":        ana_conf_mean,
        "ana_dead":             ana_dead,
        "comp_wt_mean":         comp_wt_mean,
        "comp_wt_frozen":       comp_wt_frozen,
        "ana_cyp_dominant":     ana_cyp_dominant,
        "precedent_nonzero":    precedent_nonzero,
        "voter_accs":           voter_accs,
        "lnn_conf_mean":        mean(lnn_conf_all) if lnn_conf_all else 0.0,
    }


def _frac_positive(diags: list[dict], fn) -> float:
    valid = [d for d in diags if d.get("valid")]
    return sum(1 for d in valid if fn(d)) / max(1, len(valid))


# ── scoring / comparison helpers ──────────────────────────────────────────────

def _score_metrics(metrics: dict[str, Any]) -> float:
    if not metrics:
        return float("-inf")
    top1      = float(metrics.get("site_top1_acc",  metrics.get("top1_acc",  0.0)))
    top3      = float(metrics.get("site_top3_acc",  metrics.get("top3_acc",  0.0)))
    auc       = float(metrics.get("site_auc",       0.0))
    precision = float(metrics.get("site_precision", 0.0))
    fp        = float(metrics.get("fp",             0.0))
    return 0.40 * top3 + 0.30 * auc + 0.20 * top1 + 0.10 * precision - 0.0005 * fp


def _compare_key(metrics: dict[str, Any]) -> tuple:
    return (
        _score_metrics(metrics),
        float(metrics.get("site_top3_acc", metrics.get("top3_acc", 0.0))),
        float(metrics.get("site_auc",      0.0)),
        float(metrics.get("site_top1_acc", metrics.get("top1_acc", 0.0))),
        -float(metrics.get("fp",           0.0)),
    )


def _selection_metrics_for_run(run: "ReviewedRun", scope: str = "val") -> dict[str, Any]:
    """Return the metric view used for baseline selection and promotion."""
    if scope == "test":
        tm = run.report.get("test_metrics") or {}
        if tm:
            return tm
    diag = run.diagnostics or {}
    return {
        "site_top1_acc": float(diag.get("best_val_top1", run.report.get("best_val_top1", 0.0))),
        "site_top3_acc": float(diag.get("best_val_top3", run.report.get("best_val_monitor", 0.0))),
    }


def _resolve_existing_path(raw: str | None, search_dirs: list[Path]) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    direct = Path(text)
    if direct.exists():
        return direct
    name = direct.name
    for d in search_dirs:
        candidate = d / name
        if candidate.exists():
            return candidate
    return None


def _timestamp_from_report_path(report_path: Path) -> str:
    stem = report_path.stem
    prefix = "hybrid_full_xtb_report_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return ""


def _latest_paths(pattern: str, search_dirs: list[Path], limit: int) -> list[Path]:
    items: dict[Path, float] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.glob(pattern):
            try:
                items[p.resolve()] = p.stat().st_mtime
            except FileNotFoundError:
                continue
    ranked = sorted(items.items(), key=lambda kv: kv[1], reverse=True)
    return [p for p, _ in ranked[:limit]]


def _candidate_checkpoint_dirs(search_dirs: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for path in search_dirs:
        if path.is_file():
            path = path.parent
        if path in seen or not path.exists():
            continue
        seen.add(path)
        dirs.append(path)
        for extra in ("checkpoints/hybrid_full_xtb", "hybrid_full_xtb", "."):
            cand = path / extra
            if cand.exists() and cand not in seen:
                seen.add(cand)
                dirs.append(cand)
    return dirs


def _resolve_baseline_checkpoint(explicit: str, baseline_report: Path, search_dirs: list[Path]) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    timestamp = _timestamp_from_report_path(baseline_report)
    checkpoint_dirs = _candidate_checkpoint_dirs(search_dirs)
    if timestamp:
        archive_name = f"hybrid_full_xtb_{timestamp}.pt"
        for directory in checkpoint_dirs:
            candidate = directory / archive_name
            if candidate.exists():
                return candidate
    for filename in ("hybrid_full_xtb_best.pt", "hybrid_full_xtb_latest.pt"):
        for directory in checkpoint_dirs:
            candidate = directory / filename
            if candidate.exists():
                return candidate
    return None


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ReviewedRun:
    report_path:    Path
    log_path:       Path | None
    test_analysis:  dict[str, Any] | None
    val_analysis:   dict[str, Any] | None
    report:         dict[str, Any]
    score:          float
    diagnostics:    dict[str, Any]


@dataclass
class Strategy:
    name:             str
    rationale:        str
    train_overrides:  dict[str, str]
    model_env:        dict[str, str]


# ── run loader ────────────────────────────────────────────────────────────────

def _load_run(
    report_path: Path,
    analysis_dir: Path,
    search_dirs:  list[Path],
) -> ReviewedRun:
    report    = json.loads(report_path.read_text())
    log_path  = _resolve_existing_path(report.get("episode_log_path"), search_dirs)

    test_analysis = None
    val_analysis  = None
    diagnostics   = {"valid": False}

    if log_path is not None and log_path.exists():
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analyze_fn    = _get_analyze_fn()
        test_analysis = analyze_fn(log_path, "test", 10)
        val_analysis  = analyze_fn(log_path, "val",  10)
        diagnostics   = _extract_log_diagnostics(log_path)

    metrics = _selection_metrics_for_run(
        ReviewedRun(report_path, log_path, None, None, report, 0.0, {})
    )
    score = _score_metrics(metrics)

    return ReviewedRun(
        report_path   = report_path,
        log_path      = log_path,
        test_analysis = test_analysis,
        val_analysis  = val_analysis,
        report        = report,
        score         = score,
        diagnostics   = diagnostics,
    )


# ── strategy builder ──────────────────────────────────────────────────────────

def _build_strategies(
    all_diags:    list[dict[str, Any]],
    baseline_log: str,
    iterations:   int,
    stuck:        bool = False,
) -> list[Strategy]:
    """
    Build a priority-ordered list of training strategies from the aggregated
    diagnostics of the last N episode logs.

    If `stuck=True` the function returns alternate exploration paths that
    specifically target structural dead-ends not solved by normal strategies.
    """

    def _ef(fn) -> float:          # fraction of valid logs where condition is True
        return _frac_positive(all_diags, fn)

    # ── alternate (stuck) paths ───────────────────────────────────────────────
    if stuck:
        return [
            Strategy(
                name      = "live_wave_vote_inputs",
                rationale = (
                    "Stuck. Un-detach wave encoder for vote head (HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS=1). "
                    "Removes stale-mapping problem: vote head now sees up-to-date wave features."
                ),
                train_overrides = {
                    "HYBRID_COLAB_EPOCHS":                    "50",
                    "HYBRID_COLAB_LR":                        "3e-5",
                    "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                    "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
                },
                model_env = {
                    "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":              "1",
                    "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":         "0.06",
                    "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":             "2.0",
                },
            ),
            Strategy(
                name      = "live_analogical_vote_inputs",
                rationale = (
                    "Stuck. Un-detach analogical encoder for vote head "
                    "(HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS=1). "
                    "Breaks the confidence deadlock where analogical always outputs ≈0."
                ),
                train_overrides = {
                    "HYBRID_COLAB_EPOCHS":                    "50",
                    "HYBRID_COLAB_LR":                        "3e-5",
                    "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                    "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
                },
                model_env = {
                    "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":            "1",
                    "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":       "0.05",
                    "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":        "0.04",
                },
            ),
            Strategy(
                name      = "high_entropy_board_reset",
                rationale = (
                    "Stuck. Push board weights toward uniform with high entropy "
                    "penalty (board_entropy=0.02) to force re-exploration of voter balance."
                ),
                train_overrides = {
                    "HYBRID_COLAB_EPOCHS":                    "60",
                    "HYBRID_COLAB_LR":                        "2e-5",
                    "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "8",
                    "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
                },
                model_env = {
                    "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":  "0.02",
                    "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":      "1.5",
                    "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":       "1",
                    "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS": "1",
                },
            ),
            Strategy(
                name      = "memory_unlock_precedent_fresh",
                rationale = (
                    "Stuck. Unfreeze nexus memory + enable precedent logbook "
                    "so analogical can actively learn from new cases."
                ),
                train_overrides = {
                    "HYBRID_COLAB_EPOCHS":                      "60",
                    "HYBRID_COLAB_LR":                          "4e-5",
                    "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":     "8",
                    "HYBRID_COLAB_EARLY_STOPPING_METRIC":       "site_top3",
                    "HYBRID_COLAB_FREEZE_NEXUS_MEMORY":         "0",
                    "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK":   "0",
                },
                model_env = {
                    "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":       "0.10",
                    "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":  "0.05",
                    "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":   "0.04",
                },
            ),
            Strategy(
                name      = "low_lr_high_regularization",
                rationale = (
                    "Stuck. Very low LR (1e-5) with strong regularisation "
                    "to escape the current local optimum gradually."
                ),
                train_overrides = {
                    "HYBRID_COLAB_EPOCHS":                    "60",
                    "HYBRID_COLAB_LR":                        "1e-5",
                    "HYBRID_COLAB_WD":                        "4e-4",
                    "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "10",
                    "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
                },
                model_env = {
                    "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":  "0.20",
                    "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":  "0.012",
                },
            ),
        ][:iterations]

    # ── normal mode: evidence-driven priority ordering ────────────────────────

    overfitting_frac  = _ef(lambda d: d.get("overfitting_gap", 0.0) > 0.50)
    ana_dead_frac     = _ef(lambda d: d.get("ana_dead", False))
    comp_frozen_frac  = _ef(lambda d: d.get("comp_wt_frozen", False))
    wave_degrade_frac = _ef(lambda d: d.get("wave_vote_degrading", False))
    ana_cyp_frac      = _ef(lambda d: d.get("ana_cyp_dominant", False))
    board_frozen_frac = _ef(lambda d: d.get("board_lnn_var", 1.0) < 0.005)
    precedent_never   = all(
        not d.get("precedent_nonzero", False) for d in all_diags if d.get("valid")
    )

    # Build ordered candidate list; worst problems first
    candidates: list[Strategy] = []

    # ① Analogical dead + no precedent → enable logbook (highest impact)
    if ana_dead_frac > 0.4 and precedent_never:
        candidates.append(Strategy(
            name      = "enable_precedent_logbook",
            rationale = (
                f"Analogical confidence dead in {ana_dead_frac:.0%} of runs "
                f"and precedent logbook never active. "
                "Enable logbook so analogical builds case memory and gains selectivity."
            ),
            train_overrides = {
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
                "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "0",
                "HYBRID_COLAB_PRECEDENT_LOGBOOK":         baseline_log,
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":       "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":      "0.03",
            },
        ))

    # ② Severe overfitting
    if overfitting_frac > 0.4:
        candidates.append(Strategy(
            name      = "anti_overfit_regularize",
            rationale = (
                f"Severe overfitting (train-val gap > 0.5) in {overfitting_frac:.0%} of runs. "
                "Increase arbiter dropout, weight decay, and board entropy."
            ),
            train_overrides = {
                "HYBRID_COLAB_EPOCHS":                    "60",
                "HYBRID_COLAB_LR":                        "3e-5",
                "HYBRID_COLAB_WD":                        "3e-4",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":  "0.20",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":  "0.010",
            },
        ))

    # ③ Wave voter degrading
    if wave_degrade_frac > 0.3:
        candidates.append(Strategy(
            name      = "wave_voter_recovery",
            rationale = (
                f"Wave vote loss rising in {wave_degrade_frac:.0%} of runs — "
                "vote head mapping goes stale as encoder evolves. "
                "Un-detach wave inputs and increase wave vote supervision."
            ),
            train_overrides = {
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":       "1",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":  "0.06",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":      "2.0",
            },
        ))

    # ④ Analogical CYP loss dominating site loss
    if ana_cyp_frac > 0.4:
        candidates.append(Strategy(
            name      = "analogical_cyp_rebalance",
            rationale = (
                f"Analogical CYP loss dominant (>0.5) in {ana_cyp_frac:.0%} of runs — "
                "analogical trades SoM accuracy for easier CYP task. "
                "Reduce CYP aux scale and strengthen site vote supervision."
            ),
            train_overrides = {
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":       "0.04",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":      "0.05",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":           "0.07",
            },
        ))

    # ⑤ Competition weight frozen (analogical not differentiating atoms)
    if comp_frozen_frac > 0.4:
        candidates.append(Strategy(
            name      = "competition_weight_unlock",
            rationale = (
                f"Competition weight ≈ 1/N in {comp_frozen_frac:.0%} of runs — "
                "analogical assigns equal weight to all atoms. "
                "Unfreeze nexus memory to allow active learning."
            ),
            train_overrides = {
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_LR":                        "4e-5",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
                "HYBRID_COLAB_FREEZE_NEXUS_MEMORY":       "0",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":       "0.10",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":  "0.05",
            },
        ))

    # ⑥ Board weights frozen
    if board_frozen_frac > 0.4:
        candidates.append(Strategy(
            name      = "board_entropy_push",
            rationale = (
                f"Board weights frozen (LNN weight variance < 0.005) in "
                f"{board_frozen_frac:.0%} of runs. "
                "Add entropy regularisation to force the council to explore voter balance."
            ),
            train_overrides = {
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":  "0.015",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":      "1.8",
            },
        ))

    # Always include a conservative long-run fallback
    candidates.append(Strategy(
        name      = "conservative_longer",
        rationale = (
            "Conservative: lower LR, 60 epochs, mild regularisation, "
            "let existing configuration improve with more training."
        ),
        train_overrides = {
            "HYBRID_COLAB_EPOCHS":                    "60",
            "HYBRID_COLAB_LR":                        "4e-5",
            "HYBRID_COLAB_WD":                        "2e-4",
            "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
            "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
        },
        model_env = {
            "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":  "0.005",
        },
    ))

    # Deduplicate by name while preserving order
    seen: set[str] = set()
    unique: list[Strategy] = []
    for s in candidates:
        if s.name not in seen:
            seen.add(s.name)
            unique.append(s)

    return unique[:iterations]


# ── training command builder ──────────────────────────────────────────────────

def _build_train_command(
    checkpoint_path: Path,
    output_dir:      Path,
    artifact_dir:    Path,
    settings:        dict[str, str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_hybrid_full_xtb.py"),
        "--dataset",              settings["HYBRID_COLAB_DATASET"],
        "--structure-sdf",        settings["HYBRID_COLAB_STRUCTURE_SDF"],
        "--checkpoint",           str(checkpoint_path),
        "--xtb-cache-dir",        settings["HYBRID_COLAB_XTB_CACHE_DIR"],
        "--epochs",               settings["HYBRID_COLAB_EPOCHS"],
        "--batch-size",           settings["HYBRID_COLAB_BATCH_SIZE"],
        "--learning-rate",        settings["HYBRID_COLAB_LR"],
        "--weight-decay",         settings["HYBRID_COLAB_WD"],
        "--seed",                 settings["HYBRID_COLAB_SEED"],
        "--output-dir",           str(output_dir),
        "--artifact-dir",         str(artifact_dir),
        "--early-stopping-patience", settings["HYBRID_COLAB_EARLY_STOPPING_PATIENCE"],
        "--early-stopping-metric",   settings["HYBRID_COLAB_EARLY_STOPPING_METRIC"],
    ]

    if settings.get("HYBRID_COLAB_MANUAL_CACHE_DIR"):
        cmd.extend(["--manual-feature-cache-dir", settings["HYBRID_COLAB_MANUAL_CACHE_DIR"]])
    if int(settings.get("HYBRID_COLAB_LIMIT", "0") or "0") > 0:
        cmd.extend(["--limit", settings["HYBRID_COLAB_LIMIT"]])
    if settings.get("HYBRID_COLAB_SITE_LABELED_ONLY", "1") == "1":
        cmd.append("--site-labeled-only")
    if settings.get("HYBRID_COLAB_COMPUTE_XTB_IF_MISSING", "0") == "1":
        cmd.append("--compute-xtb-if-missing")
    if settings.get("HYBRID_COLAB_FREEZE_NEXUS_MEMORY", "1") == "1":
        cmd.append("--freeze-nexus-memory")
    if settings.get("HYBRID_COLAB_INCLUDE_XENOSITE", "1") == "1":
        manifest = settings.get("HYBRID_COLAB_XENOSITE_MANIFEST", "")
        if manifest:
            cmd.extend(["--xenosite-manifest", manifest,
                        "--xenosite-topk",     settings.get("HYBRID_COLAB_XENOSITE_TOPK", "1")])
    if settings.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1") == "1":
        cmd.append("--disable-precedent-logbook")
    elif settings.get("HYBRID_COLAB_PRECEDENT_LOGBOOK"):
        cmd.extend(["--precedent-logbook", settings["HYBRID_COLAB_PRECEDENT_LOGBOOK"]])
    return cmd


def _resolve_attempt_artifacts(
    artifact_dir: Path, output_dir: Path, search_dirs: list[Path]
) -> tuple[Path, Path | None, Path | None]:
    reports = sorted(
        artifact_dir.glob("hybrid_full_xtb_report_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not reports:
        raise FileNotFoundError(f"No report produced in {artifact_dir}")
    report_path = reports[-1]
    report      = json.loads(report_path.read_text())
    log_path    = _resolve_existing_path(report.get("episode_log_path"), [artifact_dir] + search_dirs)
    best_ckpt   = output_dir / "hybrid_full_xtb_best.pt"
    if not best_ckpt.exists():
        best_ckpt = output_dir / "hybrid_full_xtb_latest.pt"
    return report_path, log_path, best_ckpt if best_ckpt.exists() else None


def _default_train_settings() -> dict[str, str]:
    return {
        "HYBRID_COLAB_DATASET":           os.environ.get("HYBRID_COLAB_DATASET",           "data/prepared_training/main5_site_conservative_singlecyp_clean.json"),
        "HYBRID_COLAB_STRUCTURE_SDF":     os.environ.get("HYBRID_COLAB_STRUCTURE_SDF",     "3D structures.sdf"),
        "HYBRID_COLAB_XTB_CACHE_DIR":     os.environ.get("HYBRID_COLAB_XTB_CACHE_DIR",     "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/full_xtb"),
        "HYBRID_COLAB_MANUAL_CACHE_DIR":  os.environ.get("HYBRID_COLAB_MANUAL_CACHE_DIR",  "/content/drive/MyDrive/enzyme_hybrid_lnn/cache/manual_engine_full"),
        "HYBRID_COLAB_XENOSITE_MANIFEST": os.environ.get("HYBRID_COLAB_XENOSITE_MANIFEST", "data/xenosite_suppl/manifest.json"),
        "HYBRID_COLAB_XENOSITE_TOPK":     os.environ.get("HYBRID_COLAB_XENOSITE_TOPK",     "1"),
        "HYBRID_COLAB_EPOCHS":            os.environ.get("HYBRID_COLAB_EPOCHS",            "50"),
        "HYBRID_COLAB_BATCH_SIZE":        os.environ.get("HYBRID_COLAB_BATCH_SIZE",        "16"),
        "HYBRID_COLAB_LR":                os.environ.get("HYBRID_COLAB_LR",                "5e-5"),
        "HYBRID_COLAB_WD":                os.environ.get("HYBRID_COLAB_WD",                "1e-4"),
        "HYBRID_COLAB_SEED":              os.environ.get("HYBRID_COLAB_SEED",              "42"),
        "HYBRID_COLAB_LIMIT":             os.environ.get("HYBRID_COLAB_LIMIT",             "0"),
        "HYBRID_COLAB_EARLY_STOPPING_PATIENCE": os.environ.get("HYBRID_COLAB_EARLY_STOPPING_PATIENCE", "6"),
        "HYBRID_COLAB_EARLY_STOPPING_METRIC":   os.environ.get("HYBRID_COLAB_EARLY_STOPPING_METRIC",   "site_top3"),
        "HYBRID_COLAB_SITE_LABELED_ONLY":       os.environ.get("HYBRID_COLAB_SITE_LABELED_ONLY",       "1"),
        "HYBRID_COLAB_COMPUTE_XTB_IF_MISSING":  os.environ.get("HYBRID_COLAB_COMPUTE_XTB_IF_MISSING",  "0"),
        "HYBRID_COLAB_FREEZE_NEXUS_MEMORY":     os.environ.get("HYBRID_COLAB_FREEZE_NEXUS_MEMORY",     "1"),
        "HYBRID_COLAB_INCLUDE_XENOSITE":        os.environ.get("HYBRID_COLAB_INCLUDE_XENOSITE",        "1"),
        "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": os.environ.get("HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK", "1"),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _promote_baseline_artifacts(artifact_root: Path, report_path: Path, checkpoint_path: Path | None, log_path: Path | None) -> None:
    if report_path.exists():
        target = artifact_root / "revisit_current_baseline_report.json"
        target.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    if checkpoint_path is not None and checkpoint_path.exists():
        shutil.copy2(checkpoint_path, artifact_root / "revisit_current_baseline_best.pt")
    if log_path is not None and log_path.exists():
        shutil.copy2(log_path, artifact_root / "revisit_current_baseline_log.jsonl")


def _print_diagnosis(all_diags: list[dict]) -> None:
    valid = [d for d in all_diags if d.get("valid")]
    if not valid:
        print("  (no valid logs to diagnose)", flush=True)
        return
    print(f"  Logs analysed:          {len(valid)}", flush=True)
    print(f"  Overfitting (>50pp gap):{_frac_positive(valid, lambda d: d.get('overfitting_gap', 0) > 0.5):.0%}", flush=True)
    print(f"  Ana dead (conf<0.05):   {_frac_positive(valid, lambda d: d.get('ana_dead', False)):.0%}", flush=True)
    print(f"  Comp-wt frozen (≈1/N):  {_frac_positive(valid, lambda d: d.get('comp_wt_frozen', False)):.0%}", flush=True)
    print(f"  Wave vote degrading:    {_frac_positive(valid, lambda d: d.get('wave_vote_degrading', False)):.0%}", flush=True)
    print(f"  Ana CYP dominant:       {_frac_positive(valid, lambda d: d.get('ana_cyp_dominant', False)):.0%}", flush=True)
    print(f"  Precedent ever active:  {any(d.get('precedent_nonzero') for d in valid)}", flush=True)
    avg_gap = mean(d.get("overfitting_gap", 0.0) for d in valid)
    print(f"  Avg train-val gap:      {avg_gap:.3f}", flush=True)
    voter_keys = ("lnn", "wave", "ana", "council")
    for vk in voter_keys:
        vals = [d.get("voter_accs", {}).get(vk, 0.0) for d in valid]
        print(f"  Val voter acc [{vk:>8}]: {mean(vals):.3f}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run iterative revisit loops for hybrid council training."
    )
    parser.add_argument("--recent-reports",     type=int, default=int(os.environ.get("HYBRID_COLAB_REVISIT_RECENT_REPORTS", "5")))
    parser.add_argument("--iterations",         type=int, default=int(os.environ.get("HYBRID_COLAB_REVISIT_ITERATIONS", "5")))
    parser.add_argument("--baseline-report",    default=os.environ.get("HYBRID_COLAB_REVISIT_BASELINE_REPORT", ""))
    parser.add_argument("--baseline-checkpoint",default=os.environ.get("HYBRID_COLAB_REVISIT_BASELINE_CHECKPOINT", ""))
    parser.add_argument("--output-root",        default=os.environ.get("HYBRID_COLAB_REVISIT_OUTPUT_ROOT",  "/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/checkpoints"))
    parser.add_argument("--artifact-root",      default=os.environ.get("HYBRID_COLAB_REVISIT_ARTIFACT_ROOT","/content/drive/MyDrive/enzyme_hybrid_lnn/revisit/artifacts"))
    parser.add_argument("--selection-scope",    choices=("val", "test"), default=os.environ.get("HYBRID_COLAB_REVISIT_SELECTION_SCOPE", "val"))
    parser.add_argument("--search-dir",         action="append", default=[])
    # parse_known_args ignores Jupyter/Colab kernel args in sys.argv
    args, _ = parser.parse_known_args()

    os.chdir(ROOT)
    _ensure_rdkit()
    _env_default("TORCHDYNAMO_DISABLE",       "1")
    _env_default("TORCH_COMPILE_DISABLE",     "1")
    _env_default("HYBRID_FORCE_MANUAL_OPTIMIZER", "1")

    # ── search dirs ───────────────────────────────────────────────────────────
    search_dirs: list[Path] = [ROOT]
    for raw in args.search_dir:
        p = Path(raw)
        if p.exists():
            search_dirs.append(p)
    for env_key in ("HYBRID_COLAB_ARTIFACT_DIR", "HYBRID_COLAB_OUTPUT_DIR"):
        p = Path(os.environ.get(env_key, ""))
        if p.exists():
            search_dirs.append(p)

    output_root  = Path(args.output_root)
    artifact_root = Path(args.artifact_root)
    analysis_dir  = artifact_root / "analysis_cache"
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    # ── find recent reports ────────────────────────────────────────────────────
    if args.baseline_report:
        recent_report_paths = [Path(args.baseline_report)]
    else:
        recent_report_paths = _latest_paths(
            "**/hybrid_full_xtb_report_*.json", search_dirs, args.recent_reports
        )
    if not recent_report_paths:
        raise FileNotFoundError(
            "No recent hybrid_full_xtb reports found. "
            "Set HYBRID_COLAB_ARTIFACT_DIR or pass --search-dir."
        )

    print(f"\n{'='*60}", flush=True)
    print("HYBRID LNN REVISIT LOOP", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Selection scope: {args.selection_scope}", flush=True)
    print(f"Reports found: {len(recent_report_paths)}", flush=True)
    for p in recent_report_paths:
        print(f"  {p}", flush=True)

    # ── load and diagnose all recent runs ─────────────────────────────────────
    reviewed_runs = [_load_run(p, analysis_dir, search_dirs) for p in recent_report_paths]
    all_diags     = [run.diagnostics for run in reviewed_runs]
    for run in reviewed_runs:
        run.score = _score_metrics(_selection_metrics_for_run(run, args.selection_scope))

    print("\n── Multi-log diagnosis ──────────────────────────────────────", flush=True)
    _print_diagnosis(all_diags)

    # ── select baseline: best score among recent runs ─────────────────────────
    baseline = max(
        reviewed_runs,
        key=lambda r: _compare_key(_selection_metrics_for_run(r, args.selection_scope)),
    )

    # ── resolve baseline checkpoint ───────────────────────────────────────────
    baseline_checkpoint = _resolve_baseline_checkpoint(
        args.baseline_checkpoint or os.environ.get("HYBRID_COLAB_REVISIT_BASELINE_CHECKPOINT", ""),
        baseline.report_path,
        search_dirs,
    )
    if baseline_checkpoint is None or not baseline_checkpoint.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found for baseline report: {baseline.report_path}")

    baseline_log = str(baseline.log_path) if baseline.log_path else ""

    print(f"\n── Baseline ─────────────────────────────────────────────────", flush=True)
    print(f"  report:     {baseline.report_path}", flush=True)
    print(f"  checkpoint: {baseline_checkpoint}", flush=True)
    print(f"  score:      {baseline.score:.6f}", flush=True)

    # ── build initial strategy list from evidence ─────────────────────────────
    strategies = _build_strategies(
        all_diags    = all_diags,
        baseline_log = baseline_log,
        iterations   = args.iterations,
        stuck        = False,
    )

    print(f"\n── Planned strategies ({len(strategies)}) ──────────────────────────", flush=True)
    for i, s in enumerate(strategies, 1):
        print(f"  [{i}] {s.name}: {s.rationale[:80]}...", flush=True)

    summary: dict[str, Any] = {
        "bootstrap_reports":   [str(p) for p in recent_report_paths],
        "baseline_report":     str(baseline.report_path),
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_score":      baseline.score,
        "selection_scope":     args.selection_scope,
        "attempts":            [],
    }
    _write_json(artifact_root / "revisit_bootstrap.json", summary)
    _promote_baseline_artifacts(artifact_root, baseline.report_path, baseline_checkpoint, baseline.log_path)

    if args.iterations <= 0:
        _write_json(
            artifact_root / "revisit_best.json",
            {
                "baseline_report": str(baseline.report_path),
                "baseline_checkpoint": str(baseline_checkpoint),
                "baseline_score": baseline.score,
                "selection_scope": args.selection_scope,
                "attempt_count": 0,
                "improvements": 0,
                "used_alt_strategies": False,
            },
        )
        print("\nBootstrap validation complete (iterations=0).", flush=True)
        print(f"  Baseline report: {baseline.report_path}", flush=True)
        print(f"  Baseline ckpt:   {baseline_checkpoint}", flush=True)
        return

    base_settings        = _default_train_settings()
    current_baseline     = baseline
    current_checkpoint   = baseline_checkpoint
    consecutive_no_improve = 0
    used_alt_strategies  = False
    tried_strategy_names: set[str] = set()

    # ── main iterative loop ───────────────────────────────────────────────────
    for attempt_idx in range(1, args.iterations + 1):
        if consecutive_no_improve >= MAX_STUCK_STREAK and not used_alt_strategies:
            print(
                f"\n⚠ Stuck for {MAX_STUCK_STREAK} consecutive attempts. "
                "Switching to alternate exploration paths.",
                flush=True,
            )
            used_alt_strategies = True
            consecutive_no_improve = 0

        current_baseline_log = str(current_baseline.log_path) if current_baseline.log_path else baseline_log
        strategies = _build_strategies(
            all_diags=all_diags,
            baseline_log=current_baseline_log,
            iterations=max(args.iterations * 2, 10),
            stuck=used_alt_strategies,
        )
        strategy = next((s for s in strategies if s.name not in tried_strategy_names), None)
        if strategy is None:
            print("\nNo untried strategies remain. Stopping revisit loop.", flush=True)
            break
        tried_strategy_names.add(strategy.name)
        attempt_name = f"attempt_{attempt_idx:02d}_{strategy.name}"
        attempt_output_dir   = output_root   / attempt_name
        attempt_artifact_dir = artifact_root / attempt_name
        attempt_output_dir.mkdir(parents=True, exist_ok=True)
        attempt_artifact_dir.mkdir(parents=True, exist_ok=True)

        # Merge base settings with strategy overrides
        settings = dict(base_settings)
        settings.update(strategy.train_overrides)

        # Build subprocess environment
        env = os.environ.copy()
        env.update(settings)
        env["PYTHONPATH"] = f"{SRC}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["HYBRID_COLAB_LOCK_PRESET_POLICY"] = "1"
        env.update(strategy.model_env)

        print(f"\n{'='*60}", flush=True)
        print(f"Revisit {attempt_idx}/{len(strategies)}: {strategy.name}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"Rationale: {strategy.rationale}", flush=True)
        print(f"Warm-start: {current_checkpoint}", flush=True)
        print(f"Epochs: {settings['HYBRID_COLAB_EPOCHS']} | LR: {settings['HYBRID_COLAB_LR']} | WD: {settings['HYBRID_COLAB_WD']}", flush=True)
        if strategy.model_env:
            print(f"Model overrides: {strategy.model_env}", flush=True)

        # ── train ─────────────────────────────────────────────────────────────
        cmd = _build_train_command(
            checkpoint_path = current_checkpoint,
            output_dir      = attempt_output_dir,
            artifact_dir    = attempt_artifact_dir,
            settings        = settings,
        )
        subprocess.check_call(cmd, cwd=str(ROOT), env=env)

        # ── analyse result ─────────────────────────────────────────────────────
        report_path, log_path, produced_ckpt = _resolve_attempt_artifacts(
            attempt_artifact_dir, attempt_output_dir, search_dirs
        )
        reviewed = _load_run(report_path, analysis_dir, [attempt_artifact_dir, attempt_output_dir] + search_dirs)
        reviewed.score = _score_metrics(_selection_metrics_for_run(reviewed, args.selection_scope))
        all_diags.append(reviewed.diagnostics)

        improved = (
            _compare_key(_selection_metrics_for_run(reviewed, args.selection_scope))
            > _compare_key(_selection_metrics_for_run(current_baseline, args.selection_scope))
        )

        attempt_record: dict[str, Any] = {
            "attempt":        attempt_idx,
            "name":           strategy.name,
            "rationale":      strategy.rationale,
            "report_path":    str(report_path),
            "log_path":       str(log_path)          if log_path          else None,
            "checkpoint":     str(produced_ckpt)     if produced_ckpt     else None,
            "score":          reviewed.score,
            "improved":       improved,
            "selection_metrics": _selection_metrics_for_run(reviewed, args.selection_scope),
            "test_metrics":   reviewed.report.get("test_metrics"),
            "diagnostics":    {
                k: reviewed.diagnostics.get(k)
                for k in ("best_val_top1", "best_val_top3", "overfitting_gap",
                          "ana_conf_mean", "comp_wt_mean", "voter_accs")
            },
        }
        summary["attempts"].append(attempt_record)

        # ── promote or count streak ────────────────────────────────────────────
        if improved:
            consecutive_no_improve = 0
            current_baseline   = reviewed
            if produced_ckpt:
                current_checkpoint = produced_ckpt
            summary["baseline_report"]     = str(report_path)
            summary["baseline_checkpoint"] = str(current_checkpoint)
            summary["baseline_score"]      = reviewed.score
            _promote_baseline_artifacts(artifact_root, report_path, current_checkpoint, log_path)
            print(f"\n✓ New baseline promoted  score={reviewed.score:.6f}", flush=True)
            print(f"  report:     {report_path}", flush=True)
            print(f"  checkpoint: {current_checkpoint}", flush=True)
        else:
            consecutive_no_improve += 1
            print(
                f"\n✗ No improvement  "
                f"(attempt score={reviewed.score:.6f}  "
                f"baseline={current_baseline.score:.6f}  "
                f"streak={consecutive_no_improve}/{MAX_STUCK_STREAK})",
                flush=True,
            )

        _write_json(artifact_root / "revisit_bootstrap.json", summary)

    # ── write final summary ───────────────────────────────────────────────────
    best_summary = {
        "baseline_report":     summary["baseline_report"],
        "baseline_checkpoint": summary["baseline_checkpoint"],
        "baseline_score":      summary["baseline_score"],
        "selection_scope":     args.selection_scope,
        "attempt_count":       len(summary["attempts"]),
        "improvements":        sum(1 for a in summary["attempts"] if a["improved"]),
        "used_alt_strategies": used_alt_strategies,
    }
    _write_json(artifact_root / "revisit_best.json", best_summary)

    print(f"\n{'='*60}", flush=True)
    print("REVISIT LOOP COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Attempts:    {len(summary['attempts'])}", flush=True)
    print(f"  Improvements:{sum(1 for a in summary['attempts'] if a['improved'])}", flush=True)
    print(f"  Best score:  {summary['baseline_score']:.6f}", flush=True)
    print(f"  Best report: {summary['baseline_report']}", flush=True)
    print(f"  Best ckpt:   {summary['baseline_checkpoint']}", flush=True)


main()
