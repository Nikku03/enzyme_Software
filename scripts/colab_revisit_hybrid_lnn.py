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

def _resolve_repo_root() -> Path:
    file_value = globals().get("__file__")
    if file_value:
        return Path(file_value).resolve().parents[1]

    env_root = os.environ.get("HYBRID_COLAB_REPO_DIR", "").strip()
    if env_root:
        candidate = Path(env_root).resolve()
        if (candidate / "scripts" / "colab_revisit_hybrid_lnn.py").exists() and (candidate / "src").exists():
            return candidate

    cwd = Path.cwd().resolve()
    candidates = [
        cwd,
        cwd.parent,
        Path("/content/enzyme_Software"),
        Path("/content/enzyme_software"),
    ]
    for candidate in candidates:
        if (candidate / "scripts" / "colab_revisit_hybrid_lnn.py").exists() and (candidate / "src").exists():
            return candidate

    raise RuntimeError(
        "Unable to resolve repo root for colab_revisit_hybrid_lnn.py. "
        "Set HYBRID_COLAB_REPO_DIR or run from the repository root."
    )


ROOT = _resolve_repo_root()
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
    family:           str            # one of the FAMILY_* constants below
    name:             str
    rationale:        str
    train_overrides:  dict[str, str]   # → merged into settings, passed as CLI args
    model_env:        dict[str, str]   # → merged into subprocess env for model knobs


# ── family constants ──────────────────────────────────────────────────────────
# Each family is a coherent area of the search space.  The selector picks one
# bundle per triggered family, then fills remaining slots from a priority list.

FAMILY_TRAINING      = "training_policy"    # LR, WD, batch, epochs, patience, metric
FAMILY_DATA          = "data_policy"        # xenosite, site-label, precedent, memory
FAMILY_COUNCIL       = "council_balance"    # vote aux weights, entropy, logit scale
FAMILY_ANALOGICAL    = "analogical_system"  # CYP scale, aux, live inputs, precedent
FAMILY_WAVE          = "wave_system"        # wave aux, vote aux, live inputs
FAMILY_ARCHITECTURE  = "architecture"       # structural: live both, full unlock, etc.


# ── bundle catalogue ──────────────────────────────────────────────────────────

def _all_bundles(baseline_log: str) -> dict[str, list[Strategy]]:
    """
    Return ALL available strategy bundles organised by family.

    Design rules
    ------------
    * Every bundle sets the FULL coherent picture for its family — training
      schedules, data flags, council weights, model knobs — not just 1-2 vars.
    * `train_overrides` keys must exist in `_default_train_settings()` or be
      handled by `_build_train_command()`.
    * `model_env` keys are passed directly into the subprocess environment;
      `train_hybrid_full_xtb.py` reads them via `_collect_model_overrides()` and
      the explicit flag checks in `_build_train_command()`.
    """

    # ── A. training_policy ────────────────────────────────────────────────────
    # Covers: LR, WD, batch size, epochs, patience, early-stop metric, dropout.
    training = [
        Strategy(
            family = FAMILY_TRAINING,
            name   = "training.slow_patient_heavy_reg",
            rationale = (
                "Very low LR (2e-5), high WD (3e-4), long patience (10), "
                "heavy arbiter dropout (0.25). For severe overfitting: slows "
                "memorisation and regularises the site arbiter strongly."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "2e-5",
                "HYBRID_COLAB_WD":                        "3e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "60",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "10",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":  "0.25",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":   "0.008",
            },
        ),
        Strategy(
            family = FAMILY_TRAINING,
            name   = "training.aggressive_medium",
            rationale = (
                "Higher LR (1.5e-4), low WD, shorter run (40 epochs, patience 5, "
                "top1 metric). Tests whether fast convergence reveals a better "
                "optimum before the model memorises training SMILES."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "1.5e-4",
                "HYBRID_COLAB_WD":                        "5e-5",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "40",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "5",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top1",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":  "0.10",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":   "0.005",
            },
        ),
        Strategy(
            family = FAMILY_TRAINING,
            name   = "training.large_batch_moderate",
            rationale = (
                "Larger batch (32), LR=8e-5, WD=2e-4. Larger batches give a "
                "more stable gradient estimate for the council head; may help "
                "board weights learn a consistent voter preference."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "8e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "32",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":  "0.15",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":   "0.005",
            },
        ),
        Strategy(
            family = FAMILY_TRAINING,
            name   = "training.very_slow_search",
            rationale = (
                "Minimal LR (1e-5), very high WD (5e-4), long patience (12). "
                "Last-resort training policy: escapes local optima extremely "
                "slowly but with the heaviest regularisation possible."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "1e-5",
                "HYBRID_COLAB_WD":                        "5e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "60",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "12",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":  "0.30",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":   "0.010",
            },
        ),
    ]

    # ── B. data_policy ────────────────────────────────────────────────────────
    # Covers: xenosite on/off/topk, site-labeled-only, precedent logbook,
    #         nexus memory freeze, compute-xtb-if-missing.
    data = [
        Strategy(
            family = FAMILY_DATA,
            name   = "data.xenosite_off_strict",
            rationale = (
                "Disable XenoSite auxiliary training entries. XenoSite labels "
                "are noisy (top-1 predictions, not experimental); removing them "
                "may sharpen the site supervision signal."
            ),
            train_overrides = {
                "HYBRID_COLAB_INCLUDE_XENOSITE":          "0",
                "HYBRID_COLAB_SITE_LABELED_ONLY":         "1",
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":  "0.15",
            },
        ),
        Strategy(
            family = FAMILY_DATA,
            name   = "data.xenosite_topk3_rich",
            rationale = (
                "Use top-3 XenoSite predictions per molecule (XENOSITE_TOPK=3) "
                "to give the model softer multi-site supervision from the "
                "XenoSite ensemble."
            ),
            train_overrides = {
                "HYBRID_COLAB_INCLUDE_XENOSITE":          "1",
                "HYBRID_COLAB_XENOSITE_TOPK":             "3",
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {},
        ),
        Strategy(
            family = FAMILY_DATA,
            name   = "data.memory_unlock_active_learning",
            rationale = (
                "Unfreeze nexus memory (FREEZE_NEXUS_MEMORY=0) so the analogical "
                "encoder can write new cases during training. Also raise analogical "
                "aux weight to strengthen the write signal."
            ),
            train_overrides = {
                "HYBRID_COLAB_FREEZE_NEXUS_MEMORY":       "0",
                "HYBRID_COLAB_LR":                        "4e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":       "0.12",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":  "0.05",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":   "0.05",
            },
        ),
        Strategy(
            family = FAMILY_DATA,
            name   = "data.precedent_warm_start",
            rationale = (
                "Enable precedent logbook seeded from the best episode log "
                "(DISABLE_PRECEDENT_LOGBOOK=0, PRECEDENT_LOGBOOK=baseline_log). "
                "Gives analogical engine prior cases to retrieve from epoch 1."
            ),
            train_overrides = {
                "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "0",
                "HYBRID_COLAB_PRECEDENT_LOGBOOK":         baseline_log,
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":       "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":      "0.04",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":           "0.09",
            },
        ),
    ]
    # Remove precedent bundle if we have no baseline log to seed from
    if not baseline_log:
        data = [b for b in data if b.name != "data.precedent_warm_start"]

    # ── C. council_balance ────────────────────────────────────────────────────
    # Covers: lnn/wave/ana vote aux weights, board entropy, vote logit scale,
    #         arbiter dropout.  These change who the council listens to.
    council = [
        Strategy(
            family = FAMILY_COUNCIL,
            name   = "council.diversity_entropy",
            rationale = (
                "High board entropy (0.020) + low logit scale (1.5) to force "
                "the council to explore voter balance instead of defaulting to "
                "LNN. All three vote aux weights set equal."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT":          "0.03",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":         "0.03",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":   "0.03",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":          "0.020",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":              "1.5",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":          "0.15",
            },
        ),
        Strategy(
            family = FAMILY_COUNCIL,
            name   = "council.lnn_focused",
            rationale = (
                "Maximise LNN voter signal: high LNN vote aux (0.05), suppress "
                "wave and analogical vote aux (0.01 each), high logit scale (3.0). "
                "Tests whether pure LNN is the performance ceiling."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT":          "0.05",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":         "0.01",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":   "0.01",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":          "0.003",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":              "3.0",
            },
        ),
        Strategy(
            family = FAMILY_COUNCIL,
            name   = "council.wave_heavy",
            rationale = (
                "Push wave to be the dominant voter: live wave inputs, high wave "
                "vote aux (0.08) and bridge aux (0.15), moderate board entropy. "
                "Tests whether wave physics can outperform LNN SMILES memorisation."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "1",
                "HYBRID_COLAB_NEXUS_WAVE_AUX_WEIGHT":               "0.15",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.08",
                "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT":           "0.01",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":    "0.02",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":           "0.008",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":               "2.0",
            },
        ),
        Strategy(
            family = FAMILY_COUNCIL,
            name   = "council.analogical_heavy",
            rationale = (
                "Push analogical to be the dominant voter: live analogical inputs, "
                "high analogical vote aux (0.08) and bridge aux (0.12), suppressed "
                "CYP scale (0.04), moderate board entropy."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":          "1",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":          "0.12",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":     "0.08",
                "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT":            "0.01",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":           "0.02",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":      "0.04",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":            "0.008",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":                "2.0",
            },
        ),
    ]

    # ── D. analogical_system ─────────────────────────────────────────────────
    # Covers: CYP aux scale, analogical aux, live analogical vote inputs,
    #         precedent logbook, memory freeze.
    analogical = [
        Strategy(
            family = FAMILY_ANALOGICAL,
            name   = "analogical.cyp_suppressed_site_focus",
            rationale = (
                "Heavily suppress CYP aux scale (0.02) so analogical encoder "
                "cannot trade SoM accuracy for the easier CYP task. Raise site "
                "vote aux (0.06) and bridge aux (0.10) to compensate."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":      "0.02",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":          "0.10",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":     "0.06",
            },
        ),
        Strategy(
            family = FAMILY_ANALOGICAL,
            name   = "analogical.precedent_frozen_mem",
            rationale = (
                "Enable precedent logbook (seeded from baseline log) with memory "
                "still frozen. Gives analogical prior cases to retrieve from "
                "epoch 1 without destabilising the memory bank."
            ),
            train_overrides = {
                "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "0",
                "HYBRID_COLAB_PRECEDENT_LOGBOOK":         baseline_log,
                "HYBRID_COLAB_FREEZE_NEXUS_MEMORY":       "1",
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":          "0.08",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":      "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":     "0.04",
            },
        ),
        Strategy(
            family = FAMILY_ANALOGICAL,
            name   = "analogical.precedent_memory_unlock",
            rationale = (
                "Both precedent logbook AND memory unfrozen. Analogical can "
                "actively write new cases and retrieve from them. High aux (0.12) "
                "to push the encoder to learn useful representations."
            ),
            train_overrides = {
                "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "0",
                "HYBRID_COLAB_PRECEDENT_LOGBOOK":         baseline_log,
                "HYBRID_COLAB_FREEZE_NEXUS_MEMORY":       "0",
                "HYBRID_COLAB_LR":                        "4e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "55",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "8",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":          "0.12",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":      "0.04",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":     "0.06",
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":           "1",
            },
        ),
        Strategy(
            family = FAMILY_ANALOGICAL,
            name   = "analogical.live_calibrate",
            rationale = (
                "Un-detach analogical encoder for the vote head (LIVE_ANALOGICAL "
                "=1). Vote head now sees up-to-date analogical features; breaks "
                "the stale-mapping dead-confidence cycle."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":           "1",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":      "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":       "0.04",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":           "0.09",
            },
        ),
    ]
    # Remove precedent bundles if no baseline log
    if not baseline_log:
        analogical = [b for b in analogical if "precedent" not in b.name]

    # ── E. wave_system ────────────────────────────────────────────────────────
    # Covers: wave bridge aux, wave vote aux, live wave vote inputs.
    wave = [
        Strategy(
            family = FAMILY_WAVE,
            name   = "wave.live_vote_full",
            rationale = (
                "Un-detach wave encoder for vote head (LIVE_WAVE=1) + high wave "
                "vote aux (0.07). Fixes the stale-mapping problem: vote head sees "
                "current wave features instead of a frozen snapshot."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "1",
                "HYBRID_COLAB_NEXUS_WAVE_AUX_WEIGHT":               "0.12",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.07",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":               "2.0",
            },
        ),
        Strategy(
            family = FAMILY_WAVE,
            name   = "wave.evidence_only_strong",
            rationale = (
                "Wave as evidence provider, not a direct voter: detached inputs "
                "(LIVE_WAVE=0), very low vote aux (0.01), high bridge aux (0.15). "
                "Tests whether wave improves LNN via board_context without "
                "being a noisy voter."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "6",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "0",
                "HYBRID_COLAB_NEXUS_WAVE_AUX_WEIGHT":               "0.15",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.01",
            },
        ),
        Strategy(
            family = FAMILY_WAVE,
            name   = "wave.vote_dominant",
            rationale = (
                "Make wave the dominant voter: live wave inputs, maximum wave "
                "vote aux (0.10), suppressed LNN/analogical vote aux (0.01 each). "
                "Tests the raw quality of the wave physics predictions."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "1",
                "HYBRID_COLAB_NEXUS_WAVE_AUX_WEIGHT":               "0.12",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.10",
                "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT":           "0.01",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":    "0.01",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":           "0.010",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":               "2.5",
            },
        ),
    ]

    # ── F. architecture ───────────────────────────────────────────────────────
    # Structural changes: live both voters, full system unlock, etc.
    # Used primarily in stuck mode but also available as fallback.
    architecture = [
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.compact_context_selective",
            rationale = (
                "Shrink the sidecar context: smaller wave hidden dim (48), graph "
                "dim (32), arbiter hidden dim (96), and retrieval top-k (12). "
                "Tests whether the current council is over-parameterised and "
                "memorising instead of learning selective analogies."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_WAVE_HIDDEN_DIM":             "48",
                "HYBRID_COLAB_NEXUS_GRAPH_DIM":                   "32",
                "HYBRID_COLAB_NEXUS_MEMORY_TOPK":                 "12",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_HIDDEN_DIM":     "96",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":        "0.18",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":        "0.008",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.wide_context_memory_rich",
            rationale = (
                "Expand the sidecar context: larger wave hidden dim (96), graph "
                "dim (64), memory capacity (6144), retrieval top-k (48), and "
                "arbiter hidden dim (192). Tests whether the council is currently "
                "capacity-limited rather than calibration-limited."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "3e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "60",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "9",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_WAVE_HIDDEN_DIM":             "96",
                "HYBRID_COLAB_NEXUS_GRAPH_DIM":                   "64",
                "HYBRID_COLAB_NEXUS_MEMORY_CAPACITY":             "6144",
                "HYBRID_COLAB_NEXUS_MEMORY_TOPK":                 "48",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_HIDDEN_DIM":     "192",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":        "0.15",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":        "0.006",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.memory_small_sharp_neighbors",
            rationale = (
                "Force sharper analogical retrieval with a smaller memory bank "
                "(2048) and smaller top-k (8). Reduces precedent dilution and "
                "pushes the analogical branch to commit to fewer neighbors."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "4e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "55",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "8",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_MEMORY_CAPACITY":             "2048",
                "HYBRID_COLAB_NEXUS_MEMORY_TOPK":                 "8",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":       "0.10",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":  "0.05",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":    "0.04",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":        "0.18",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.prior_biased_bridge",
            rationale = (
                "Change the bridge priors directly: stronger initial wave site "
                "bias (0.24) and analogical site bias (0.30), but lower initial "
                "analogical CYP gate (0.08). Tests whether the current council "
                "starts too CYP-biased and too weak on site proposals."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_WAVE_SITE_INIT":              "0.24",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_SITE_INIT":        "0.30",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_INIT":         "0.08",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":    "0.04",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_HIDDEN_DIM":     "160",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.live_grad_gentle",
            rationale = (
                "Use live wave and analogical vote inputs with very gentle gradient "
                "leak (0.02 each). Lets the voters adapt to current sidecar "
                "features without reopening the batch-1 NaN instability."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "3e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "55",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "8",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":                 "1",
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":            "1",
                "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE":        "0.02",
                "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE":  "0.02",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":            "0.05",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":      "0.05",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":            "0.18",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.live_grad_stronger",
            rationale = (
                "Use live wave and analogical vote inputs with stronger gradient "
                "leak (wave 0.10, analogical 0.08) plus a wider arbiter. Tests "
                "whether the council is underfitting because the sidecar vote "
                "heads learn too slowly from detached features."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "2e-5",
                "HYBRID_COLAB_WD":                        "3e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "60",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "9",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":                 "1",
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":            "1",
                "HYBRID_COLAB_NEXUS_LIVE_WAVE_VOTE_GRAD_SCALE":        "0.10",
                "HYBRID_COLAB_NEXUS_LIVE_ANALOGICAL_VOTE_GRAD_SCALE":  "0.08",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":            "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":      "0.06",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_HIDDEN_DIM":         "192",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":            "0.20",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.both_live_voters",
            rationale = (
                "Un-detach BOTH wave and analogical encoders for their vote heads "
                "(LIVE_WAVE=1, LIVE_ANALOGICAL=1). All three voters receive "
                "up-to-date features; board weights must learn a real preference."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "3e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_EPOCHS":                    "55",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "8",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "1",
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":          "1",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":    "0.05",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":     "0.04",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":           "0.012",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":               "2.0",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":           "0.15",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.full_system_unlock",
            rationale = (
                "Full system unlock: live both voters, memory unfrozen, precedent "
                "logbook enabled, CYP suppressed, high board entropy. Maximum "
                "freedom for all three voters to learn useful representations."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "3e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_EPOCHS":                    "60",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "9",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
                "HYBRID_COLAB_FREEZE_NEXUS_MEMORY":       "0",
                "HYBRID_COLAB_DISABLE_PRECEDENT_LOGBOOK": "0",
                "HYBRID_COLAB_PRECEDENT_LOGBOOK":         baseline_log,
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "1",
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":          "1",
                "HYBRID_COLAB_NEXUS_WAVE_AUX_WEIGHT":               "0.12",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":          "0.12",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":    "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":     "0.03",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":           "0.015",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":               "2.0",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":           "0.20",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.lnn_dominant_ceiling",
            rationale = (
                "Suppress wave and analogical vote aux to near-zero (0.005) and "
                "give LNN full control (lnn_vote_aux=0.05, logit_scale=3.5). "
                "Measures the LNN-alone performance ceiling."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "1e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT":          "0.05",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":         "0.005",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":   "0.005",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":              "3.5",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":          "0.003",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.slow_full_regularized",
            rationale = (
                "Slow deep search: very low LR (1e-5), high WD (5e-4), long "
                "patience (12), live both voters, memory frozen, high dropout "
                "(0.30) and entropy (0.020). Maximum regularisation + maximum "
                "information from all voters."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "1e-5",
                "HYBRID_COLAB_WD":                        "5e-4",
                "HYBRID_COLAB_BATCH_SIZE":                "16",
                "HYBRID_COLAB_EPOCHS":                    "60",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "12",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "1",
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":          "1",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.06",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":    "0.05",
                "HYBRID_COLAB_NEXUS_BOARD_ENTROPY_WEIGHT":           "0.020",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":               "1.5",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":           "0.30",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_CYP_AUX_SCALE":     "0.04",
            },
        ),
        Strategy(
            family = FAMILY_ARCHITECTURE,
            name   = "arch.wave_ana_evidence_lnn_vote",
            rationale = (
                "Hybrid evidence mode: wave and analogical as evidence-only "
                "(very low vote aux), LNN as sole voter (high vote aux + logit "
                "scale), but all bridge aux weights kept high so the arbiter "
                "still benefits from wave/analogical context."
            ),
            train_overrides = {
                "HYBRID_COLAB_LR":                        "5e-5",
                "HYBRID_COLAB_WD":                        "2e-4",
                "HYBRID_COLAB_EPOCHS":                    "50",
                "HYBRID_COLAB_EARLY_STOPPING_PATIENCE":   "7",
                "HYBRID_COLAB_EARLY_STOPPING_METRIC":     "site_top3",
            },
            model_env = {
                "HYBRID_COLAB_LIVE_WAVE_VOTE_INPUTS":               "0",
                "HYBRID_COLAB_LIVE_ANALOGICAL_VOTE_INPUTS":          "0",
                "HYBRID_COLAB_NEXUS_WAVE_AUX_WEIGHT":               "0.15",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_AUX_WEIGHT":          "0.12",
                "HYBRID_COLAB_NEXUS_LNN_VOTE_AUX_WEIGHT":           "0.05",
                "HYBRID_COLAB_NEXUS_WAVE_VOTE_AUX_WEIGHT":          "0.005",
                "HYBRID_COLAB_NEXUS_ANALOGICAL_VOTE_AUX_WEIGHT":    "0.005",
                "HYBRID_COLAB_NEXUS_VOTE_LOGIT_SCALE":               "3.0",
                "HYBRID_COLAB_NEXUS_SITE_ARBITER_DROPOUT":           "0.15",
            },
        ),
    ]
    # Remove bundles that reference baseline_log if none exists
    if not baseline_log:
        architecture = [b for b in architecture if "precedent" not in b.name and b.name != "arch.full_system_unlock"]

    return {
        FAMILY_TRAINING:     training,
        FAMILY_DATA:         data,
        FAMILY_COUNCIL:      council,
        FAMILY_ANALOGICAL:   analogical,
        FAMILY_WAVE:         wave,
        FAMILY_ARCHITECTURE: architecture,
    }


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
    tried_names:  set[str] | None = None,
) -> list[Strategy]:
    """
    Return an ordered list of Strategy bundles to try next.

    Normal mode (stuck=False)
    -------------------------
    Evidence-driven selection across all families.  Each triggered condition
    picks the highest-priority untried bundle from its family.  After filling
    triggered slots, the remaining quota is filled from the full catalogue in
    priority order.

    Stuck mode (stuck=True)
    -----------------------
    Skip normal-mode families and go directly to FAMILY_ARCHITECTURE bundles,
    then fill any remaining slots with data_policy and training_policy bundles
    not yet tried.  Architecture bundles make structural changes (live inputs,
    full unlock, evidence-only modes) that normal bundles don't touch.

    Family priority within normal mode (higher = tried earlier):
        1. analogical_system    (if ana dead / CYP dominant / comp frozen)
        2. data_policy          (if precedent never used / memory frozen)
        3. training_policy      (if severe overfitting)
        4. wave_system          (if wave degrading or wave voter weak)
        5. council_balance      (if board frozen / voter imbalance)
        6. architecture         (always included as fallback)
    """
    tried = tried_names or set()
    bundles = _all_bundles(baseline_log)

    def _pick(family: str, index: int = 0) -> Strategy | None:
        """Return the `index`-th untried bundle from `family`."""
        avail = [b for b in bundles[family] if b.name not in tried]
        return avail[index] if index < len(avail) else None

    def _all_untried(family: str) -> list[Strategy]:
        return [b for b in bundles[family] if b.name not in tried]

    def _ef(fn) -> float:
        return _frac_positive(all_diags, fn)

    valid_diags = [d for d in all_diags if d.get("valid")]

    # ── stuck mode: architecture family first ─────────────────────────────────
    if stuck:
        selected: list[Strategy] = []
        # Architecture bundles: structural changes not yet tried
        selected.extend(_all_untried(FAMILY_ARCHITECTURE))
        # Then data policy (xenosite off, memory unlock, precedent)
        selected.extend(_all_untried(FAMILY_DATA))
        # Then training policy extremes
        selected.extend(_all_untried(FAMILY_TRAINING))
        # Then wave and council
        selected.extend(_all_untried(FAMILY_WAVE))
        selected.extend(_all_untried(FAMILY_COUNCIL))
        selected.extend(_all_untried(FAMILY_ANALOGICAL))
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[Strategy] = []
        for s in selected:
            if s.name not in seen:
                seen.add(s.name)
                result.append(s)
        return result[:iterations]

    # ── normal mode: evidence-driven slot allocation ──────────────────────────

    # Compute evidence fractions
    overfitting_frac  = _ef(lambda d: d.get("overfitting_gap", 0.0) > 0.50)
    severe_overfit    = _ef(lambda d: d.get("overfitting_gap", 0.0) > 0.65)
    ana_dead_frac     = _ef(lambda d: d.get("ana_dead", False))
    comp_frozen_frac  = _ef(lambda d: d.get("comp_wt_frozen", False))
    wave_degrade_frac = _ef(lambda d: d.get("wave_vote_degrading", False))
    ana_cyp_frac      = _ef(lambda d: d.get("ana_cyp_dominant", False))
    board_frozen_frac = _ef(lambda d: d.get("board_lnn_var", 1.0) < 0.005)
    precedent_never   = not any(d.get("precedent_nonzero", False) for d in valid_diags)

    # Average voter accs across valid logs
    def _avg_voter(key: str) -> float:
        vals = [d.get("voter_accs", {}).get(key, 0.0) for d in valid_diags]
        return mean(vals) if vals else 0.0

    lnn_acc  = _avg_voter("lnn")
    wave_acc = _avg_voter("wave")
    ana_acc  = _avg_voter("ana")

    # Ordered trigger → (family, bundle_index) pairs
    # The trigger conditions are checked in priority order; each adds one bundle.
    triggered: list[tuple[str, int]] = []  # (family, index)

    # ① Analogical dead and precedent never used → highest priority fix
    if ana_dead_frac > 0.4 and precedent_never:
        triggered.append((FAMILY_ANALOGICAL, 1))   # analogical.precedent_frozen_mem
    # ② Analogical CYP domination → suppress CYP, raise site
    if ana_cyp_frac > 0.4:
        triggered.append((FAMILY_ANALOGICAL, 0))   # analogical.cyp_suppressed_site_focus
    # ③ Competition weight frozen → unlock memory
    if comp_frozen_frac > 0.4:
        triggered.append((FAMILY_DATA, 2))          # data.memory_unlock_active_learning
    # ④ Severe overfitting → heavy regularisation training bundle
    if severe_overfit > 0.4:
        triggered.append((FAMILY_TRAINING, 0))      # training.slow_patient_heavy_reg
    elif overfitting_frac > 0.4:
        triggered.append((FAMILY_TRAINING, 0))      # same, less severe condition
    # ⑤ Wave voter degrading or much weaker than LNN
    if wave_degrade_frac > 0.3 or (lnn_acc > 0 and wave_acc < lnn_acc * 0.6):
        triggered.append((FAMILY_WAVE, 0))          # wave.live_vote_full
    # ⑥ Board weights frozen → entropy exploration
    if board_frozen_frac > 0.4:
        triggered.append((FAMILY_COUNCIL, 0))       # council.diversity_entropy
    # ⑦ Analogical competitive with LNN → give it more weight
    if ana_acc > 0 and lnn_acc > 0 and ana_acc > lnn_acc * 0.85:
        triggered.append((FAMILY_COUNCIL, 3))       # council.analogical_heavy
    # ⑧ No precedent ever used but analogical not dead yet → warm-start logbook
    if precedent_never and ana_dead_frac < 0.4:
        triggered.append((FAMILY_DATA, 3))          # data.precedent_warm_start

    # Resolve triggered slots to actual bundles, deduplicate
    candidates: list[Strategy] = []
    seen_names: set[str] = set()

    for family, idx in triggered:
        bundle = _pick(family, idx)
        if bundle is not None and bundle.name not in seen_names:
            seen_names.add(bundle.name)
            candidates.append(bundle)

    # Fill remaining quota from all families in priority order
    fill_order = [
        FAMILY_ANALOGICAL,
        FAMILY_DATA,
        FAMILY_TRAINING,
        FAMILY_WAVE,
        FAMILY_COUNCIL,
        FAMILY_ARCHITECTURE,
    ]
    for family in fill_order:
        for bundle in bundles[family]:
            if len(candidates) >= iterations:
                break
            if bundle.name not in seen_names:
                seen_names.add(bundle.name)
                candidates.append(bundle)
        if len(candidates) >= iterations:
            break

    return candidates[:iterations]


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
        "--split-mode",           settings["HYBRID_COLAB_SPLIT_MODE"],
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
        "HYBRID_COLAB_SPLIT_MODE":        os.environ.get("HYBRID_COLAB_SPLIT_MODE",        "scaffold_source_size"),
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
            tried_names=tried_strategy_names,
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

if __name__ == "__main__":
    main()
