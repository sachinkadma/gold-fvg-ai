"""
Pre-Registered Out-of-Sample Evaluation Engine for XAUUSD 15m ICT Strategy.
Enforces strict anti-overfitting protocol:
- Combined dry-run mode (2015-2021) for verification.
- TEST split (2022+) execution strictly locked behind the --final flag.
- Decision rule: PASS / FAIL / INCONCLUSIVE.
- Persistent audit logging to test_runs.log.
"""

import os
import sys
import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import config
from core.data_loader import load_and_clean_data
from core.signals import is_in_killzone
from research.meta_labeling.pipeline_metalabeling import generate_candidates_and_features

PREREG_DOC_PATH = os.path.join(REPO_ROOT, "research", "PREREG_TEST.md")
LOG_PATH = os.path.join(REPO_ROOT, "test_runs.log")
DATA_PATH = os.path.join(REPO_ROOT, "data", "xauusd_15m_long.csv")
CANDIDATES_CACHE_PATH = os.path.join(REPO_ROOT, "data", "candidates.parquet")

DEFAULT_P_SCALED = 0.6077  # 8.0 / 1316.4990 * 100%


def get_file_sha256(filepath):
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        return res.stdout.strip() if res.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def compute_parameters_hash(p_scaled):
    param_str = (
        f"STRATEGY=ICT_CAUSAL_SWEEP_FVG;"
        f"KILLZONES=LONDON_0200_0500_NY_AM_0700_1000;"
        f"P_SCALED={p_scaled:.4f};"
        f"RR=2.0;"
        f"SPREAD={config.SPREAD};"
        f"SLIPPAGE={config.SLIPPAGE};"
        f"COMMISSION={config.COMMISSION_PER_LOT};"
        f"ORDER_EXPIRY={config.ORDER_EXPIRY_BARS};"
        f"COLLISION=SL_FIRST"
    )
    return hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]


def block_bootstrap_ci(r_vals, month_periods, n_resamples=2000, seed=42):
    """Computes 95% monthly block bootstrap confidence interval."""
    if len(r_vals) < 5 or len(month_periods) == 0:
        mean_val = float(np.mean(r_vals)) if len(r_vals) > 0 else 0.0
        return (mean_val, mean_val)

    df_tmp = pd.DataFrame({"r": r_vals, "month": month_periods})
    monthly_groups = [group["r"].values for _, group in df_tmp.groupby("month")]
    n_blocks = len(monthly_groups)

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_resamples):
        sampled_blocks = rng.choice(n_blocks, size=n_blocks, replace=True)
        sampled_trades = np.concatenate([monthly_groups[i] for i in sampled_blocks])
        boot_means.append(np.mean(sampled_trades))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return round(ci_lower, 3), round(ci_upper, 3)


def evaluate_setups(df_cand, p_scaled=DEFAULT_P_SCALED):
    """
    Evaluates filled candidates under:
    1. Primary Model: Killzone Only (London & NY AM) + Stop >= P% of close
    2. Secondary Diagnostic: Killzone Only + Fixed Stop >= 8.0 pts
    """
    df_filled = df_cand[df_cand["filled"] == True].copy()
    if df_filled.empty:
        return None, None

    # Compute percentage stop distance relative to signal close
    df_filled["stop_pct"] = (df_filled["stop_dist"] / df_filled["close"]) * 100.0
    df_filled["month_period"] = pd.to_datetime(df_filled["signal_time"]).dt.tz_localize(None).dt.to_period("M")

    # 1. Primary Model Filter: Killzone (killzone > 0) AND stop_pct >= p_scaled
    f_primary = (df_filled["killzone"] > 0) & (df_filled["stop_pct"] >= p_scaled)
    primary_trades = df_filled[f_primary].copy()

    # 2. Secondary Benchmark: Killzone AND stop_dist >= 8.0 pts
    f_secondary = (df_filled["killzone"] > 0) & (df_filled["stop_dist"] >= 8.0)
    secondary_trades = df_filled[f_secondary].copy()

    def summarize(trades_sub):
        n = len(trades_sub)
        if n == 0:
            return {
                "n_trades": 0, "win_rate_pct": 0.0, "expectancy_r": 0.0,
                "ci_95": (0.0, 0.0), "total_r": 0.0, "profit_factor": 0.0, "trades": trades_sub
            }
        r_vals = trades_sub["net_r_20"].values
        wins = r_vals[r_vals > 0]
        losses = r_vals[r_vals <= 0]
        wr = (len(wins) / n) * 100.0
        exp = float(np.mean(r_vals))
        tot = float(np.sum(r_vals))
        gross_gains = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_losses = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
        pf = (gross_gains / gross_losses) if gross_losses > 0 else 0.0
        ci = block_bootstrap_ci(r_vals, trades_sub["month_period"].values)

        return {
            "n_trades": n,
            "win_rate_pct": round(wr, 1),
            "expectancy_r": round(exp, 3),
            "ci_95": ci,
            "total_r": round(tot, 2),
            "profit_factor": round(pf, 2),
            "trades": trades_sub
        }

    return summarize(primary_trades), summarize(secondary_trades)


def run_combined_dry_run(p_scaled=DEFAULT_P_SCALED):
    """
    Executes dry run on COMBINED in-sample / validation data (2015-2021).
    Must reproduce Trial 23 byte-for-byte.
    """
    print("=" * 70)
    print(f"PRE-REGISTRATION DRY-RUN: COMBINED SET (2015–2021)")
    print(f"Primary Rule: Killzone Only (02-05, 07-10 NY) + Stop >= {p_scaled:.4f}% of close")
    print("=" * 70)

    if not os.path.exists(CANDIDATES_CACHE_PATH):
        print(f"ERROR: Cached candidates not found at {CANDIDATES_CACHE_PATH}")
        sys.exit(1)

    df_cand = pd.read_parquet(CANDIDATES_CACHE_PATH)
    df_bars = pd.read_csv(DATA_PATH)
    closes = df_bars["close"].values
    df_cand["close"] = closes[df_cand["signal_bar"].values]

    # Filter to 2015-2021
    df_cand_comb = df_cand[(df_cand["year"] >= 2015) & (df_cand["year"] <= 2021)].copy()

    primary_res, secondary_res = evaluate_setups(df_cand_comb, p_scaled=p_scaled)

    print(f"\n[PRIMARY MODEL RESULTS (Trial 23 Benchmark)]")
    print(f"  Executed Trades (N) : {primary_res['n_trades']}")
    print(f"  Win Rate            : {primary_res['win_rate_pct']:.1f}%")
    print(f"  Net Expectancy      : {primary_res['expectancy_r']:+.3f} R")
    print(f"  Total Net R         : {primary_res['total_r']:+.2f} R")
    print(f"  Profit Factor       : {primary_res['profit_factor']:.2f}")
    print(f"  Monthly-Block 95% CI: [{primary_res['ci_95'][0]:+.3f} R, {primary_res['ci_95'][1]:+.3f} R]")

    print(f"\n[SECONDARY BENCHMARK (Fixed SL >= 8.0 pts)]")
    print(f"  Executed Trades (N) : {secondary_res['n_trades']}")
    print(f"  Win Rate            : {secondary_res['win_rate_pct']:.1f}%")
    print(f"  Net Expectancy      : {secondary_res['expectancy_r']:+.3f} R")
    print(f"  Total Net R         : {secondary_res['total_r']:+.2f} R")
    print(f"  Monthly-Block 95% CI: [{secondary_res['ci_95'][0]:+.3f} R, {secondary_res['ci_95'][1]:+.3f} R]")

    # Verification against Trial 23 targets
    match_trades = (primary_res['n_trades'] == 62)
    match_exp = (abs(primary_res['expectancy_r'] - 0.408) < 1e-3)
    match_ci = (primary_res['ci_95'] == (0.007, 0.824))

    print("\n" + "-" * 70)
    if match_trades and match_exp and match_ci:
        print("VERIFICATION CHECK: PASS (Exact match with Trial 23: 62 trades, +0.408R, CI [+0.007, +0.824])")
        print("-" * 70 + "\n")
        return True
    else:
        print("VERIFICATION CHECK: FAIL (Discrepancy detected with Trial 23)")
        print(f"  Trades match: {match_trades} (got {primary_res['n_trades']}, expected 62)")
        print(f"  Exp match   : {match_exp} (got {primary_res['expectancy_r']}, expected +0.408)")
        print(f"  CI match    : {match_ci} (got {primary_res['ci_95']}, expected (0.007, 0.824))")
        print("-" * 70 + "\n")
        return False


def run_test_evaluation(p_scaled=DEFAULT_P_SCALED, is_final=False):
    """
    Executes one-shot evaluation on the locked TEST set (2022-01-01 to latest).
    Strictly guarded behind the --final flag.
    """
    if not is_final:
        print("\n" + "!" * 70)
        print("SECURITY LOCK ENFORCED:")
        print("Evaluation on the locked TEST split (2022+) requires the explicit '--final' flag.")
        print("Candidate generation and simulation on TEST have been blocked.")
        print("!" * 70 + "\n")
        sys.exit(1)

    print("=" * 70)
    print("ONE-SHOT TEST EVALUATION (2022-01-01 TO LATEST)")
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # 1. Load data with warmup
    print("[DATA] Loading historical data for TEST split...")
    df_all = load_and_clean_data(DATA_PATH, data_tz="UTC", verbose=False)

    # Include warmup from 2021-12-01 so 5-bar swings, ATR, EMA are fully populated
    df_test_full = df_all.loc["2021-12-01":].copy()
    print(f"[DATA] Total candles loaded (including Dec 2021 warmup): {len(df_test_full):,}")

    # 2. Generate TEST candidates dynamically
    print("[CANDIDATE GENERATION] Generating causal setups on out-of-sample data...")
    test_candidates = generate_candidates_and_features(df_test_full)

    if not test_candidates:
        print("ERROR: No candidates generated on TEST data.")
        sys.exit(1)

    df_cand_test = pd.DataFrame(test_candidates)
    closes = df_test_full["close"].values
    df_cand_test["close"] = closes[df_cand_test["signal_bar"].values]

    # Filter strictly to signals occurring on or after 2022-01-01
    df_cand_test = df_cand_test[pd.to_datetime(df_cand_test["signal_time"]) >= "2022-01-01"].copy()
    print(f"[CANDIDATE GENERATION] Out-of-sample candidate pool: {len(df_cand_test):,} candidates.")

    # 3. Evaluate setups
    primary_res, secondary_res = evaluate_setups(df_cand_test, p_scaled=p_scaled)

    n_test = primary_res["n_trades"]
    exp_test = primary_res["expectancy_r"]
    ci_lower, ci_upper = primary_res["ci_95"]
    wr_test = primary_res["win_rate_pct"]
    tot_test = primary_res["total_r"]

    # 4. Apply Decision Rule
    # PASS: N >= 60 AND exp > 0 AND ci_lower > 0
    # FAIL: exp <= 0 (N >= 30)
    # INCONCLUSIVE: everything else (exp > 0 but CI includes 0, or N < 60)
    if n_test >= 60 and exp_test > 0.0 and ci_lower > 0.0:
        verdict = "PASS"
        verdict_detail = "Positive expectancy confirmed with 95% statistical significance and adequate sample."
    elif exp_test <= 0.0 and n_test >= 30:
        verdict = "FAIL"
        verdict_detail = "Zero or negative net expectancy observed on adequate sample."
    else:
        verdict = "INCONCLUSIVE"
        if n_test < 60:
            verdict_detail = f"Sample underpowered (N={n_test} < 60 required). Forward paper-trading required."
        else:
            verdict_detail = f"Positive expectancy observed ({exp_test:+.3f}R), but 95% CI [{ci_lower:+.3f}, {ci_upper:+.3f}] spans zero."

    # Print Report
    print("\n" + "=" * 70)
    print(f"OFFICIAL TEST OUTCOME: {verdict}")
    print(f"Detail: {verdict_detail}")
    print("=" * 70)
    print(f"  Primary Model (P = {p_scaled:.4f}%):")
    print(f"    Executed Trades (N) : {n_test}")
    print(f"    Win Rate            : {wr_test:.1f}%")
    print(f"    Net Expectancy      : {exp_test:+.3f} R")
    print(f"    Total Net R         : {tot_test:+.2f} R")
    print(f"    Profit Factor       : {primary_res['profit_factor']:.2f}")
    print(f"    Monthly-Block 95% CI: [{ci_lower:+.3f} R, {ci_upper:+.3f} R]")

    print(f"\n  Secondary Diagnostic (Fixed SL >= 8.0 pts):")
    print(f"    Executed Trades (N) : {secondary_res['n_trades']}")
    print(f"    Win Rate            : {secondary_res['win_rate_pct']:.1f}%")
    print(f"    Net Expectancy      : {secondary_res['expectancy_r']:+.3f} R")
    print(f"    Total Net R         : {secondary_res['total_r']:+.2f} R")
    print(f"    Monthly-Block 95% CI: [{secondary_res['ci_95'][0]:+.3f} R, {secondary_res['ci_95'][1]:+.3f} R]")
    print("=" * 70 + "\n")

    # 5. Log run to test_runs.log
    commit_hash = get_git_commit()
    param_hash = compute_parameters_hash(p_scaled)
    prereg_hash = get_file_sha256(PREREG_DOC_PATH)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    log_entry = (
        f"[{timestamp_utc}] VERDICT={verdict} | SPLIT=TEST (2022+) | "
        f"COMMIT={commit_hash} | PARAMS_HASH={param_hash} | PREREG_HASH={prereg_hash[:16]} | "
        f"N={n_test} | WR={wr_test:.1f}% | EXP={exp_test:+.3f}R | CI_95=[{ci_lower:+.3f},{ci_upper:+.3f}] | "
        f"DETAIL={verdict_detail}\n"
    )

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(log_entry)

    print(f"[AUDIT] Run recorded to {LOG_PATH}.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-Registered Out-of-Sample Evaluation Engine")
    parser.add_argument("--split", choices=["combined", "test"], default="combined", help="Data partition to evaluate")
    parser.add_argument("--p-scaled", type=float, default=DEFAULT_P_SCALED, help="Scaled stop floor percent (default 0.6077)")
    parser.add_argument("--final", action="store_true", help="Explicit authorization flag required for TEST split")

    args = parser.parse_args()

    if args.split == "combined":
        success = run_combined_dry_run(p_scaled=args.p_scaled)
        sys.exit(0 if success else 1)
    elif args.split == "test":
        run_test_evaluation(p_scaled=args.p_scaled, is_final=args.final)


if __name__ == "__main__":
    main()
