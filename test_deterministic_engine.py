import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import zoneinfo
from deriv_client import (
    detect_swing_points,
    detect_displacement_candles,
    detect_3candle_fvgs,
    compute_deterministic_market_structure
)

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

def generate_test_candle_df():
    """Generates a synthetic 30-candle DataFrame with known ICT structure."""
    base_time = datetime(2026, 11, 10, 9, 0, tzinfo=NY_TZ)
    data = []
    
    # 0-5: Initial range (Swing High at index 3: 4125.0, Swing Low at index 1: 4110.0)
    prices = [
        (4115.0, 4118.0, 4112.0, 4116.0),
        (4116.0, 4117.0, 4108.0, 4110.0), # Swing low candidate: low 4108
        (4110.0, 4120.0, 4109.0, 4118.0),
        (4118.0, 4128.0, 4117.0, 4126.0), # Swing high candidate: high 4128
        (4126.0, 4127.0, 4120.0, 4122.0),
        (4122.0, 4124.0, 4118.0, 4120.0),
    ]
    
    # Add downward run to sweep the low at 4108
    # 6-9: Push down
    prices += [
        (4120.0, 4121.0, 4114.0, 4115.0),
        (4115.0, 4116.0, 4110.0, 4111.0),
        # Candle 8: SSL Sweep of 4108! Low = 4105.0, but Close = 4110.0 (Wick sweep!)
        (4111.0, 4112.0, 4105.0, 4110.0),
        # Candle 9: Reversal displacement up
        (4110.0, 4122.0, 4109.0, 4121.0), # Body: 11, Range: 13, ratio: 0.84 (Displacement!)
        # Candle 10: Continues up, creating 3-candle FVG between Candle 8 High (4112.0) and Candle 10 Low (4115.0)
        (4121.0, 4132.0, 4116.0, 4130.0), # MSS breaking 4128!
        (4130.0, 4133.0, 4128.0, 4131.0),
        (4131.0, 4132.0, 4127.0, 4129.0),
    ]

    for i, (o, h, l, c) in enumerate(prices):
        data.append({
            "Date_NY": base_time + timedelta(minutes=15 * i),
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c
        })

    return pd.DataFrame(data)

def run_tests():
    print("=== RUNNING DETERMINISTIC ALGORITHMIC ENGINE TESTS ===")
    df = generate_test_candle_df()
    print(f"Generated {len(df)} synthetic candles.")

    # 1. Test Swing Points
    shs, sls = detect_swing_points(df, window=5)
    print(f"Detected Swing Highs: {len(shs)} -> {[s['price'] for s in shs]}")
    print(f"Detected Swing Lows: {len(sls)} -> {[s['price'] for s in sls]}")
    assert any(s['price'] >= 4126.0 for s in shs), "Swing high detection failed!"
    assert any(s['price'] <= 4110.0 for s in sls), "Swing low detection failed!"

    # 2. Test Displacement Filter
    is_disp, ratios = detect_displacement_candles(df, min_body_ratio=0.70)
    print(f"Displacement candles detected: {int(np.sum(is_disp))} / {len(df)}")
    assert np.max(ratios) >= 0.70, "Displacement ratio filter failed!"

    # 3. Test 3-Candle FVG & 50% CE
    fvgs = detect_3candle_fvgs(df, min_gap_pts=0.40)
    print(f"Detected FVGs: {len(fvgs)}")
    for f in fvgs:
        print(f"  -> {f['type']}: Top=${f['fvg_top']}, Bot=${f['fvg_bot']}, 50% CE=${f['ce']}, Mitigated={f['mitigated']}")
    assert len(fvgs) > 0, "3-Candle FVG detection failed!"

    # 4. Test Complete Deterministic Market Structure
    ms = compute_deterministic_market_structure(df, account_balance=10000.0, risk_percent=1.0)
    print("\nDeterministic Market Structure Output:")
    for k in ["has_setup", "action", "bias", "verdict", "sweep_type", "sweep_level", "mss_confirmed", "entry_ce", "stop_loss", "take_profit_1", "risk_reward_ratio", "recommended_lot"]:
        print(f"  {k}: {ms.get(k)}")

    # Strict RR check
    if ms.get("has_setup"):
        assert ms["risk_reward_ratio"] >= 2.0, "Risk reward ratio is below 2.0!"
        assert ms["action"] == "EXECUTE_LIMIT", "Action must be EXECUTE_LIMIT when has_setup is True!"
    
    print("\n[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
