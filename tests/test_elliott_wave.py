import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import zoneinfo

NY_TZ = zoneinfo.ZoneInfo("America/New_York")

def build_mock_df_and_pivots(pivots_list, current_close):
    """
    Constructs a minimal DataFrame and confirmed swing_highs/lows
    from a list of (index, price, 'high'|'low') tuples.
    """
    n = max(p[0] for p in pivots_list) + 5
    base_time = datetime(2026, 11, 10, 8, 0, tzinfo=NY_TZ)
    
    rows = []
    for i in range(n):
        rows.append({
            "Date_NY": base_time + timedelta(minutes=15 * i),
            "Open": 4120.0,
            "High": 4125.0,
            "Low": 4115.0,
            "Close": current_close if i == n - 1 else 4120.0
        })
    df = pd.DataFrame(rows)

    swing_highs = []
    swing_lows = []
    for idx, price, p_type in pivots_list:
        df.loc[idx, "High" if p_type == "high" else "Low"] = price
        if p_type == "high":
            swing_highs.append({"index": idx, "price": float(price), "time": df["Date_NY"].iloc[idx]})
        else:
            swing_lows.append({"index": idx, "price": float(price), "time": df["Date_NY"].iloc[idx]})

    return df, swing_highs, swing_lows

def run_ewt_unit_tests():
    import sys, os
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from legacy.deriv_client import detect_elliott_wave_cycle
    except ImportError:
        from deriv_client import detect_elliott_wave_cycle
    print("=== RUNNING DETERMINISTIC ELLIOTT WAVE (EWT) UNIT TESTS ===")

    # 1. TEST BULLISH 5-WAVE IMPULSE (All 3 Cardinal Rules Pass)
    # P0 (low 4100), P1 (high 4120), P2 (low 4110), P3 (high 4150), P4 (low 4130), P5 (high 4165)
    # W1 = 20, W3 = 40, W5 = 35. W3 is longest (40). P2 (4110) > P0 (4100). P4 (4130) > P1 (4120).
    pivots_valid = [
        (2, 4100.0, "low"),
        (5, 4120.0, "high"),
        (8, 4110.0, "low"),
        (12, 4150.0, "high"),
        (16, 4130.0, "low"),
        (20, 4165.0, "high")
    ]
    df1, shs1, sls1 = build_mock_df_and_pivots(pivots_valid, current_close=4158.0)
    ewt1 = detect_elliott_wave_cycle(df1, shs1, sls1)
    
    print("\n[Test 1: Valid Bullish 5-Wave]")
    print(f"Status: {ewt1['status']}, Phase: {ewt1['cycle_phase']}, W3 Ext: {ewt1['fib_w3_ratio']}x")
    assert ewt1["status"] == "VALID_IMPULSE", f"Expected VALID_IMPULSE, got {ewt1['status']}"
    assert ewt1["cycle_phase"] == "COMPLETED_IMPULSE_AWAITING_ABC", f"Expected COMPLETED_IMPULSE_AWAITING_ABC, got {ewt1['cycle_phase']}"
    assert ewt1["cardinal_rules"]["rule1_w2"] is True, "Rule 1 should be True"
    assert ewt1["cardinal_rules"]["rule2_w3"] is True, "Rule 2 should be True"
    assert ewt1["cardinal_rules"]["rule3_w4"] is True, "Rule 3 should be True"
    assert len(ewt1["wave_points"]) == 6, f"Expected 6 wave points, got {len(ewt1['wave_points'])}"
    print("[PASS] Test 1: Valid Bullish 5-Wave Passed!")

    # 2. TEST CARDINAL RULE 1 VIOLATION (Wave 2 retraces beyond Wave 0 origin)
    # P2 = 4095.0 < P0 = 4100.0
    pivots_r1_violation = [
        (2, 4100.0, "low"),
        (5, 4120.0, "high"),
        (8, 4095.0, "low"),   # VIOLATION: breached below 4100
        (12, 4150.0, "high"),
        (16, 4130.0, "low"),
        (20, 4165.0, "high")
    ]
    df2, shs2, sls2 = build_mock_df_and_pivots(pivots_r1_violation, current_close=4158.0)
    ewt2 = detect_elliott_wave_cycle(df2, shs2, sls2)
    print("\n[Test 2: Rule 1 Violation (P2 < P0)]")
    print(f"Status: {ewt2['status']}, Phase: {ewt2['cycle_phase']}, Reason: {ewt2['rule_violation_reason']}")
    assert ewt2["status"] == "INVALID", f"Expected INVALID, got {ewt2['status']}"
    assert ewt2["cardinal_rules"]["rule1_w2"] is False, "Rule 1 should be False"
    assert ewt2["cycle_phase"] == "CARDINAL_RULE_VIOLATION"
    print("[PASS] Test 2: Rule 1 Violation Passed!")

    # 3. TEST CARDINAL RULE 2 VIOLATION (Wave 3 is the shortest wave)
    # W1 = 30 (4100 to 4130), P2 = 4120
    # W3 = 10 (4120 to 4130) -> SHORTEST!
    # P4 = 4135, W5 = 25 (4135 to 4160)
    pivots_r2_violation = [
        (2, 4100.0, "low"),
        (5, 4130.0, "high"),
        (8, 4120.0, "low"),
        (12, 4130.0, "high"),  # W3 = 10
        (16, 4135.0, "low"),
        (20, 4160.0, "high")   # W5 = 25, W1 = 30 -> W3 is shortest
    ]
    df3, shs3, sls3 = build_mock_df_and_pivots(pivots_r2_violation, current_close=4155.0)
    ewt3 = detect_elliott_wave_cycle(df3, shs3, sls3)
    print("\n[Test 3: Rule 2 Violation (Wave 3 shortest)]")
    print(f"Status: {ewt3['status']}, Phase: {ewt3['cycle_phase']}, Reason: {ewt3['rule_violation_reason']}")
    assert ewt3["status"] == "INVALID", f"Expected INVALID, got {ewt3['status']}"
    assert ewt3["cardinal_rules"]["rule2_w3"] is False, "Rule 2 should be False"
    assert ewt3["cycle_phase"] == "CARDINAL_RULE_VIOLATION"
    print("[PASS] Test 3: Rule 2 Violation Passed!")

    # 4. TEST CARDINAL RULE 3 VIOLATION (Wave 4 enters Wave 1 price territory)
    # P1 = 4120.0, P4 = 4115.0 (P4 <= P1 overlap!)
    pivots_r3_violation = [
        (2, 4100.0, "low"),
        (5, 4120.0, "high"),
        (8, 4110.0, "low"),
        (12, 4150.0, "high"),
        (16, 4115.0, "low"),   # VIOLATION: Wave 4 low 4115 <= Wave 1 high 4120
        (20, 4165.0, "high")
    ]
    df4, shs4, sls4 = build_mock_df_and_pivots(pivots_r3_violation, current_close=4158.0)
    ewt4 = detect_elliott_wave_cycle(df4, shs4, sls4)
    print("\n[Test 4: Rule 3 Violation (P4 <= P1 Overlap)]")
    print(f"Status: {ewt4['status']}, Phase: {ewt4['cycle_phase']}, Reason: {ewt4['rule_violation_reason']}")
    assert ewt4["status"] == "INVALID", f"Expected INVALID, got {ewt4['status']}"
    assert ewt4["cardinal_rules"]["rule3_w4"] is False, "Rule 3 should be False"
    assert ewt4["cycle_phase"] == "CARDINAL_RULE_VIOLATION"
    print("[PASS] Test 4: Rule 3 Violation Passed!")

    print("\n[SUCCESS] ALL 4 ELLIOTT WAVE UNIT TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_ewt_unit_tests()


def test_elliott_wave():
    run_ewt_unit_tests()
