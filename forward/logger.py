"""
Immutable Forward Signal & Event Logger for XAUUSD 15m ICT Strategy.
Complies with PREREG_TEST.md Section 5:
- Immutable append-only JSONL event stream with SHA256 hash chain.
- Event types: GENESIS, SIGNAL, FILL, EXPIRE, INVALIDATE, CLOSE, GAP.
- Pre-flight checks: verifies git working tree clean and frozen file hashes intact.
- Reuses exact backtester fill/exit logic (BID limit, trades through by 0.01 pt, 8-bar expiry, SL-first tie).
- External head-hash anchoring hook (Telegram).
- Detects missing candles and logs GAP events.
- Closed candles only; converts all feed timestamps to UTC and America/New_York.
"""

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import requests

try:
    import zoneinfo
    NY_TZ = zoneinfo.ZoneInfo("America/New_York")
except Exception:
    try:
        from backports import zoneinfo
        NY_TZ = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        import datetime as dt
        NY_TZ = dt.timezone.utc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core import config
from core.signals import evaluate_causal_signal_at_bar, is_in_killzone
from core.backtester import check_position_exit, check_pending_order, check_same_candle_sl

# FROZEN PARAMETERS (PREREG SECTION 5)
FORWARD_START_DATE = "2026-11-12T00:00:00Z"
PARAMS_HASH = "cf8e2a9c695a3496"
P_SCALED = 0.6077  # Stop floor as percentage of close price
RR_TARGET = 2.0
SPREAD_ASSUMED = 0.35
SLIPPAGE_ASSUMED = 0.10
COMMISSION_PER_LOT = 7.0
ORDER_EXPIRY_BARS = 8
MIN_BUFFER_WARMUP = 135
GENESIS_PREV_HASH = "0" * 64

# FROZEN FILE SHA256 HASHES FOR PRE-FLIGHT INTEGRITY
FROZEN_FILE_HASHES = {
    "core/signals.py": "786c1c3b820831706db091ea7e904765b4614c0e3c356893078c2c9eb09e2d28",
    "core/config.py": "af1afb8cdf46d8120a81000635193fe43e85ced5628b0d9296bc5ccd56af6576",
    "core/backtester.py": "398d2f251f5f8a96ddfae89096e9120128797243e655739f8fa92540d0ff615d",
    "research/PREREG_TEST.md": "6aca6d3c8f3d00a973daab9d55cd2ce42dcc142be587890ce11fd003ce2b801c",
    "research/meta_labeling/pipeline_metalabeling.py": "f7e7bcee6e108c93d4e182206bd3f4ba0f2da8ab68e9531d11bce487a0d705c4"
}


def get_file_sha256(filepath):
    """Calculates SHA256 of a file."""
    abs_path = os.path.join(REPO_ROOT, filepath) if not os.path.isabs(filepath) else filepath
    if not os.path.exists(abs_path):
        return None
    h = hashlib.sha256()
    with open(abs_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit():
    """Returns current git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        return res.stdout.strip() if res.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def get_git_tag():
    """Auto-reads the current or closest git tag from the repository."""
    try:
        res = subprocess.run(["git", "describe", "--tags", "--exact-match"], capture_output=True, text=True, cwd=REPO_ROOT)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
        res = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, cwd=REPO_ROOT)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


FORWARD_GIT_TAG = get_git_tag()


def is_git_clean(allow_untracked=True):
    """Checks whether git working tree is clean."""
    try:
        cmd = ["git", "status", "--porcelain"]
        if allow_untracked:
            cmd.append("-uno")
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        return len(res.stdout.strip()) == 0
    except Exception:
        return False


def canonical_json(data_dict):
    """Serializes dictionary to deterministic canonical JSON."""
    return json.dumps(data_dict, sort_keys=True, separators=(",", ":"))


def compute_record_hash(prev_hash, event_type, timestamp_utc, data_dict):
    """Computes SHA256 of (prev_hash | event | timestamp | canonical_data)."""
    payload = f"{prev_hash}|{event_type}|{timestamp_utc}|{canonical_json(data_dict)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_chain(log_path):
    """
    Verifies full hash chain integrity of an append-only JSONL log.
    Returns: (is_valid: bool, msg: str, record_count: int)
    """
    if not os.path.exists(log_path):
        return False, f"File does not exist: {log_path}", 0

    count = 0
    expected_prev = GENESIS_PREV_HASH

    with open(log_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line_str = raw_line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except Exception as e:
                return False, f"JSON parse error at line {line_no}: {e}", count

            rec_idx = record.get("index")
            if rec_idx != count:
                return False, f"Index sequence mismatch at line {line_no}: expected {count}, got {rec_idx}", count

            prev_hash = record.get("prev_hash")
            if prev_hash != expected_prev:
                return False, f"Prev hash mismatch at line {line_no}: expected {expected_prev}, got {prev_hash}", count

            event_type = record.get("event")
            ts_utc = record.get("timestamp_utc")
            data = record.get("data", {})
            claimed_hash = record.get("record_hash")

            recalc_hash = compute_record_hash(prev_hash, event_type, ts_utc, data)
            if recalc_hash != claimed_hash:
                return False, f"Hash tamper detected at line {line_no}: claimed {claimed_hash}, recomputed {recalc_hash}", count

            expected_prev = claimed_hash
            count += 1

    if count == 0:
        return False, "Log file is empty", 0

    return True, "CHAIN_VALID", count


def verify_against_anchor(log_path, anchored_hashes):
    """
    Verifies log file integrity against externally anchored hashes.
    anchored_hashes can be:
      - Path to external anchor JSON ledger file (str) (e.g., pulled from git remote)
      - List of anchor dicts [{'index': 0, 'record_hash': '...'}, ...]
      - List of tuples [(index, hash), ...] or [(index, event, hash), ...]
      - List of raw hash strings [hash0, hash1, ...]

    Reads anchors from the specified independent external source and verifies
    that every anchored checkpoint matches the append-only log exactly.
    Detects:
    1. Internal hash chain tamper or corruption.
    2. Log truncation (e.g. attacker deleted recent records to hide losses).
    3. Log forking/divergence from externally anchored checkpoints.

    Returns: (is_valid: bool, msg: str, record_count: int)
    """
    chain_valid, chain_msg, count = verify_chain(log_path)
    if not chain_valid:
        return False, f"CHAIN_INVALID: {chain_msg}", count

    if isinstance(anchored_hashes, str):
        anchored_hashes = load_external_anchors(anchored_hashes)

    if not anchored_hashes:
        return True, "CHAIN_VALID_NO_ANCHORS", count

    # Parse anchor dictionary: index -> expected_hash
    anchor_dict = {}
    for item in anchored_hashes:
        if isinstance(item, dict):
            idx = item.get("index")
            h = item.get("record_hash")
            if idx is not None and h:
                anchor_dict[int(idx)] = h
        elif isinstance(item, (tuple, list)):
            idx = item[0]
            h = item[-1]
            anchor_dict[int(idx)] = h
        else:
            anchor_dict[len(anchor_dict)] = item

    max_anchor_idx = max(anchor_dict.keys())
    if count <= max_anchor_idx:
        return False, f"TRUNCATION_DETECTED: Log has {count} records, but external anchor requires index {max_anchor_idx}", count

    # Read log lines and compare hashes
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for a_idx, expected_h in anchor_dict.items():
        if a_idx >= len(lines):
            return False, f"TRUNCATION_DETECTED: Missing anchored record at index {a_idx}", count
        rec = json.loads(lines[a_idx])
        actual_h = rec.get("record_hash")
        if actual_h != expected_h:
            return False, f"ANCHOR_MISMATCH at index {a_idx}: log has {actual_h}, external anchor has {expected_h}", count

    return True, "ANCHOR_AND_CHAIN_VALID", count


def convert_mt5_time_to_utc(server_time, server_tz="US_EET"):
    """
    Converts MT5 broker server time to UTC datetime.
    Retail Forex/Gold brokers (MetaQuotes Demo, IC Markets, Pepperstone, etc.)
    operate on Eastern European Time (UTC+2 in winter, UTC+3 in summer) that shifts
    on US Daylight Saving Time rules (2nd Sunday in March to 1st Sunday in November)
    so that 00:00:00 server time always equals 17:00:00 America/New_York close.

    Formula for US_EET:
        Server_Time = America/New_York + 7 hours (exact all year round)
        America/New_York = Server_Time - 7 hours
        UTC = (Server_Time - 7 hours) localized to America/New_York, converted to UTC.

    Also supports:
        - 'EU_EET' / 'Europe/Athens' (European Union directive DST transitions)
        - 'UTC' / 'GMT' (Direct UTC timestamp)
    """
    ts = pd.Timestamp(server_time)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC")

    tz_mode = server_tz.upper() if isinstance(server_tz, str) else "US_EET"

    if tz_mode in ("UTC", "GMT"):
        return ts.tz_localize("UTC")
    elif tz_mode in ("EU_EET", "EUROPE/ATHENS", "EUROPE/HELSINKI"):
        return ts.tz_localize("Europe/Athens").tz_convert("UTC")
    else:
        # Standard US-DST broker rule: Server = NY + 7h => NY = Server - 7h
        ts_ny = ts - timedelta(hours=7)
        ts_ny_aware = ts_ny.tz_localize(NY_TZ, nonexistent="shift_forward")
        return ts_ny_aware.tz_convert("UTC")


def check_git_remote_reachable(remote="origin", timeout=10):
    """
    Verifies that the external git remote repository is reachable.
    Returns: (reachable: bool, msg: str)
    """
    try:
        res = subprocess.run(["git", "ls-remote", "--heads", remote], capture_output=True, text=True, cwd=REPO_ROOT, timeout=timeout)
        if res.returncode == 0:
            return True, "REMOTE_REACHABLE"
        return False, res.stderr.strip() or f"exit code {res.returncode}"
    except Exception as e:
        return False, str(e)


def push_external_anchor_to_git(anchor_path, remote="origin", branch="forward-anchors"):
    """
    Commits the external anchor ledger directly to a dedicated git branch
    and pushes to git remote without altering the active working tree branch.

    BRANCH PROTECTION CONTRACT:
    Force-push ('git push --force') must be strictly disabled on the remote repository
    and target branch to ensure forward anchor immutability and prevent history tampering.

    Returns: (success: bool, msg: str)
    """
    try:
        abs_path = os.path.abspath(anchor_path)
        if not os.path.exists(abs_path):
            return False, f"Anchor file does not exist: {abs_path}"
        rel_path = os.path.relpath(abs_path, REPO_ROOT)

        # 1. Create git blob object directly from file
        blob_hash = subprocess.check_output(["git", "hash-object", "-w", rel_path], cwd=REPO_ROOT, text=True).strip()

        # 2. Build tree object containing the anchor ledger (bytes input prevents Windows CRLF translation)
        file_name = os.path.basename(rel_path)
        tree_input_bytes = f"100644 blob {blob_hash}\t{file_name}\n".encode("utf-8")
        tree_proc = subprocess.run(["git", "mktree"], input=tree_input_bytes, capture_output=True, cwd=REPO_ROOT)
        if tree_proc.returncode != 0:
            err_raw = tree_proc.stderr
            err_msg = err_raw.decode("utf-8", errors="replace").strip() if isinstance(err_raw, bytes) else str(err_raw).strip()
            return False, f"git mktree failed: {err_msg}"
        out_raw = tree_proc.stdout
        tree_hash = out_raw.decode("utf-8", errors="replace").strip() if isinstance(out_raw, bytes) else str(out_raw).strip()

        # 3. Locate parent commit on target branch (local ref or remote tracking ref)
        target_branch = branch or "forward-anchors"
        commit_msg = f"chore(anchor): forward anchor checkpoint {datetime.now(timezone.utc).isoformat()}"
        commit_args = ["git", "commit-tree", tree_hash, "-m", commit_msg]

        parent_res = subprocess.run(["git", "rev-parse", f"refs/heads/{target_branch}"], capture_output=True, text=True, cwd=REPO_ROOT)
        if parent_res.returncode == 0:
            commit_args.extend(["-p", parent_res.stdout.strip()])
        else:
            parent_res_remote = subprocess.run(["git", "rev-parse", f"refs/remotes/{remote}/{target_branch}"], capture_output=True, text=True, cwd=REPO_ROOT)
            if parent_res_remote.returncode == 0:
                commit_args.extend(["-p", parent_res_remote.stdout.strip()])

        commit_proc = subprocess.run(commit_args, capture_output=True, text=True, cwd=REPO_ROOT)
        if commit_proc.returncode != 0:
            return False, f"git commit-tree failed: {commit_proc.stderr.strip()}"
        commit_hash = commit_proc.stdout.strip()

        # 4. Update local ref for target branch
        update_res = subprocess.run(["git", "update-ref", f"refs/heads/{target_branch}", commit_hash], cwd=REPO_ROOT, check=True)

        # 5. Push to remote
        cmd = ["git", "push", remote, f"refs/heads/{target_branch}:refs/heads/{target_branch}"]
        push_res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=15)
        if push_res.returncode != 0:
            return False, f"git push failed: {push_res.stderr.strip()}"
        return True, "GIT_PUSH_OK"
    except Exception as e:
        return False, str(e)


def save_external_anchor(anchor_path, record_index, event_type, record_hash, git_push=False, git_remote="origin", git_branch="forward-anchors"):
    """
    Saves an anchor record to an independent external JSON ledger.
    If git_push=True, automatically commits and pushes the ledger to git remote.
    Returns: (success: bool, msg: str)
    """
    abs_path = os.path.abspath(anchor_path) if not os.path.isabs(anchor_path) else anchor_path
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    anchors = []
    if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                anchors = json.load(f)
        except Exception:
            anchors = []

    anchors.append({
        "index": record_index,
        "event": event_type,
        "record_hash": record_hash,
        "anchored_at_utc": datetime.now(timezone.utc).isoformat()
    })

    tmp_path = abs_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(anchors, f, indent=2)
        try:
            os.replace(tmp_path, abs_path)
        except OSError:
            with open(abs_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(anchors, f, indent=2)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception:
        pass

    if git_push:
        return push_external_anchor_to_git(abs_path, remote=git_remote, branch=git_branch)
    return True, "LOCAL_ANCHOR_SAVED"


def load_external_anchors(anchor_path):
    """Loads list of anchored records from an external anchor ledger."""
    abs_path = os.path.abspath(anchor_path) if not os.path.isabs(anchor_path) else anchor_path
    if not os.path.exists(abs_path):
        return []
    with open(abs_path, "r", encoding="utf-8") as f:
        return json.load(f)


def anchor_head_hash_to_telegram(record_index, event_type, record_hash, token=None, chat_id=None):
    """Anchors record hash to external Telegram channel/chat."""
    tok = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    cid = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not cid:
        return False, "TELEGRAM_CREDENTIALS_MISSING"

    text = (
        f"🔗 [FORWARD ANCHOR]\n"
        f"Index: #{record_index}\n"
        f"Event: {event_type}\n"
        f"Head Hash: {record_hash}"
    )
    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": cid, "text": text}, timeout=5)
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)


class ForwardSignalLogger:
    """
    Stateful Forward Paper-Trading Logger.
    Processes live closed 15m candles, detects ICT setups, tracks pending orders
    and active positions using backtester logic, and writes an immutable hash chain.
    """

    def __init__(
        self,
        forward_start_date=None,
        log_path="forward/signals.jsonl",
        check_git=True,
        check_frozen_hashes=True,
        anchor_callback=None,
        external_anchor_path=None,
        telegram_token=None,
        telegram_chat_id=None,
        killzone_gate=True,
        p_scaled_filter=True,
        killzone_expiry=False,
        skip_outside_kz_eval=False,
        min_buffer_warmup=MIN_BUFFER_WARMUP,
        buffer_size=300,
        server_tz="EU_EET",
        server_name="MetaQuotes-Demo",
        symbol="XAUUSD",
        feed_name="MT5",
        git_push_anchors=False,
        git_remote="origin",
        git_branch=None,
        enforce_staleness=None,
        track_spread=True,
        point_size=0.01,
        rehearsal=False
    ):
        self.rehearsal = rehearsal
        if self.rehearsal:
            if git_branch is None or git_branch == "forward-anchors-rehearsal":
                git_branch = "forward-anchors-rehearsal"
            elif git_branch == "forward-anchors":
                raise ValueError("REHEARSAL ERROR: Rehearsal mode strictly prohibits pushing to production 'forward-anchors' branch!")
            prod_log = os.path.abspath(os.path.join(REPO_ROOT, "forward", "signals.jsonl"))
            if os.path.abspath(log_path) == prod_log:
                raise ValueError("REHEARSAL ERROR: Rehearsal mode strictly prohibits writing to production 'forward/signals.jsonl'!")
        elif git_branch is None:
            git_branch = "forward-anchors"

        self.log_path = os.path.abspath(log_path) if not os.path.isabs(log_path) else log_path
        self.check_git = check_git
        self.check_frozen_hashes = check_frozen_hashes
        self.anchor_callback = anchor_callback
        self.external_anchor_path = os.path.abspath(external_anchor_path) if external_anchor_path else None
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.killzone_gate = killzone_gate
        self.p_scaled_filter = p_scaled_filter
        self.killzone_expiry = killzone_expiry
        self.skip_outside_kz_eval = skip_outside_kz_eval
        self.min_buffer_warmup = min_buffer_warmup
        self.buffer_size = max(buffer_size, min_buffer_warmup)
        self.server_tz = server_tz
        self.server_name = server_name
        self.symbol = symbol
        self.feed_name = feed_name
        self.git_push_anchors = git_push_anchors
        self.git_remote = git_remote
        self.git_branch = git_branch
        self.enforce_staleness = check_git if enforce_staleness is None else enforce_staleness
        self.track_spread = track_spread
        self.point_size = point_size

        if self.track_spread:
            from forward.execution_tracker import ExecutionTracker
            self.execution_tracker = ExecutionTracker(
                assumed_spread_pts=SPREAD_ASSUMED,
                point_size=point_size,
                symbol=self.symbol
            )
        else:
            self.execution_tracker = None

        # Forward Start Date staleness validation (< 1 day in the past)
        if forward_start_date is not None:
            try:
                start_dt = pd.Timestamp(forward_start_date)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.tz_localize("UTC")
                else:
                    start_dt = start_dt.tz_convert("UTC")
            except Exception as e:
                raise ValueError(f"Invalid forward_start_date format '{forward_start_date}': {e}")

            if self.enforce_staleness:
                now_utc = datetime.now(timezone.utc)
                staleness = (now_utc - start_dt).total_seconds()
                if staleness > 86400:  # > 1 day in the past
                    raise ValueError(
                        f"STARTUP ERROR: forward_start_date '{forward_start_date}' is >1 day in the past "
                        f"({staleness/3600:.1f} hours ago). Refusing to start forward paper trading with stale start date."
                    )
            self.forward_start_date = start_dt.isoformat()
        else:
            self.forward_start_date = None

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Internal state
        self.pending_order = None
        self.active_position = None
        self.last_candle_time_utc = None
        self.signal_counter = 0

        # Persisted anchor retry queue (survives crashes & restarts)
        self.queue_path = os.path.splitext(self.log_path)[0] + "_anchor_queue.json"
        self.anchor_queue = self._load_anchor_queue()

        # Buffer persistence companion file
        self.buffer_path = os.path.splitext(self.log_path)[0] + "_buffer.json"

        # Fast candle buffer
        self._times_ny = []
        self._opens = []
        self._highs = []
        self._lows = []
        self._closes = []
        self._volumes = []

        # Pre-flight integrity validation
        self._run_preflight_checks()

        # Initialize or resume chain
        self._initialize_or_resume_chain()

    def _load_anchor_queue(self):
        """Loads pending un-anchored records from disk."""
        if os.path.exists(self.queue_path) and os.path.getsize(self.queue_path) > 0:
            try:
                with open(self.queue_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [tuple(x) for x in data]
            except Exception:
                return []
        return []

    def _save_anchor_queue(self):
        """Persists pending anchor retry queue to disk atomically."""
        if self.anchor_queue:
            tmp_path = self.queue_path + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(self.anchor_queue, f)
                try:
                    os.replace(tmp_path, self.queue_path)
                except OSError:
                    with open(self.queue_path, "w", encoding="utf-8", newline="\n") as f:
                        json.dump(self.anchor_queue, f)
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
            except Exception:
                pass
        elif os.path.exists(self.queue_path):
            try:
                os.remove(self.queue_path)
            except Exception:
                pass

    def _save_buffer_snapshot(self):
        """Saves rolling candle buffer snapshot for crash recovery."""
        if not self._times_ny:
            return
        buf_data = []
        for i in range(len(self._times_ny)):
            buf_data.append([
                self._times_ny[i].isoformat(),
                self._opens[i],
                self._highs[i],
                self._lows[i],
                self._closes[i],
                self._volumes[i]
            ])
        tmp = self.buffer_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(buf_data, f)
            try:
                os.replace(tmp, self.buffer_path)
            except OSError:
                with open(self.buffer_path, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(buf_data, f)
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
        except Exception:
            pass

    def _load_buffer_snapshot(self):
        """Loads rolling candle buffer snapshot from companion file."""
        if os.path.exists(self.buffer_path) and os.path.getsize(self.buffer_path) > 0:
            try:
                with open(self.buffer_path, "r", encoding="utf-8") as f:
                    buf_data = json.load(f)
                self._times_ny = [pd.Timestamp(x[0]) for x in buf_data]
                self._opens = [float(x[1]) for x in buf_data]
                self._highs = [float(x[2]) for x in buf_data]
                self._lows = [float(x[3]) for x in buf_data]
                self._closes = [float(x[4]) for x in buf_data]
                self._volumes = [float(x[5]) for x in buf_data]
                if self._times_ny:
                    last_buf_time = self._times_ny[-1].tz_convert("UTC")
                    if self.last_candle_time_utc is None or last_buf_time > self.last_candle_time_utc:
                        self.last_candle_time_utc = last_buf_time
            except Exception:
                pass

    def seed_buffer(self, candles):
        """
        Seeds rolling candle buffer with historical closed bars without emitting signals
        or appending events. Used on startup/restart to pre-warm the 135-bar window.
        """
        if isinstance(candles, pd.DataFrame):
            times = candles.index
            opens = candles["open"].values
            highs = candles["high"].values
            lows = candles["low"].values
            closes = candles["close"].values
            vols = candles["volume"].values if "volume" in candles.columns else [0.0] * len(candles)
            n = len(candles)
            for i in range(n):
                raw_t = times[i]
                if isinstance(raw_t, (int, float, np.integer, np.floating)):
                    ts = pd.to_datetime(raw_t, unit="s")
                else:
                    ts = pd.Timestamp(raw_t)
                ts_ny = ts.tz_convert(NY_TZ) if ts.tzinfo is not None else ts.tz_localize(NY_TZ)
                self._times_ny.append(ts_ny)
                self._opens.append(float(opens[i]))
                self._highs.append(float(highs[i]))
                self._lows.append(float(lows[i]))
                self._closes.append(float(closes[i]))
                self._volumes.append(float(vols[i]))
        elif isinstance(candles, (list, tuple)):
            for c in candles:
                raw_t = c.get("time", c.get("Date_NY", c.get("timestamp")))
                if isinstance(raw_t, (int, float, np.integer, np.floating)):
                    ts = pd.to_datetime(raw_t, unit="s")
                else:
                    ts = pd.Timestamp(raw_t)
                if ts.tzinfo is None:
                    ts_utc = convert_mt5_time_to_utc(ts, server_tz=self.server_tz)
                else:
                    ts_utc = ts.tz_convert("UTC")
                ts_ny = ts_utc.tz_convert(NY_TZ)
                self._times_ny.append(ts_ny)
                self._opens.append(float(c["open"]))
                self._highs.append(float(c["high"]))
                self._lows.append(float(c["low"]))
                self._closes.append(float(c["close"]))
                self._volumes.append(float(c.get("volume", 0.0)))
        while len(self._times_ny) > self.buffer_size:
            self._times_ny.pop(0)
            self._opens.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)
            self._volumes.pop(0)
        self._save_buffer_snapshot()

    def _run_preflight_checks(self):
        """Enforces pre-flight checks: clean git tree, matching frozen file hashes, remote reachability."""
        if self.check_frozen_hashes:
            for rel_path, expected_hash in FROZEN_FILE_HASHES.items():
                actual_hash = get_file_sha256(rel_path)
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"PRE-FLIGHT ERROR: Frozen file hash mismatch for '{rel_path}'. "
                        f"Expected {expected_hash}, got {actual_hash}. Strategy logic is frozen."
                    )

        if self.check_git:
            if not is_git_clean(allow_untracked=True):
                raise RuntimeError(
                    "PRE-FLIGHT ERROR: Git working tree is dirty. "
                    "Forward logger refuses to start with uncommitted tracked modifications."
                )

        if self.git_push_anchors:
            reachable, err = check_git_remote_reachable(self.git_remote)
            if not reachable:
                raise RuntimeError(
                    f"PRE-FLIGHT ERROR: External git remote '{self.git_remote}' is unreachable or network is down ({err}). "
                    "Forward execution requires verified external anchor push capability."
                )

    def _initialize_or_resume_chain(self):
        """Verifies existing log chain or writes the GENESIS record if starting anew."""
        if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
            valid, msg, count = verify_chain(self.log_path)
            if not valid:
                raise RuntimeError(f"FORWARD LOG INTEGRITY VIOLATION: {msg}")
            self.current_index = count
            self.head_hash = self._read_last_hash()
            with open(self.log_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.strip():
                    rec0 = json.loads(first_line)
                    gen_start = rec0.get("data", {}).get("forward_start_date")
                    if self.forward_start_date is None:
                        self.forward_start_date = gen_start
                    elif gen_start and self.forward_start_date != gen_start:
                        raise RuntimeError(
                            f"STARTUP ERROR: forward_start_date mismatch with existing log: "
                            f"log has '{gen_start}', startup passed '{self.forward_start_date}'."
                        )
            self._replay_state_from_log()
            self._load_buffer_snapshot()

            # Recalculate pending order bars_elapsed from loaded buffer
            if self.pending_order is not None and self._times_ny:
                sig_t = pd.Timestamp(self.pending_order["sig_time_utc"])
                if sig_t.tzinfo is None:
                    sig_t = sig_t.tz_localize("UTC")
                bars_elapsed = sum(1 for t_ny in self._times_ny if t_ny.tz_convert("UTC") > sig_t)
                self.pending_order["bars_elapsed"] = bars_elapsed

            # Crash-recovery: check for crash after append before anchor push (A4)
            if self.external_anchor_path and os.path.exists(self.external_anchor_path):
                anchors = load_external_anchors(self.external_anchor_path)
                anchored_indices = {a["index"] for a in anchors if isinstance(a, dict) and "index" in a}
                last_idx = self.current_index - 1
                if last_idx >= 0 and last_idx not in anchored_indices:
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        last_rec = json.loads(lines[-1])
                        anchor_item = (last_rec["index"], last_rec["event"], last_rec["record_hash"])
                        if anchor_item not in self.anchor_queue:
                            self.anchor_queue.append(anchor_item)
                            self._save_anchor_queue()
        else:
            if not self.forward_start_date:
                raise ValueError(
                    "STARTUP ERROR: forward_start_date is a required startup argument when initializing "
                    "a new forward paper-trading run. It must not be omitted or None."
                )
            self.current_index = 0
            self.head_hash = GENESIS_PREV_HASH
            self._write_genesis_record()

    def _read_last_hash(self):
        """Reads the record_hash of the final line in the log."""
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if not lines:
                return GENESIS_PREV_HASH
            last_record = json.loads(lines[-1])
            return last_record["record_hash"]

    def _replay_state_from_log(self):
        """Reconstructs pending order, active position and counters from the log."""
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                evt = rec["event"]
                d = rec.get("data", {})
                if evt == "SIGNAL":
                    self.signal_counter = max(self.signal_counter, d.get("signal_num", 0))
                    if d.get("passed_primary", False):
                        self.pending_order = {
                            "signal_id": d["signal_id"],
                            "direction": d["direction"],
                            "entry": d["entry"],
                            "sl": d["sl"],
                            "tp": d["tp"],
                            "risk_dist": d["risk_dist"],
                            "killzone": d.get("killzone"),
                            "sig_time_utc": rec["timestamp_utc"],
                            "bars_elapsed": 0
                        }
                elif evt in ("INVALIDATE", "EXPIRE"):
                    self.pending_order = None
                elif evt == "FILL":
                    self.active_position = {
                        "signal_id": d["signal_id"],
                        "direction": d["direction"],
                        "entry_price": d["fill_price"],
                        "sl": d["sl"],
                        "tp": d["tp"],
                        "fill_time_utc": rec["timestamp_utc"],
                        "planned_risk_dist": d["planned_risk_dist"]
                    }
                    self.pending_order = None
                elif evt == "CLOSE":
                    self.active_position = None
                    self.pending_order = None

    def _flush_anchor_queue(self):
        """Retries sending queued anchors that previously failed. Never drops an anchor."""
        if not self.anchor_queue:
            return
        remaining = []
        for item in self.anchor_queue:
            idx, evt, h = item
            success = False
            if self.anchor_callback:
                try:
                    self.anchor_callback(idx, evt, h)
                    success = True
                except Exception:
                    success = False
            elif self.telegram_token and self.telegram_chat_id:
                ok, _ = anchor_head_hash_to_telegram(
                    idx, evt, h,
                    token=self.telegram_token, chat_id=self.telegram_chat_id
                )
                success = ok
            if not success:
                remaining.append(item)
        self.anchor_queue = remaining
        self._save_anchor_queue()

    def _append_event(self, event_type, timestamp_utc, data_dict):
        """Appends a new event record to the hash chain and anchors head hash."""
        # Attempt to flush any previously failed anchors first
        self._flush_anchor_queue()

        rec_hash = compute_record_hash(self.head_hash, event_type, timestamp_utc, data_dict)
        record = {
            "index": self.current_index,
            "prev_hash": self.head_hash,
            "event": event_type,
            "timestamp_utc": timestamp_utc,
            "data": data_dict,
            "record_hash": rec_hash
        }

        with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        # Update chain pointers
        self.head_hash = rec_hash
        self.current_index += 1

        # External anchor: on failure, enqueue for retry and persist to disk (never skip)
        anchor_item = (record["index"], event_type, rec_hash)
        anchor_ok = False
        if self.anchor_callback:
            try:
                self.anchor_callback(record["index"], event_type, rec_hash)
                anchor_ok = True
            except Exception:
                anchor_ok = False
        elif self.telegram_token and self.telegram_chat_id:
            ok, _ = anchor_head_hash_to_telegram(
                record["index"], event_type, rec_hash,
                token=self.telegram_token, chat_id=self.telegram_chat_id
            )
            anchor_ok = ok
        else:
            anchor_ok = True

        if not anchor_ok:
            self.anchor_queue.append(anchor_item)
            self._save_anchor_queue()

        if self.external_anchor_path:
            ok, msg = save_external_anchor(
                self.external_anchor_path,
                record["index"],
                event_type,
                rec_hash,
                git_push=self.git_push_anchors,
                git_remote=self.git_remote,
                git_branch=self.git_branch
            )
            if not ok:
                self.anchor_queue.append(anchor_item)
                self._save_anchor_queue()

        return record

    def _write_genesis_record(self):
        """Writes the initial GENESIS record (Index 0)."""
        now_utc = self.forward_start_date if self.forward_start_date else datetime.now(timezone.utc).isoformat()
        
        # Calendar-month computation: exact 30 calendar months from forward_start_date
        forward_end_date_30m = None
        if self.forward_start_date:
            try:
                start_ts = pd.Timestamp(self.forward_start_date)
                end_ts_30m = start_ts + pd.DateOffset(months=30)
                forward_end_date_30m = end_ts_30m.isoformat()
            except Exception:
                forward_end_date_30m = None

        genesis_data = {
            "genesis_version": "v1.0",
            "git_tag": get_git_tag(),
            "git_commit": get_git_commit(),
            "forward_start_date": self.forward_start_date,
            "forward_end_date_30m": forward_end_date_30m,
            "symbol": self.symbol,
            "server_name": self.server_name,
            "feed_name": self.feed_name,
            "server_tz": self.server_tz,
            "params_hash": PARAMS_HASH,
            "p_scaled": P_SCALED,
            "rr_target": RR_TARGET,
            "spread_assumed": SPREAD_ASSUMED,
            "slippage_assumed": SLIPPAGE_ASSUMED,
            "commission_per_lot": COMMISSION_PER_LOT,
            "frozen_file_hashes": FROZEN_FILE_HASHES
        }
        self._append_event("GENESIS", now_utc, genesis_data)

    def on_candle_closed(self, candle_dict, realized_spread=None):
        """
        Ingests a single closed 15m candle.
        Expected keys in candle_dict:
          - time / Date_NY / timestamp (datetime, Timestamp, or str)
          - open, high, low, close (float)
          - volume (optional int/float)
        """
        raw_time = candle_dict.get("time", candle_dict.get("Date_NY", candle_dict.get("timestamp")))
        if isinstance(raw_time, (int, float, np.integer, np.floating)):
            ts = pd.to_datetime(raw_time, unit="s")
        else:
            ts = pd.Timestamp(raw_time)

        # Flush any pending external anchors on each candle close
        self._flush_anchor_queue()

        if ts.tzinfo is None:
            ts_utc = convert_mt5_time_to_utc(ts, server_tz=self.server_tz)
        else:
            ts_utc = ts.tz_convert("UTC")

        ts_ny = ts_utc.tz_convert(NY_TZ)

        o = float(candle_dict["open"])
        h = float(candle_dict["high"])
        l = float(candle_dict["low"])
        c = float(candle_dict["close"])
        v = float(candle_dict.get("volume", 0.0))

        tick_bid = candle_dict.get("tick_bid", candle_dict.get("bid"))
        tick_ask = candle_dict.get("tick_ask", candle_dict.get("ask"))
        bar_spread_raw = candle_dict.get("spread")

        if realized_spread is not None:
            cur_spread = float(realized_spread)
        elif bar_spread_raw is not None:
            raw_sp = float(bar_spread_raw)
            cur_spread = raw_sp * self.point_size if raw_sp > 2.0 else raw_sp
        else:
            cur_spread = SPREAD_ASSUMED

        # Duplicate candle deduplication
        if self.last_candle_time_utc is not None and ts_utc <= self.last_candle_time_utc:
            return

        # A3: History Preload Guard - candles before forward_start_date only seed buffer
        if self.forward_start_date is not None:
            start_dt_utc = pd.Timestamp(self.forward_start_date)
            if start_dt_utc.tzinfo is None:
                start_dt_utc = start_dt_utc.tz_localize("UTC")
            else:
                start_dt_utc = start_dt_utc.tz_convert("UTC")
            if ts_utc < start_dt_utc:
                self.last_candle_time_utc = ts_utc
                self._times_ny.append(ts_ny)
                self._opens.append(o)
                self._highs.append(h)
                self._lows.append(l)
                self._closes.append(c)
                self._volumes.append(v)
                if len(self._times_ny) > self.buffer_size:
                    self._times_ny.pop(0)
                    self._opens.pop(0)
                    self._highs.pop(0)
                    self._lows.pop(0)
                    self._closes.pop(0)
                    self._volumes.pop(0)
                self._save_buffer_snapshot()
                return None

        # 1. GAP Detection
        if self.last_candle_time_utc is not None:
            diff = ts_utc - self.last_candle_time_utc
            if diff > timedelta(minutes=15):
                prev_ny = self.last_candle_time_utc.tz_convert(NY_TZ)
                missing_bars = int(diff.total_seconds() // 900) - 1

                # Weekend close: Friday >= 16:45 NY to Sunday >= 17:00 NY
                is_weekend = (prev_ny.weekday() == 4 and (prev_ny.hour > 16 or (prev_ny.hour == 16 and prev_ny.minute >= 45))) and \
                             (ts_ny.weekday() == 6 and ts_ny.hour >= 17)
                # Daily break: Monday-Thursday 16:45-17:15 NY to 18:00 NY
                is_daily_break = (prev_ny.weekday() in (0, 1, 2, 3) and prev_ny.hour >= 16 and ts_ny.hour >= 18 and diff <= timedelta(hours=2))

                if is_weekend:
                    gap_type = "WEEKEND_MARKET_CLOSE"
                elif is_daily_break:
                    gap_type = "DAILY_MARKET_BREAK"
                else:
                    gap_type = "DOWNTIME_GAP"

                self._append_event("GAP", ts_utc.isoformat(), {
                    "gap_type": gap_type,
                    "gap_start_utc": self.last_candle_time_utc.isoformat(),
                    "gap_end_utc": ts_utc.isoformat(),
                    "elapsed_minutes": int(diff.total_seconds() // 60),
                    "missing_bars": missing_bars
                })

        self.last_candle_time_utc = ts_utc

        # Append to fast candle buffer
        self._times_ny.append(ts_ny)
        self._opens.append(o)
        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)

        if len(self._times_ny) > self.buffer_size:
            self._times_ny.pop(0)
            self._opens.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)
            self._volumes.pop(0)

        self._save_buffer_snapshot()

        # 2. Evaluate Active Position (Exit check using imported backtester logic)
        if self.active_position is not None:
            p = self.active_position
            exit_trig, exit_p, exit_reason = check_position_exit(
                direction=p["direction"],
                sl_level=p["sl"],
                tp_level=p["tp"],
                cur_hi=h,
                cur_lo=l,
                spread=SPREAD_ASSUMED,
                slippage=SLIPPAGE_ASSUMED
            )
            if exit_trig:
                if p["direction"] == "BUY":
                    pnl_pts = exit_p - p["entry_price"]
                else:
                    pnl_pts = p["entry_price"] - exit_p

                comm_r = COMMISSION_PER_LOT / (config.CONTRACT_SIZE * p["planned_risk_dist"]) if p["planned_risk_dist"] > 0 else 0.0
                gross_r = pnl_pts / p["planned_risk_dist"] if p["planned_risk_dist"] > 0 else 0.0
                net_r = round(gross_r - comm_r, 3)

                self._append_event("CLOSE", ts_utc.isoformat(), {
                    "signal_id": p["signal_id"],
                    "direction": p["direction"],
                    "entry_price": p["entry_price"],
                    "exit_price": exit_p,
                    "exit_reason": exit_reason,
                    "net_r": net_r,
                    "planned_risk_dist": p["planned_risk_dist"],
                    "fill_time_utc": p["fill_time_utc"],
                    "exit_time_utc": ts_utc.isoformat(),
                    "realized_spread_at_exit": cur_spread
                })
                self.active_position = None

        # 3. Evaluate Pending Order (Fill/Invalidation/Expiry using imported backtester logic)
        elif self.pending_order is not None:
            p = self.pending_order
            p["bars_elapsed"] += 1

            kz_open = True
            if self.killzone_expiry and p.get("killzone"):
                kz_open = (is_in_killzone(ts_ny.time()) == p.get("killzone"))

            order_status, actual_entry = check_pending_order(
                p_dir=p["direction"],
                p_entry=p["entry"],
                p_sl=p["sl"],
                cur_hi=h,
                cur_lo=l,
                bars_elapsed=p["bars_elapsed"],
                spread=SPREAD_ASSUMED,
                max_expiry_bars=ORDER_EXPIRY_BARS,
                killzone_window_open=kz_open
            )

            if order_status == "INVALIDATED":
                self._append_event("INVALIDATE", ts_utc.isoformat(), {
                    "signal_id": p["signal_id"],
                    "direction": p["direction"],
                    "entry": p["entry"],
                    "sl": p["sl"],
                    "reason": "SL_TOUCHED_BEFORE_FILL",
                    "bars_elapsed": p["bars_elapsed"]
                })
                self.pending_order = None

            elif order_status == "EXPIRED":
                self._append_event("EXPIRE", ts_utc.isoformat(), {
                    "signal_id": p["signal_id"],
                    "direction": p["direction"],
                    "entry": p["entry"],
                    "reason": f"ORDER_EXPIRY_BARS_{ORDER_EXPIRY_BARS}",
                    "bars_elapsed": p["bars_elapsed"]
                })
                self.pending_order = None

            elif order_status == "FILLED":
                self._append_event("FILL", ts_utc.isoformat(), {
                    "signal_id": p["signal_id"],
                    "direction": p["direction"],
                    "entry_target": p["entry"],
                    "fill_price": actual_entry,
                    "sl": p["sl"],
                    "tp": p["tp"],
                    "planned_risk_dist": p["risk_dist"],
                    "bars_elapsed": p["bars_elapsed"],
                    "realized_spread_at_fill": cur_spread,
                    "assumed_spread": SPREAD_ASSUMED
                })

                if self.track_spread and self.execution_tracker is not None:
                    fill_kz = is_in_killzone(ts_ny.time())
                    spread_obs = self.execution_tracker.build_spread_observation(
                        trigger_event="FILL",
                        ref_id=p["signal_id"],
                        timestamp_ny=ts_ny.isoformat(),
                        session=fill_kz or p.get("killzone") or "Outside Killzone",
                        tick_bid=tick_bid,
                        tick_ask=tick_ask,
                        bar_spread_raw=bar_spread_raw,
                        realized_spread=cur_spread
                    )
                    self._append_event("SPREAD_OBS", ts_utc.isoformat(), spread_obs)

                same_candle_sl, actual_exit = check_same_candle_sl(
                    p_dir=p["direction"],
                    p_sl=p["sl"],
                    cur_hi=h,
                    cur_lo=l,
                    spread=SPREAD_ASSUMED,
                    slippage=SLIPPAGE_ASSUMED
                )

                if same_candle_sl:
                    if p["direction"] == "BUY":
                        pnl_pts = actual_exit - actual_entry
                    else:
                        pnl_pts = actual_entry - actual_exit
                    comm_r = COMMISSION_PER_LOT / (config.CONTRACT_SIZE * p["risk_dist"]) if p["risk_dist"] > 0 else 0.0
                    gross_r = pnl_pts / p["risk_dist"] if p["risk_dist"] > 0 else -1.0
                    net_r = round(gross_r - comm_r, 3)

                    self._append_event("CLOSE", ts_utc.isoformat(), {
                        "signal_id": p["signal_id"],
                        "direction": p["direction"],
                        "entry_price": actual_entry,
                        "exit_price": actual_exit,
                        "exit_reason": "SL (Conservative Same-Candle)",
                        "net_r": net_r,
                        "planned_risk_dist": p["risk_dist"],
                        "fill_time_utc": ts_utc.isoformat(),
                        "exit_time_utc": ts_utc.isoformat(),
                        "realized_spread_at_exit": cur_spread
                    })
                    self.pending_order = None
                else:
                    self.active_position = {
                        "signal_id": p["signal_id"],
                        "direction": p["direction"],
                        "entry_price": actual_entry,
                        "sl": p["sl"],
                        "tp": p["tp"],
                        "fill_time_utc": ts_utc.isoformat(),
                        "planned_risk_dist": p["risk_dist"]
                    }
                    self.pending_order = None

        # 4. Evaluate New Signal on Closed Candle (when flat and no pending order)
        if self.active_position is None and self.pending_order is None:
            n_buffered = len(self._times_ny)
            if n_buffered < self.min_buffer_warmup:
                # Buffer in warm-up period (< min_buffer_warmup): do not evaluate signals yet
                return None

            kz_now = is_in_killzone(ts_ny.time())
            if not self.skip_outside_kz_eval or kz_now:
                df_eval = pd.DataFrame({
                    "open": self._opens,
                    "high": self._highs,
                    "low": self._lows,
                    "close": self._closes,
                    "volume": self._volumes
                }, index=pd.to_datetime(self._times_ny, utc=True).tz_convert(NY_TZ))

                t_idx = n_buffered - 1
                sig = evaluate_causal_signal_at_bar(df_eval, t_idx, killzone_gate=False)
                if sig is not None:
                    self.signal_counter += 1
                    sig_id = f"SIG-{self.signal_counter:04d}"

                    kz = sig["killzone"]
                    risk_dist = sig["risk_dist"]
                    stop_pct = round((risk_dist / c) * 100.0, 4)

                    is_kz_valid = (kz in ("LONDON", "NY_AM")) if self.killzone_gate else True
                    is_stop_valid = (stop_pct >= P_SCALED) if self.p_scaled_filter else True

                    passed_primary = is_kz_valid and is_stop_valid

                    if passed_primary:
                        filter_status = "PASSED_PRIMARY"
                    elif not is_kz_valid:
                        filter_status = "SKIPPED_OUTSIDE_KILLZONE"
                    else:
                        filter_status = "SKIPPED_STOP_TOO_TIGHT"

                    sig_data = {
                        "signal_id": sig_id,
                        "signal_num": self.signal_counter,
                        "timestamp_ny": ts_ny.isoformat(),
                        "direction": sig["direction"],
                        "entry": sig["entry"],
                        "sl": sig["sl"],
                        "tp": sig["tp"],
                        "risk_dist": risk_dist,
                        "stop_pct": stop_pct,
                        "p_scaled_floor": P_SCALED,
                        "killzone": kz,
                        "candle_ohlc": [o, h, l, c],
                        "passed_primary": passed_primary,
                        "filter_status": filter_status,
                        "realized_spread_at_signal": cur_spread,
                        "git_commit": get_git_commit(),
                        "params_hash": PARAMS_HASH
                    }

                    self._append_event("SIGNAL", ts_utc.isoformat(), sig_data)

                    if self.track_spread and self.execution_tracker is not None:
                        spread_obs = self.execution_tracker.build_spread_observation(
                            trigger_event="SIGNAL",
                            ref_id=sig_id,
                            timestamp_ny=ts_ny.isoformat(),
                            session=kz or "Outside Killzone",
                            tick_bid=tick_bid,
                            tick_ask=tick_ask,
                            bar_spread_raw=bar_spread_raw,
                            realized_spread=cur_spread
                        )
                        self._append_event("SPREAD_OBS", ts_utc.isoformat(), spread_obs)

                    # Only register pending order if signal passed primary filter
                    if passed_primary:
                        self.pending_order = {
                            "signal_id": sig_id,
                            "direction": sig["direction"],
                            "entry": sig["entry"],
                            "sl": sig["sl"],
                            "tp": sig["tp"],
                            "risk_dist": risk_dist,
                            "killzone": kz,
                            "sig_time_utc": ts_utc.isoformat(),
                            "bars_elapsed": 0
                        }


def replay_dry_run(
    candles_df,
    log_path="forward/dry_run_signals.jsonl",
    anchor_callback=None,
    killzone_gate=True,
    p_scaled_filter=True
):
    """
    Replays historical candle series through ForwardSignalLogger for dry-run verification.
    Processes closed candles one-by-one without any live broker connection.
    Returns summary stats dict: valid, record_count, candles_replayed, signals, fills, closes.
    """
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = ForwardSignalLogger(
        forward_start_date="2026-11-12T00:00:00Z",
        log_path=log_path,
        check_git=False,
        check_frozen_hashes=False,
        anchor_callback=anchor_callback,
        killzone_gate=killzone_gate,
        p_scaled_filter=p_scaled_filter
    )

    for ts, row in candles_df.iterrows():
        c_dict = {
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0))
        }
        logger.on_candle_closed(c_dict)

    valid, msg, count = verify_chain(log_path)

    # Read event counts
    counts = {"SIGNAL": 0, "FILL": 0, "CLOSE": 0, "EXPIRE": 0, "INVALIDATE": 0, "GAP": 0}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            evt = r.get("event")
            if evt in counts:
                counts[evt] += 1

    return {
        "valid": valid,
        "msg": msg,
        "record_count": count,
        "candles_replayed": len(candles_df),
        "events": counts,
        "log_path": log_path
    }

