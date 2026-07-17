"""
Snapshot history loader with monthly rotation.

data/snapshots.jsonl grew past GitHub's 100 MB per-file push limit, which broke
every commit (the scrape itself was fine -- only the push failed). To keep
collecting we rotate: the active file holds only the current month; older months
are gzipped into data/archive/snapshots-YYYY-MM.jsonl.gz (a fraction of the raw
size, well under the limit).

Tools that need the FULL history call iter_all_snapshot_lines() / iter_all_snapshots(),
which streams the gzipped archives then the active file, in chronological order.
Hot-path writers and "latest per event" readers keep using the active file
directly -- they only need recent data anyway.
"""
import gzip
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
ARCHIVE_DIR = DATA_DIR / "archive"
ACTIVE = DATA_DIR / "snapshots.jsonl"


def iter_all_snapshot_lines():
    """Yield raw JSONL lines from all archives (oldest first) then the active file."""
    if ARCHIVE_DIR.exists():
        for gz in sorted(ARCHIVE_DIR.glob("snapshots-*.jsonl.gz")):
            with gzip.open(gz, "rt") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    if ACTIVE.exists():
        with open(ACTIVE) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def iter_all_snapshots():
    """Yield parsed snapshot dicts from the full history (archives + active)."""
    for line in iter_all_snapshot_lines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def rotate_if_needed(active_path=ACTIVE, prefix="snapshots", max_active_mb=60):
    """Roll completed months out of an append-only JSONL into gzipped archives.

    Generic over any timestamped JSONL (snapshots, orderbook depth, ...). Keeps
    the current month in the active file; older months move to
    data/archive/{prefix}-YYYY-MM.jsonl.gz. No-op unless the active file exceeds
    max_active_mb, so most runs do nothing. Returns records archived.
    """
    active_path = Path(active_path)
    if not active_path.exists():
        return 0
    if active_path.stat().st_size < max_active_mb * 1024 * 1024:
        return 0

    by_month = {}
    with open(active_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                month = json.loads(line).get("timestamp", "")[:7]
            except json.JSONDecodeError:
                month = ""
            by_month.setdefault(month or "unknown", []).append(line)

    if len(by_month) <= 1:
        return 0  # all one month; nothing to roll out yet

    current = max(by_month)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived = 0
    for month, lines in by_month.items():
        if month == current:
            continue
        gz = ARCHIVE_DIR / f"{prefix}-{month}.jsonl.gz"
        existing = []
        if gz.exists():
            with gzip.open(gz, "rt") as f:
                existing = [l.strip() for l in f if l.strip()]
        with gzip.open(gz, "wt") as f:
            for l in existing + lines:
                f.write(l + "\n")
        archived += len(lines)

    with open(active_path, "w") as f:
        for l in by_month[current]:
            f.write(l + "\n")
    return archived


def iter_archived_lines(prefix):
    """Yield raw lines from gzipped archives for a given prefix (oldest first)."""
    if not ARCHIVE_DIR.exists():
        return
    for gz in sorted(ARCHIVE_DIR.glob(f"{prefix}-*.jsonl.gz")):
        with gzip.open(gz, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


if __name__ == "__main__":
    # One-time / manual split: force-archive all completed months now.
    import sys
    n = rotate_if_needed(max_active_mb=0)  # 0 => always rotate
    total = sum(1 for _ in iter_all_snapshot_lines())
    active = sum(1 for _ in open(ACTIVE)) if ACTIVE.exists() else 0
    print(f"Archived {n} records. Active now {active}, total across all files {total}.")
    if ARCHIVE_DIR.exists():
        for gz in sorted(ARCHIVE_DIR.glob("*.gz")):
            print(f"  {gz.name}: {gz.stat().st_size/1024/1024:.1f} MB")
