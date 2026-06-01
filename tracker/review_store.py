"""
Cumulative per-movie review store for backtesting.

Each snapshot captures market prices and the model's output, but NOT the
raw reviews the model saw -- storing 200 reviews in every 15-min snapshot
would duplicate the same data hundreds of times per movie.

Instead we keep one cumulative file per movie: data/reviews/{ticker}.json.
Every review carries a create_date (when RT published it). To reconstruct
the exact review set available at any past moment T -- which is what you
need to replay a new model (e.g. a rebuilt drift predictor) against the
real historical Kalshi prices we recorded -- you filter to reviews with
create_date <= T.

This is the irreplaceable pairing for a true P&L backtest:
  - historical Kalshi prices       (in data/snapshots.jsonl)
  - point-in-time review inputs    (reconstructed here via create_date)
  - actual settlement outcome      (in data/snapshots.jsonl)

Quotes are dropped to keep files small; everything a model needs is kept.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
REVIEW_DIR = ROOT / "data" / "reviews"

# Fields kept per review. We drop the long 'quote' text -- not used by any
# model and it would bloat the store ~10x.
KEEP_FIELDS = (
    "critic_name", "sentiment", "publication", "top_critic",
    "rating_text", "create_date", "review_id",
)


def _path(event_ticker):
    return REVIEW_DIR / f"{event_ticker}.json"


def upsert_reviews(event_ticker, movie, rt_slug, reviews):
    """Merge a freshly-scraped review list into the cumulative store.

    Dedup by review_id (falling back to critic_name+create_date when an
    older scrape lacked review_id). Keeps the EARLIEST create_date seen for
    each review so re-publication date bumps don't corrupt arrival order.
    Returns the number of new reviews added.
    """
    if not reviews:
        return 0

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(event_ticker)

    existing = {}
    if path.exists():
        try:
            with open(path) as f:
                blob = json.load(f)
            for r in blob.get("reviews", []):
                existing[_key(r)] = r
        except (json.JSONDecodeError, IOError):
            existing = {}

    added = 0
    for r in reviews:
        key = _key(r)
        compact = {k: r.get(k, "") for k in KEEP_FIELDS}
        if key in existing:
            # Keep earliest create_date if we now have an earlier one.
            old_date = existing[key].get("create_date", "")
            new_date = compact.get("create_date", "")
            if new_date and (not old_date or new_date < old_date):
                existing[key]["create_date"] = new_date
        else:
            existing[key] = compact
            added += 1

    blob = {
        "event_ticker": event_ticker,
        "movie": movie,
        "rt_slug": rt_slug,
        "review_count": len(existing),
        "reviews": sorted(
            existing.values(),
            key=lambda x: x.get("create_date", "") or "",
        ),
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=0)
    return added


def reviews_as_of(event_ticker, timestamp_iso):
    """Return the review set that existed at a given ISO timestamp.

    Filters the cumulative store to reviews with create_date <= timestamp.
    This is the function a backtest calls to feed a candidate model the
    exact inputs available when a historical snapshot was taken.
    """
    path = _path(event_ticker)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            blob = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    out = []
    for r in blob.get("reviews", []):
        cd = r.get("create_date", "")
        # Reviews with no date are conservatively included (assume they
        # predate everything -- they came from an old scrape).
        if not cd or cd <= timestamp_iso:
            out.append(r)
    return out


def _key(r):
    rid = r.get("review_id")
    if rid:
        return f"id:{rid}"
    return f"nm:{r.get('critic_name','')}|{r.get('create_date','')}"


if __name__ == "__main__":
    import sys
    # Quick inspection: python -m tracker.review_store [event_ticker]
    if len(sys.argv) > 1:
        tk = sys.argv[1]
        path = _path(tk)
        if not path.exists():
            print(f"No review store for {tk}")
        else:
            with open(path) as f:
                blob = json.load(f)
            revs = blob["reviews"]
            print(f"{tk}: {len(revs)} reviews")
            if revs:
                dated = [r for r in revs if r.get("create_date")]
                if dated:
                    print(f"  Earliest: {dated[0]['create_date']}")
                    print(f"  Latest:   {dated[-1]['create_date']}")
    else:
        if REVIEW_DIR.exists():
            for p in sorted(REVIEW_DIR.glob("*.json")):
                with open(p) as f:
                    blob = json.load(f)
                print(f"  {p.stem}: {blob.get('review_count')} reviews")
        else:
            print("No review store yet.")
