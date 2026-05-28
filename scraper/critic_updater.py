"""
Incremental critic database updater.

After a movie settles on Kalshi, we know its final tomatometer and which
critics reviewed it. This module feeds that data back into the critic
database so calibration improves with every resolved movie.

Two entry points:
  ingest_settled_movie() -- called by settlement workflow after a movie resolves
  refresh_from_snapshots() -- batch-process all resolved snapshots

The key insight: every settled movie is a new calibration data point.
A critic who reviewed Star Wars at 61% and got it "right" (Fresh when
consensus is Fresh, or Rotten when Rotten) has their agreement rate
updated to reflect that.
"""
import csv
import json
import re
from pathlib import Path

import config

ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "critic_database.csv"
REVIEWS_PATH = ROOT / "critic_reviews.json"
CACHE_DIR = Path.home() / ".cache" / "kalshi-rt"
MOVIES_CACHE = CACHE_DIR / "movie_scores.json"
PROGRESS_PATH = CACHE_DIR / "critic_scrape_v2.json"
REVIEWS_CACHE = CACHE_DIR / "reviews"
CI_SNAPSHOTS = ROOT / "data" / "snapshots.jsonl"

FIELDNAMES = [
    "critic_name", "top_critic", "publisher_name",
    "total_reviews", "agreement_rate", "big_movie_agreement_rate",
    "fresh_rate", "tier",
]


def ingest_settled_movie(rt_slug, actual_score, reviews=None, force=False):
    """Feed a settled movie's data back into the critic database.

    IMPORTANT: Only call this AFTER a movie has actually settled on Kalshi.
    The settlement score determines whether each critic was "right" or "wrong",
    so ingesting a pre-settlement score contaminates the database. The score
    can shift between now and settlement Monday.

    Args:
        rt_slug: RT movie slug (e.g. "m/star_wars_the_mandalorian_and_grogu")
        actual_score: final tomatometer at settlement
        reviews: list of review dicts from scraper.rt_reviews.scrape_reviews().
                 If None, loads from the review cache.
        force: skip the settlement verification check (use only for testing)

    Returns dict with counts of what was updated.
    """
    if not force:
        # Verify this movie's Kalshi market has actually closed
        try:
            from market.kalshi_client import KalshiClient
            from market.mapper import TickerMapper
            from datetime import datetime, timezone
            client = KalshiClient()
            mapper = TickerMapper()
            events = client.get_rt_events()
            for event in events:
                slug = mapper.get_rt_slug(event)
                if slug and slug == rt_slug:
                    markets = client.get_markets(event["event_ticker"])
                    if markets:
                        close_str = markets[0].get("close_time", "")
                        if close_str:
                            close_time = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                            if close_time > datetime.now(timezone.utc):
                                return {"error": f"Market for {rt_slug} hasn't closed yet (closes {close_str}). "
                                        f"Don't ingest pre-settlement scores -- they can change. "
                                        f"Use force=True only for testing."}
        except Exception:
            pass  # If we can't verify, proceed (market may already be delisted)
    if reviews is None:
        reviews = _load_cached_reviews(rt_slug)
    if not reviews:
        # No local cache (typical in CI). Fetch fresh from RT.
        try:
            from scraper.rt_page import get_movie_summary
            from scraper.rt_reviews import scrape_reviews
            slug_to_fetch = rt_slug if rt_slug.startswith("m/") else f"m/{rt_slug}"
            summary = get_movie_summary(slug_to_fetch)
            if summary and summary.get("ems_id"):
                reviews = scrape_reviews(
                    summary["ems_id"], slug=slug_to_fetch,
                    expected_count=summary.get("review_count"),
                )
                print(f"  [critic_updater] Fetched {len(reviews) if reviews else 0} reviews from RT for {rt_slug}")
        except Exception as e:
            print(f"  [critic_updater] RT fetch failed for {rt_slug}: {e}")

    if not reviews:
        return {"error": f"No reviews found for {rt_slug} (local cache empty and RT fetch failed)"}

    ems_id = _get_ems_id(rt_slug, reviews)

    movie_map = _load_movie_map()
    movie_map[ems_id] = {"slug": f"/{rt_slug}" if not rt_slug.startswith("/") else rt_slug,
                         "tomatometer": actual_score}
    _save_movie_map(movie_map)

    movie_review_counts = _load_movie_review_counts()
    movie_review_counts[ems_id] = movie_review_counts.get(ems_id, 0)

    critic_db = _load_csv()
    review_data = _load_review_json()

    updated = 0
    new_big = 0
    consensus_fresh = actual_score >= 60

    for r in reviews:
        name = r.get("critic_name", "")
        if not name:
            continue

        nl = name.lower()
        sentiment = r.get("sentiment", "")
        is_fresh = sentiment in ("Fresh", "POSITIVE")
        correct = is_fresh == consensus_fresh

        if ems_id not in movie_review_counts or movie_review_counts[ems_id] == 0:
            movie_review_counts[ems_id] = len(reviews)

        is_big = movie_review_counts.get(ems_id, 0) >= config.BIG_MOVIE_REVIEW_THRESHOLD

        if nl not in critic_db:
            continue

        critic = critic_db[nl]
        rd = review_data.get(name, review_data.get(nl, {}))

        old_n = rd.get("_settlement_n", 0)
        old_agree = rd.get("_settlement_agree", 0)
        old_big_n = rd.get("_settlement_big_n", 0)
        old_big_agree = rd.get("_settlement_big_agree", 0)

        new_n = old_n + 1
        new_agree = old_agree + (1 if correct else 0)

        rd["_settlement_n"] = new_n
        rd["_settlement_agree"] = new_agree

        if is_big:
            new_big_n = old_big_n + 1
            new_big_agree = old_big_agree + (1 if correct else 0)
            rd["_settlement_big_n"] = new_big_n
            rd["_settlement_big_agree"] = new_big_agree

            existing_big = critic.get("big_movie_agreement_rate", "")
            existing_big_n = rd.get("big_movie_agreement_n", 0)

            if existing_big and existing_big_n:
                total_big_n = existing_big_n + new_big_n
                total_big_agree = round(float(existing_big) * existing_big_n) + new_big_agree
                critic["big_movie_agreement_rate"] = round(total_big_agree / total_big_n, 4)
                rd["big_movie_agreement_n"] = total_big_n
            elif new_big_n >= 3:
                critic["big_movie_agreement_rate"] = round(new_big_agree / new_big_n, 4)
                rd["big_movie_agreement_n"] = new_big_n
                new_big += 1

        review_data[name] = rd
        updated += 1

    _save_csv(critic_db)
    _save_review_json(review_data)

    return {
        "movie": rt_slug,
        "actual_score": actual_score,
        "reviews_processed": len(reviews),
        "critics_updated": updated,
        "new_big_movie_calibrated": new_big,
    }


def refresh_from_snapshots():
    """Batch-process all resolved snapshots to update critic database.

    Reads data/snapshots.jsonl, finds resolved movies, and calls
    ingest_settled_movie() for each unique resolved movie.
    """
    if not CI_SNAPSHOTS.exists():
        print("No snapshots file found")
        return []

    resolved_movies = {}
    with open(CI_SNAPSHOTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                if s.get("resolved") and s.get("actual_score") is not None:
                    ticker = s["event_ticker"]
                    if ticker not in resolved_movies:
                        resolved_movies[ticker] = {
                            "movie": s.get("movie"),
                            "rt_slug": s.get("rt_slug"),
                            "actual_score": s["actual_score"],
                        }
            except (json.JSONDecodeError, KeyError):
                continue

    if not resolved_movies:
        print("No resolved movies in snapshots")
        return []

    results = []
    for ticker, info in resolved_movies.items():
        slug = info["rt_slug"]
        if not slug:
            continue
        print(f"  Ingesting {info['movie']} ({slug}): score={info['actual_score']}%")
        result = ingest_settled_movie(slug, info["actual_score"])
        result["event_ticker"] = ticker
        results.append(result)
        if "error" not in result:
            print(f"    Updated {result['critics_updated']} critics, {result['new_big_movie_calibrated']} gained big_movie rate")

    return results


def _load_cached_reviews(rt_slug):
    slug_part = rt_slug.replace("/", "_").replace("m_", "", 1)
    if not REVIEWS_CACHE.exists():
        return []
    files = sorted(REVIEWS_CACHE.glob(f"*{slug_part}*"), reverse=True)
    if not files:
        return []
    try:
        with open(files[0]) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _get_ems_id(rt_slug, reviews):
    """Try to get EMS ID from reviews or generate a stable one from slug."""
    for r in reviews:
        ems = r.get("ems_id") or r.get("movie_ems_id")
        if ems:
            return ems
    return f"settled_{rt_slug.replace('/', '_')}"


def _load_movie_map():
    if MOVIES_CACHE.exists():
        try:
            with open(MOVIES_CACHE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_movie_map(movie_map):
    MOVIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MOVIES_CACHE, "w") as f:
        json.dump(movie_map, f)


def _load_movie_review_counts():
    if not PROGRESS_PATH.exists():
        return {}
    try:
        with open(PROGRESS_PATH) as f:
            progress = json.load(f)
        counts = {}
        for critic_data in progress.get("critics", {}).values():
            for r in critic_data.get("reviews", []):
                ems = r.get("movie_ems_id")
                if ems:
                    counts[ems] = counts.get(ems, 0) + 1
        return counts
    except (json.JSONDecodeError, IOError):
        return {}


def _load_csv():
    db = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            db[row["critic_name"].lower()] = dict(row)
    return db


def _save_csv(db):
    rows = sorted(db.values(), key=lambda x: x.get("critic_name", "").lower())
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def _load_review_json():
    if REVIEWS_PATH.exists():
        try:
            with open(REVIEWS_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_review_json(data):
    with open(REVIEWS_PATH, "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Incremental critic database updater")
    parser.add_argument("--from-snapshots", action="store_true",
                        help="Process all resolved snapshots and update critic DB")
    parser.add_argument("--ingest", nargs=2, metavar=("SLUG", "SCORE"),
                        help="Ingest a single settled movie")
    args = parser.parse_args()

    if args.from_snapshots:
        results = refresh_from_snapshots()
        total = sum(r.get("critics_updated", 0) for r in results if "error" not in r)
        print(f"\nProcessed {len(results)} movies, updated {total} critic records")
    elif args.ingest:
        slug, score = args.ingest
        result = ingest_settled_movie(slug, int(score))
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
