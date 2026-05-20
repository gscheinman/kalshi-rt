"""
Integrate enriched critic data (from targeted scrape of Kalshi movie reviewers)
into the main critic_database.csv and critic_reviews.json.

Reads:
  ~/.cache/kalshi-rt/enriched_critics.json  -- reviews from targeted scrape
  ~/.cache/kalshi-rt/movie_scores.json      -- tomatometer cache
  critic_database.csv                        -- current merged database
  critic_reviews.json                        -- current review data

Writes:
  critic_database.csv  -- updated with enriched critics
  critic_reviews.json  -- updated with enriched critics' review data
"""
import csv
import json
import re
import string
from pathlib import Path

import config

ROOT = Path(__file__).parent.parent
CACHE_DIR = Path.home() / ".cache" / "kalshi-rt"
ENRICHED_PATH = CACHE_DIR / "enriched_critics.json"
MOVIES_CACHE = CACHE_DIR / "movie_scores.json"
CSV_PATH = ROOT / "critic_database.csv"
REVIEWS_PATH = ROOT / "critic_reviews.json"

FIELDNAMES = [
    "critic_name", "top_critic", "publisher_name",
    "total_reviews", "agreement_rate", "big_movie_agreement_rate",
    "fresh_rate", "tier",
]


def parse_score_to_number(score_text):
    if not score_text:
        return None
    text = score_text.strip()
    grade_map = {
        "A+": 0.97, "A": 0.93, "A-": 0.90,
        "B+": 0.87, "B": 0.83, "B-": 0.80,
        "C+": 0.77, "C": 0.73, "C-": 0.70,
        "D+": 0.67, "D": 0.63, "D-": 0.60,
        "F": 0.50,
    }
    if text.upper() in grade_map:
        return grade_map[text.upper()]
    m = re.match(r"([\d.]+)\s*/\s*([\d.]+)", text)
    if m:
        num, denom = float(m.group(1)), float(m.group(2))
        if denom > 0:
            return min(1.0, num / denom)
    return None


def parse_review(raw):
    """Convert raw NAPI review object to our standard format."""
    critic = raw.get("critic", {})
    media = raw.get("media", {})
    return {
        "sentiment": critic.get("scoreSentiment", ""),
        "original_score": critic.get("originalScore", ""),
        "publication": critic.get("publicationName", ""),
        "date": critic.get("reviewCreateDate", ""),
        "movie_title": media.get("title", ""),
        "movie_ems_id": media.get("emsId", ""),
        "movie_slug": media.get("url", ""),
    }


def compute_calibration(all_reviews, movie_map, movie_review_counts):
    """Compute agreement rates for a set of reviews, same logic as phase 3."""
    agreements = 0
    total_with_score = 0
    big_agreements = 0
    big_total = 0
    fresh_scores = []
    rotten_scores = []
    fresh_scores_big = []
    rotten_scores_big = []

    for r in all_reviews:
        ems = r.get("movie_ems_id")
        if not ems or ems not in movie_map:
            continue
        tomato = movie_map[ems].get("tomatometer")
        if tomato is None:
            continue

        total_with_score += 1
        is_fresh = r["sentiment"] == "POSITIVE"
        consensus_fresh = tomato >= 60
        correct = is_fresh == consensus_fresh

        if correct:
            agreements += 1

        review_count = movie_review_counts.get(ems, 0)
        is_big = review_count >= config.BIG_MOVIE_REVIEW_THRESHOLD

        if is_big:
            big_total += 1
            if correct:
                big_agreements += 1

        if is_fresh:
            fresh_scores.append(tomato)
            if is_big:
                fresh_scores_big.append(tomato)
        else:
            rotten_scores.append(tomato)
            if is_big:
                rotten_scores_big.append(tomato)

    result = {}
    if total_with_score >= 5:
        result["agreement_rate"] = round(agreements / total_with_score, 4)
        result["agreement_n"] = total_with_score
    if big_total >= 3:
        result["big_movie_agreement_rate"] = round(big_agreements / big_total, 4)
        result["big_movie_agreement_n"] = big_total
    if fresh_scores:
        result["avg_tomato_when_fresh"] = round(sum(fresh_scores) / len(fresh_scores), 1)
    if rotten_scores:
        result["avg_tomato_when_rotten"] = round(sum(rotten_scores) / len(rotten_scores), 1)
    if fresh_scores_big:
        result["big_avg_tomato_when_fresh"] = round(sum(fresh_scores_big) / len(fresh_scores_big), 1)
    if rotten_scores_big:
        result["big_avg_tomato_when_rotten"] = round(sum(rotten_scores_big) / len(rotten_scores_big), 1)

    return result


def main():
    if not ENRICHED_PATH.exists():
        print(f"ERROR: {ENRICHED_PATH} not found. Run the enrichment script first.")
        return

    print("Loading data...")
    with open(ENRICHED_PATH) as f:
        enriched = json.load(f)
    print(f"  Enriched critics: {len(enriched)}")

    with open(MOVIES_CACHE) as f:
        movie_map = json.load(f)
    print(f"  Movies with scores: {sum(1 for m in movie_map.values() if m.get('tomatometer') is not None)}")

    # Load existing DB and reviews
    existing_db = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            existing_db[row["critic_name"].lower()] = dict(row)
    print(f"  Existing DB: {len(existing_db)} critics")

    existing_reviews = {}
    if REVIEWS_PATH.exists():
        with open(REVIEWS_PATH) as f:
            existing_reviews = json.load(f)
    print(f"  Existing review data: {len(existing_reviews)} critics")

    # Build movie review count from existing progress data
    # (needed for big_movie threshold computation)
    progress_path = CACHE_DIR / "critic_scrape_v2.json"
    movie_review_counts = {}
    if progress_path.exists():
        with open(progress_path) as f:
            progress = json.load(f)
        for critic_data in progress["critics"].values():
            for r in critic_data.get("reviews", []):
                ems = r.get("movie_ems_id")
                if ems:
                    movie_review_counts[ems] = movie_review_counts.get(ems, 0) + 1
    print(f"  Movie review counts: {len(movie_review_counts)} movies")
    print(f"  Big movies (80+ reviews): {sum(1 for c in movie_review_counts.values() if c >= config.BIG_MOVIE_REVIEW_THRESHOLD)}")

    # Process each enriched critic
    added = 0
    updated = 0
    with_big = 0

    for slug, data in enriched.items():
        name = data["name"]
        pub = data.get("pub", "")
        reviews_raw = data.get("reviews_raw", [])

        # Parse reviews into standard format
        reviews = [parse_review(r) for r in reviews_raw]
        reviews = [r for r in reviews if r.get("sentiment") and r.get("movie_ems_id")]

        if not reviews:
            continue

        total = len(reviews)
        fresh = sum(1 for r in reviews if r["sentiment"] == "POSITIVE")
        fresh_rate = fresh / total if total else 0

        cal = compute_calibration(reviews, movie_map, movie_review_counts)
        big_rate = cal.get("big_movie_agreement_rate")

        if big_rate is not None:
            with_big += 1

        # Build DB row
        if total >= 100:
            tier = "veteran"
        elif total >= 30:
            tier = "established"
        else:
            tier = "newcomer"

        row = {
            "critic_name": name,
            "top_critic": "False",
            "publisher_name": pub,
            "total_reviews": total,
            "agreement_rate": cal.get("agreement_rate", max(fresh_rate, 1 - fresh_rate)),
            "big_movie_agreement_rate": big_rate if big_rate is not None else "",
            "fresh_rate": round(fresh_rate, 4),
            "tier": tier,
        }

        nl = name.lower()
        if nl in existing_db:
            # Only update if we have better data (big_movie rate or more reviews)
            existing = existing_db[nl]
            old_big = existing.get("big_movie_agreement_rate", "")
            if (not old_big and big_rate is not None) or (total > int(existing.get("total_reviews", 0))):
                existing_db[nl] = row
                updated += 1
        else:
            existing_db[nl] = row
            added += 1

        # Update review data
        existing_reviews[name] = {
            "slug": slug,
            "fresh_rate": round(fresh_rate, 4),
            "agreement_rate": cal.get("agreement_rate"),
            "big_movie_agreement_rate": big_rate,
            "big_movie_agreement_n": cal.get("big_movie_agreement_n", 0),
            "avg_tomato_when_fresh": cal.get("avg_tomato_when_fresh"),
            "avg_tomato_when_rotten": cal.get("avg_tomato_when_rotten"),
            "big_avg_tomato_when_fresh": cal.get("big_avg_tomato_when_fresh"),
            "big_avg_tomato_when_rotten": cal.get("big_avg_tomato_when_rotten"),
            "reviews": reviews,
        }

    print(f"\nEnrichment results:")
    print(f"  Added: {added} new critics")
    print(f"  Updated: {updated} existing critics")
    print(f"  With big_movie_agreement_rate: {with_big}/{len(enriched)}")

    # Write updated DB
    rows = sorted(existing_db.values(), key=lambda x: x["critic_name"].lower())
    total_with_big = sum(1 for r in rows if r.get("big_movie_agreement_rate", ""))
    print(f"\nFinal DB: {len(rows)} critics, {total_with_big} with big_movie_agreement_rate ({total_with_big/len(rows)*100:.1f}%)")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    with open(REVIEWS_PATH, "w") as f:
        json.dump(existing_reviews, f, indent=1)

    print(f"Updated {CSV_PATH.name} and {REVIEWS_PATH.name}")


if __name__ == "__main__":
    main()
