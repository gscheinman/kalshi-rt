"""
Scrape the RT critic directory to build a rich critic database.
Uses RT's internal NAPI -- no browser automation needed.

Two-phase scrape:
  Phase 1: Collect all critics and their individual review scores + movie IDs
  Phase 2: Fetch final tomatometers for each unique movie
  Phase 3: Compute per-critic calibration (score -> final tomatometer correlation)

Usage:
    python -m scraper.rt_critics

Outputs:
  critic_database_new.csv  -- drop-in replacement for critic_database.csv
  critic_reviews.json      -- raw per-critic review data for the model
"""
import csv
import json
import re
import string
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}
HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}
CRITICS_API = "https://www.rottentomatoes.com/napi/critics/authors"
REVIEWS_API = "https://www.rottentomatoes.com/napi/critics/{slug}/movies"
DELAY = 0.3
ROOT = Path(__file__).parent.parent
CSV_OUTPUT = ROOT / "critic_database_new.csv"
REVIEWS_OUTPUT = ROOT / "critic_reviews.json"
CACHE_DIR = Path.home() / ".cache" / "kalshi-rt"
PROGRESS_PATH = CACHE_DIR / "critic_scrape_v2.json"
MOVIES_CACHE = CACHE_DIR / "movie_scores.json"


def scrape_all():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    progress = _load_progress()
    all_critics = progress.get("critics", {})
    completed_letters = set(progress.get("completed_letters", []))
    phase = progress.get("phase", "critics")

    # Movie map is saved separately (too large for progress file)
    movie_map = {}
    if MOVIES_CACHE.exists():
        try:
            with open(MOVIES_CACHE) as f:
                movie_map = json.load(f)
            print(f"  Loaded {len(movie_map)} movies from cache ({sum(1 for v in movie_map.values() if v.get('tomatometer') is not None)} scored)", flush=True)
        except (json.JSONDecodeError, IOError):
            movie_map = {}

    if phase == "critics":
        _phase1_critics(all_critics, completed_letters, movie_map)
        progress["phase"] = "movies"
        _save_progress(all_critics, completed_letters, movie_map, "movies")

    if progress.get("phase") == "movies":
        _phase2_movies(movie_map)
        progress["phase"] = "calibrate"
        _save_progress(all_critics, completed_letters, movie_map, "calibrate")

    _phase3_calibrate(all_critics, movie_map)
    _write_outputs(all_critics, movie_map)

    p = f"\nDone! {len(all_critics)} critics -> {CSV_OUTPUT.name}"
    print(p)
    print(f"Review data -> {REVIEWS_OUTPUT.name}")


def _phase1_critics(all_critics, completed_letters, movie_map):
    print("=== Phase 1: Scraping critic reviews ===\n")

    for letter in string.ascii_lowercase:
        if letter in completed_letters:
            print(f"  Skipping '{letter}' (done)", flush=True)
            continue

        print(f"\n--- Letter '{letter}' ---", flush=True)
        slugs = _get_critic_slugs(letter)
        print(f"  Found {len(slugs)} critics", flush=True)

        for i, (name, slug, pub, is_top) in enumerate(slugs):
            if slug in all_critics:
                continue

            reviews = _get_critic_reviews(slug)
            if not reviews or len(reviews) < 5:
                time.sleep(DELAY)
                continue

            # Collect movie EMS IDs for phase 2
            for r in reviews:
                ems = r.get("movie_ems_id")
                slug_m = r.get("movie_slug")
                if ems and ems not in movie_map:
                    movie_map[ems] = {"slug": slug_m, "tomatometer": None}

            total = len(reviews)
            fresh = sum(1 for r in reviews if r["sentiment"] == "POSITIVE")
            fresh_rate = fresh / total

            all_critics[slug] = {
                "critic_name": name,
                "top_critic": is_top,
                "publisher_name": pub,
                "total_reviews": total,
                "fresh_rate": round(fresh_rate, 4),
                "slug": slug,
                "reviews": reviews,
            }

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(slugs)} critics", flush=True)
                _save_progress(all_critics, completed_letters, movie_map, "critics")

            time.sleep(DELAY)

        completed_letters.add(letter)
        _save_progress(all_critics, completed_letters, movie_map, "critics")
        print(f"  Letter '{letter}' done: {len(all_critics)} critics, {len(movie_map)} unique movies", flush=True)


def _phase2_movies(movie_map):
    """Fetch tomatometer for each unique movie."""
    to_fetch = [ems for ems, info in movie_map.items() if info.get("tomatometer") is None and info.get("slug")]
    print(f"\n=== Phase 2: Fetching tomatometers for {len(to_fetch)} movies ===\n", flush=True)

    for i, ems_id in enumerate(to_fetch):
        info = movie_map[ems_id]
        slug = info["slug"]
        if not slug:
            continue

        score = _get_tomatometer(slug)
        if score is not None:
            movie_map[ems_id]["tomatometer"] = score

        if (i + 1) % 50 == 0:
            done_count = sum(1 for m in movie_map.values() if m.get("tomatometer") is not None)
            print(f"  {i+1}/{len(to_fetch)} fetched ({done_count} with scores)", flush=True)
            with open(MOVIES_CACHE, "w") as f:
                json.dump(movie_map, f)

        time.sleep(DELAY)

    with open(MOVIES_CACHE, "w") as f:
        json.dump(movie_map, f)

    done = sum(1 for m in movie_map.values() if m.get("tomatometer") is not None)
    print(f"  Got tomatometers for {done}/{len(movie_map)} movies", flush=True)


def _phase3_calibrate(all_critics, movie_map):
    """Compute per-critic agreement rates and score calibration.

    Computes two agreement rates per critic:
      - agreement_rate: overall (all movies)
      - big_movie_agreement_rate: only movies with 80+ reviews on RT
    Kalshi only lists major releases, so the big-movie rate is more predictive
    for trading. A critic being "right" about a tiny 15-review indie doesn't
    tell us much about their accuracy on a 200-review blockbuster.
    """
    print(f"\n=== Phase 3: Computing calibration ===\n", flush=True)

    # Pre-compute review counts per movie (how many critics in our dataset reviewed it)
    movie_review_counts = {}
    for slug, critic in all_critics.items():
        for r in critic.get("reviews", []):
            ems = r.get("movie_ems_id")
            if ems:
                movie_review_counts[ems] = movie_review_counts.get(ems, 0) + 1

    for slug, critic in all_critics.items():
        reviews = critic.get("reviews", [])
        if not reviews:
            continue

        agreements = 0
        total_with_score = 0
        big_agreements = 0
        big_total = 0
        score_errors = []
        fresh_scores_all = []
        rotten_scores_all = []
        fresh_scores_big = []
        rotten_scores_big = []

        for r in reviews:
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

            if r.get("original_score"):
                parsed = _parse_score_to_number(r["original_score"])
                if parsed is not None:
                    score_errors.append({
                        "critic_score": parsed,
                        "tomatometer": tomato,
                        "sentiment": r["sentiment"],
                        "is_big": is_big,
                    })

            if is_fresh:
                fresh_scores_all.append(tomato)
                if is_big:
                    fresh_scores_big.append(tomato)
            else:
                rotten_scores_all.append(tomato)
                if is_big:
                    rotten_scores_big.append(tomato)

        if total_with_score >= 5:
            critic["agreement_rate"] = round(agreements / total_with_score, 4)
            critic["agreement_n"] = total_with_score
        else:
            critic["agreement_rate"] = max(critic["fresh_rate"], 1.0 - critic["fresh_rate"])
            critic["agreement_n"] = 0

        if big_total >= 3:
            critic["big_movie_agreement_rate"] = round(big_agreements / big_total, 4)
            critic["big_movie_agreement_n"] = big_total
        else:
            critic["big_movie_agreement_rate"] = None
            critic["big_movie_agreement_n"] = 0

        critic["avg_tomato_when_fresh"] = round(sum(fresh_scores_all) / len(fresh_scores_all), 1) if fresh_scores_all else None
        critic["avg_tomato_when_rotten"] = round(sum(rotten_scores_all) / len(rotten_scores_all), 1) if rotten_scores_all else None

        # Big-movie specific averages (more relevant for Kalshi predictions)
        critic["big_avg_tomato_when_fresh"] = round(sum(fresh_scores_big) / len(fresh_scores_big), 1) if fresh_scores_big else None
        critic["big_avg_tomato_when_rotten"] = round(sum(rotten_scores_big) / len(rotten_scores_big), 1) if rotten_scores_big else None

        if len(score_errors) >= 10:
            critic["score_calibration"] = _compute_calibration(score_errors)

    calibrated = sum(1 for c in all_critics.values() if c.get("agreement_n", 0) >= 5)
    big_calibrated = sum(1 for c in all_critics.values() if c.get("big_movie_agreement_n", 0) >= 3)
    print(f"  {calibrated}/{len(all_critics)} critics with overall agreement rates", flush=True)
    print(f"  {big_calibrated}/{len(all_critics)} critics with big-movie agreement rates", flush=True)


def _get_critic_slugs(letter):
    slugs = []
    cursor = None

    while True:
        params = {"letter": letter, "pageCount": 100, "isInactiveCriticsList": "false"}
        if cursor:
            params["after"] = cursor

        try:
            resp = requests.get(CRITICS_API, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
        except (requests.RequestException, ValueError):
            break

        for c in data.get("critics", []):
            name = c.get("name", "").strip()
            url = c.get("url", "")
            slug = url.split("/")[-1] if url else ""
            if not name or not slug:
                continue

            pub = ""
            pubs = c.get("affiliatedPublications", [])
            if pubs:
                pub = pubs[0].get("name", "")

            badges = c.get("badges", [])
            is_top = any("top" in b.lower() for b in badges) or \
                     any("approved" in b.lower() for b in badges)

            slugs.append((name, slug, pub, is_top))

        page_info = data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            break
        time.sleep(DELAY)

    return slugs


def _get_critic_reviews(slug):
    """Get a critic's last 100 reviews with scores and movie IDs."""
    try:
        resp = requests.get(
            REVIEWS_API.format(slug=slug),
            params={"pageCount": 100},
            headers=HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    reviews = []
    for r in data.get("reviews", []):
        critic_data = r.get("critic", {})
        media = r.get("media", {})
        reviews.append({
            "sentiment": critic_data.get("scoreSentiment", ""),
            "original_score": critic_data.get("originalScore", ""),
            "publication": critic_data.get("publicationName", ""),
            "date": critic_data.get("reviewCreateDate", ""),
            "movie_title": media.get("title", ""),
            "movie_ems_id": media.get("emsId", ""),
            "movie_slug": media.get("url", ""),
        })
    return reviews


def _get_tomatometer(slug):
    """Fetch the final tomatometer for a movie from its RT page."""
    if not slug:
        return None
    url = f"https://www.rottentomatoes.com{slug}"
    try:
        resp = requests.get(url, headers=HTML_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
    except requests.RequestException:
        return None

    for script in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', resp.text, re.DOTALL):
        try:
            data = json.loads(script.group(1))
            if isinstance(data, dict) and "aggregateRating" in data:
                return int(float(data["aggregateRating"]["ratingValue"]))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            pass
    return None


def _parse_score_to_number(score_text):
    """Convert critic scores like 'B+', '3/4', '7/10' to 0-1 scale."""
    if not score_text:
        return None
    text = score_text.strip()

    # Letter grades
    grade_map = {
        "A+": 0.97, "A": 0.93, "A-": 0.90,
        "B+": 0.87, "B": 0.83, "B-": 0.80,
        "C+": 0.77, "C": 0.73, "C-": 0.70,
        "D+": 0.67, "D": 0.63, "D-": 0.60,
        "F": 0.50,
    }
    if text.upper() in grade_map:
        return grade_map[text.upper()]

    # Fraction: 3/4, 7/10, 2.5/5
    m = re.match(r"([\d.]+)\s*/\s*([\d.]+)", text)
    if m:
        num, denom = float(m.group(1)), float(m.group(2))
        if denom > 0:
            return min(1.0, num / denom)

    # Star ratings
    stars = text.count("★") + text.count("⭐")
    if stars > 0:
        half = text.count("½")
        total_stars = stars + (0.5 if half else 0)
        max_stars = 5 if stars <= 5 else 10
        return total_stars / max_stars

    return None


def _compute_calibration(score_errors):
    """Build a simple mapping: critic's numeric score bucket -> avg tomatometer."""
    buckets = {}
    for e in score_errors:
        bucket = round(e["critic_score"] * 10) / 10  # round to 0.1
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(e["tomatometer"])

    calibration = {}
    for bucket, scores in sorted(buckets.items()):
        if len(scores) >= 3:
            calibration[str(bucket)] = {
                "avg_tomatometer": round(sum(scores) / len(scores), 1),
                "n": len(scores),
            }
    return calibration if calibration else None


def _load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {}


def _save_progress(all_critics, completed_letters, movie_map, phase):
    # Save full critics data (including reviews) so Phase 3 can resume
    with open(PROGRESS_PATH, "w") as f:
        json.dump({
            "critics": all_critics,
            "completed_letters": list(completed_letters),
            "movie_count": len(movie_map),
            "phase": phase,
        }, f)

    # Save movie map separately (can get large)
    with open(MOVIES_CACHE, "w") as f:
        json.dump(movie_map, f)


def _write_outputs(all_critics, movie_map):
    """Write the CSV database and the rich review JSON."""
    rows = sorted(all_critics.values(), key=lambda x: x["critic_name"].lower())

    for row in rows:
        total = row["total_reviews"]
        if total >= 100:
            row["tier"] = "veteran"
        elif total >= 30:
            row["tier"] = "established"
        else:
            row["tier"] = "newcomer"

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "critic_name", "top_critic", "publisher_name",
            "total_reviews", "agreement_rate", "big_movie_agreement_rate",
            "fresh_rate", "tier",
        ])
        writer.writeheader()
        for row in rows:
            overall = row.get("agreement_rate", max(row["fresh_rate"], 1.0 - row["fresh_rate"]))
            big = row.get("big_movie_agreement_rate")
            writer.writerow({
                "critic_name": row["critic_name"],
                "top_critic": row["top_critic"],
                "publisher_name": row["publisher_name"],
                "total_reviews": row["total_reviews"],
                "agreement_rate": overall,
                "big_movie_agreement_rate": big if big is not None else "",
                "fresh_rate": row["fresh_rate"],
                "tier": row["tier"],
            })

    # Write rich review data for model use
    review_data = {}
    for slug, critic in all_critics.items():
        review_data[critic["critic_name"]] = {
            "slug": slug,
            "fresh_rate": critic["fresh_rate"],
            "agreement_rate": critic.get("agreement_rate"),
            "big_movie_agreement_rate": critic.get("big_movie_agreement_rate"),
            "big_movie_agreement_n": critic.get("big_movie_agreement_n", 0),
            "avg_tomato_when_fresh": critic.get("avg_tomato_when_fresh"),
            "avg_tomato_when_rotten": critic.get("avg_tomato_when_rotten"),
            "big_avg_tomato_when_fresh": critic.get("big_avg_tomato_when_fresh"),
            "big_avg_tomato_when_rotten": critic.get("big_avg_tomato_when_rotten"),
            "score_calibration": critic.get("score_calibration"),
            "reviews": critic.get("reviews", []),
        }

    with open(REVIEWS_OUTPUT, "w") as f:
        json.dump(review_data, f, indent=1)


if __name__ == "__main__":
    print("Scraping RT critic directory (enhanced v2)...", flush=True)
    print(f"CSV output: {CSV_OUTPUT}", flush=True)
    print(f"Review data: {REVIEWS_OUTPUT}", flush=True)
    print(f"Progress: {PROGRESS_PATH}", flush=True)
    print(flush=True)
    scrape_all()
