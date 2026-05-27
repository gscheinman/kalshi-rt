"""
Scrape individual RT critic reviews via the internal NAPI.
No Playwright needed -- plain HTTP requests to the JSON API.
"""
import html as html_lib
import json
import os
import time
from datetime import date
from pathlib import Path

import requests

CACHE_DIR = Path.home() / ".cache" / "kalshi-rt" / "reviews"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}
REVIEW_API = "https://www.rottentomatoes.com/napi/rtcf/v1/movies/{ems_id}/reviews"

# Cache TTL: reviews go stale during the active review window. RT can publish
# 5-10 new reviews per hour after a screener. The snapshot workflow runs every
# 15 min, so a 10-min TTL means each snapshot gets fresh data without hammering
# RT (5 calls/hr per movie instead of 1/day).
CACHE_TTL_SECONDS = 600  # 10 minutes


def scrape_reviews(ems_id, slug="", max_pages=25, expected_count=None, force_refresh=False):
    """
    Fetch critic reviews from the RT internal API.

    ems_id: the EMS ID from the movie's main page (get via get_movie_summary)
    slug: used for cache file naming only
    max_pages: max pagination requests (20 reviews per page). Default 25
        covers up to 500 reviews -- enough for any wide-release film.
        Loop exits early when RT signals hasNextPage=false.
    expected_count: if provided, invalidate cache when cached review count
        is significantly below this. Pass the review_count from
        get_movie_summary() to keep snapshots in sync with RT's live count.
    force_refresh: if True, bypass cache entirely.

    Returns list of review dicts.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = slug.replace("/", "_") or ems_id
    cache_file = CACHE_DIR / f"{cache_key}_{date.today().isoformat()}.json"

    if cache_file.exists() and not force_refresh:
        age = time.time() - os.path.getmtime(cache_file)
        with open(cache_file) as f:
            cached = json.load(f)
        if cached:
            stale_by_age = age > CACHE_TTL_SECONDS
            # If RT reports notably more reviews than we have cached, refetch.
            # Allow a 2-review slop since RT pagination can be off-by-one.
            stale_by_count = (
                expected_count is not None
                and len(cached) + 2 < expected_count
            )
            if not stale_by_age and not stale_by_count:
                return cached

    # First fetch attempt.
    reviews = _fetch_from_napi(ems_id, max_pages)

    # When RT main page says reviews exist but the NAPI returns empty, treat
    # it as a transient hiccup. Retry once. If still empty, fall back to
    # whatever's cached rather than overwriting good history with garbage.
    if not reviews and expected_count and expected_count > 0:
        time.sleep(2)
        reviews = _fetch_from_napi(ems_id, max_pages)
        if not reviews and cache_file.exists():
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
                if cached:
                    print(
                        f"  [scrape_reviews] NAPI returned empty for "
                        f"{slug or ems_id} (RT says {expected_count} reviews); "
                        f"falling back to stale cache of {len(cached)}",
                        flush=True,
                    )
                    return cached
            except (json.JSONDecodeError, IOError):
                pass

    # Deduplicate by critic name
    seen = set()
    unique = []
    for r in reviews:
        if r["critic_name"] not in seen:
            seen.add(r["critic_name"])
            unique.append(r)
    reviews = unique

    if reviews:
        with open(cache_file, "w") as f:
            json.dump(reviews, f, indent=2)

    return reviews


def _fetch_from_napi(ems_id, max_pages):
    """Single attempt to pull reviews from RT's NAPI. Returns parsed list
    (possibly empty). All caching/retry logic lives in scrape_reviews."""
    url = REVIEW_API.format(ems_id=ems_id)
    out = []
    cursor = None
    for _ in range(max_pages):
        params = {"pageCount": 20, "topOnly": "false", "type": "critic"}
        if cursor:
            params["after"] = cursor
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
        except (requests.RequestException, ValueError):
            break
        batch = data.get("reviews", [])
        if not batch:
            break
        for raw in batch:
            r = _parse_review(raw)
            if r:
                out.append(r)
        page_info = data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            break
        time.sleep(0.5)
    return out


def _parse_review(raw):
    """Convert a raw API review object into our standard format."""
    critic = raw.get("critic", {})
    name = critic.get("displayName", "").strip()
    if not name:
        return None

    sentiment_raw = raw.get("scoreSentiment", "")
    sentiment = "Fresh" if sentiment_raw == "POSITIVE" else "Rotten"

    publication = raw.get("publication", {})
    pub_name = publication.get("name", "")

    quote = raw.get("reviewQuote", "")
    if quote:
        quote = html_lib.unescape(quote)

    return {
        "critic_name": name,
        "sentiment": sentiment,
        "publication": pub_name,
        "top_critic": critic.get("isTopCritic", False),
        "rating_text": raw.get("originalScore", ""),
        "quote": quote,
    }
