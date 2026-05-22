"""
Scrape individual RT critic reviews via the internal NAPI.
No Playwright needed -- plain HTTP requests to the JSON API.
"""
import html as html_lib
import json
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


def scrape_reviews(ems_id, slug="", max_pages=25):
    """
    Fetch critic reviews from the RT internal API.

    ems_id: the EMS ID from the movie's main page (get via get_movie_summary)
    slug: used for cache file naming only
    max_pages: max pagination requests (20 reviews per page). Default 25
        covers up to 500 reviews -- enough for any wide-release film.
        Loop exits early when RT signals hasNextPage=false.

    Returns list of review dicts.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = slug.replace("/", "_") or ems_id
    cache_file = CACHE_DIR / f"{cache_key}_{date.today().isoformat()}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
            if cached:
                return cached

    url = REVIEW_API.format(ems_id=ems_id)
    reviews = []
    cursor = None

    for page_num in range(max_pages):
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
            review = _parse_review(raw)
            if review:
                reviews.append(review)

        page_info = data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            break

        time.sleep(0.5)

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
