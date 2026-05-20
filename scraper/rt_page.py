import json
import re

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def get_movie_summary(slug):
    """
    Fetch tomatometer, review count, and title from the RT main page.
    slug: e.g. "m/mission_impossible_the_final_reckoning"
    Returns dict or None.
    """
    url = f"https://www.rottentomatoes.com/{slug}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    result = {"slug": slug}
    soup = BeautifulSoup(resp.text, "lxml")

    # Extract from JSON-LD
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict):
                if "name" in data:
                    result["title"] = data["name"]
                if "aggregateRating" in data:
                    rating = data["aggregateRating"]
                    result["tomatometer"] = int(float(rating.get("ratingValue", 0)))
                    result["review_count"] = int(rating.get("ratingCount", 0))
                if "genre" in data:
                    genres = data["genre"]
                    if isinstance(genres, str):
                        genres = [genres]
                    result["genres"] = genres
                if "director" in data:
                    directors = data["director"]
                    if isinstance(directors, dict):
                        directors = [directors]
                    result["directors"] = [d.get("name", "") for d in directors if isinstance(d, dict)]
                if "contentRating" in data:
                    result["content_rating"] = data["contentRating"]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Fallback title from og:title meta tag
    if "title" not in result:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            result["title"] = og["content"].split(" - Rotten")[0].strip()

    # Get EMS ID for potential API use
    ems_match = re.search(r'"emsId"\s*:\s*"([^"]+)"', resp.text)
    if ems_match:
        result["ems_id"] = ems_match.group(1)

    # Extract full release date
    date_match = re.search(r'In Theaters\s+(\w+\s+\d+,?\s*\d{4})', resp.text)
    if not date_match:
        date_match = re.search(r'"releaseDate"\s*:\s*"([^"]+)"', resp.text)
    if date_match:
        result["release_date"] = date_match.group(1).strip()

    # Extract release year from page
    year_match = re.search(r'In Theaters\s+\w+\s+\d+,?\s*(\d{4})', resp.text)
    if not year_match:
        year_match = re.search(r'Release Date.*?(\d{4})', resp.text, re.DOTALL)
    if not year_match:
        year_match = re.search(r'"releaseDate"\s*:\s*"(\d{4})', resp.text)
    if not year_match:
        year_match = re.search(r'releaseDateStr.*?(\d{4})', resp.text)
    if year_match:
        result["year"] = int(year_match.group(1))

    # Check for "Coming Soon" / no score yet
    if "Coming Soon" in resp.text and "tomatometer" not in result:
        result["coming_soon"] = True

    return result if "title" in result or "tomatometer" in result or result.get("coming_soon") else None
