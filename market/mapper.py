import json
import re
from datetime import date
from pathlib import Path

import requests

CACHE_DIR = Path.home() / ".cache" / "kalshi-rt"
MAPPINGS_FILE = CACHE_DIR / "mappings.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
CURRENT_YEAR = date.today().year


class TickerMapper:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._mappings = self._load_mappings()

    def get_rt_slug(self, event):
        """Given a Kalshi event dict, return the RT movie slug (e.g. 'm/scary_movie')."""
        ticker = event["event_ticker"]
        if ticker in self._mappings:
            return self._mappings[ticker]

        movie_name = event.get("movie_name", "")
        if not movie_name:
            return None

        slug = self._search_rt(movie_name)
        if slug:
            self._mappings[ticker] = slug
            self._save_mappings()
        return slug

    def set_mapping(self, event_ticker, rt_slug):
        """Manually set a ticker -> slug mapping."""
        self._mappings[event_ticker] = rt_slug
        self._save_mappings()

    def clear_mapping(self, event_ticker):
        """Remove a cached mapping (useful when it pointed to the wrong movie)."""
        self._mappings.pop(event_ticker, None)
        self._save_mappings()

    def _search_rt(self, movie_name):
        """Try to find the RT slug for a movie name, preferring recent/upcoming movies."""
        name_lower = movie_name.lower().replace("-", " ").replace(":", "").replace("'", "").replace("’", "")
        slug_base = re.sub(r'[^a-z0-9\s]', '', name_lower)
        slug_base = re.sub(r'\s+', '_', slug_base.strip())

        # Try year-suffixed slugs first (2026, 2025), then bare slug
        candidates = [
            f"m/{slug_base}_{CURRENT_YEAR}",
            f"m/{slug_base}_{CURRENT_YEAR - 1}",
            f"m/{slug_base}_{CURRENT_YEAR + 1}",
            f"m/{slug_base}",
        ]

        for slug in candidates:
            result = self._check_slug(slug)
            if result == "recent":
                return slug
            if result == "exists_but_old":
                continue

        # Try RT search API for better results
        slug = self._search_rt_api(movie_name)
        if slug:
            return slug

        return None

    def _check_slug(self, slug):
        """Check if a slug exists and whether it's a recent movie.
        Returns: 'recent', 'exists_but_old', or None."""
        try:
            resp = requests.get(
                f"https://www.rottentomatoes.com/{slug}",
                headers=HEADERS, timeout=10, allow_redirects=True,
            )
            if resp.status_code != 200:
                return None

            # Check release year
            year_match = re.search(r'In Theaters\s+\w+\s+\d+,?\s*(\d{4})', resp.text)
            if not year_match:
                year_match = re.search(r'"releaseDate"\s*:\s*"(\d{4})', resp.text)
            if not year_match:
                year_match = re.search(r'releaseDateStr.*?(\d{4})', resp.text)

            if year_match:
                year = int(year_match.group(1))
                if year >= CURRENT_YEAR - 1:
                    return "recent"
                return "exists_but_old"

            if "Coming Soon" in resp.text:
                return "recent"

            return "recent"
        except requests.RequestException:
            return None

    def _search_rt_api(self, movie_name):
        """Search RT and pick the most recent matching movie."""
        try:
            resp = requests.get(
                "https://www.rottentomatoes.com/search",
                params={"search": movie_name},
                headers=HEADERS, timeout=10,
            )
            if resp.status_code != 200:
                return None

            slugs = re.findall(r'href="(/m/[^"]+)"', resp.text)
            seen = set()
            for raw in slugs:
                slug = raw.lstrip("/")
                if slug in seen:
                    continue
                seen.add(slug)
                result = self._check_slug(slug)
                if result == "recent":
                    return slug
        except requests.RequestException:
            pass
        return None

    def _load_mappings(self):
        if MAPPINGS_FILE.exists():
            with open(MAPPINGS_FILE) as f:
                return json.load(f)
        return {}

    def _save_mappings(self):
        with open(MAPPINGS_FILE, "w") as f:
            json.dump(self._mappings, f, indent=2)
