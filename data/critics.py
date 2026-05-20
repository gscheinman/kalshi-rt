import csv
import difflib
import json
from pathlib import Path

import config


class CriticDatabase:
    def __init__(self, csv_path=None, reviews_path=None):
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "critic_database.csv"
        self._lookup = {}
        self._names = []
        self._calibration = {}
        all_agreements = []
        top_agreements = []

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                name = row["critic_name"]
                is_top = row["top_critic"] == "True"
                agreement = float(row["agreement_rate"])
                big_agreement = row.get("big_movie_agreement_rate", "")
                entry = {
                    "critic_name": name,
                    "top_critic": is_top,
                    "publisher_name": row["publisher_name"],
                    "total_reviews": int(row["total_reviews"]),
                    "agreement_rate": agreement,
                    "big_movie_agreement_rate": float(big_agreement) if big_agreement else None,
                    "fresh_rate": float(row["fresh_rate"]),
                    "tier": row["tier"],
                }
                self._lookup[name] = entry
                self._lookup[name.lower()] = entry
                self._names.append(name)
                all_agreements.append(agreement)
                if is_top:
                    top_agreements.append(agreement)

        all_agreements.sort()
        top_agreements.sort()
        self.median_agreement = all_agreements[len(all_agreements) // 2]
        self.top_critic_median = (
            top_agreements[len(top_agreements) // 2] if top_agreements else self.median_agreement
        )
        self.count = len(self._names)

        if reviews_path is None:
            reviews_path = Path(__file__).parent.parent / "critic_reviews.json"
        if reviews_path.exists():
            self._load_calibration(reviews_path)

    def _load_calibration(self, path):
        """Load per-critic calibration data (score curves, avg tomatometer by sentiment).

        Prefers big-movie calibration data when available, since Kalshi markets
        are always major releases.
        """
        with open(path) as f:
            data = json.load(f)
        for critic_name, cdata in data.items():
            cal = {}
            # Prefer big-movie averages; fall back to overall
            cal["avg_fresh"] = (
                cdata.get("big_avg_tomato_when_fresh")
                or cdata.get("avg_tomato_when_fresh")
            )
            cal["avg_rotten"] = (
                cdata.get("big_avg_tomato_when_rotten")
                or cdata.get("avg_tomato_when_rotten")
            )
            if cdata.get("score_calibration"):
                cal["score_curve"] = cdata["score_calibration"]
            # Only store if we have at least one useful field
            if cal.get("avg_fresh") or cal.get("avg_rotten") or cal.get("score_curve"):
                self._calibration[critic_name] = cal
                self._calibration[critic_name.lower()] = cal

    def get_critic(self, name):
        exact = self._lookup.get(name) or self._lookup.get(name.lower())
        if exact:
            return exact
        match = self._fuzzy_match(name)
        if match:
            return self._lookup[match]
        return None

    def get_weight(self, name, is_top_critic=False):
        """Get critic weight, preferring big-movie agreement rate.

        Kalshi only lists major releases, so a critic's accuracy on big movies
        (80+ reviews) is more predictive than their overall agreement rate.
        Falls back to overall rate if no big-movie data exists.
        """
        critic = self.get_critic(name)
        if critic:
            big = critic.get("big_movie_agreement_rate")
            if big is not None:
                return big
            return critic["agreement_rate"]
        return self.top_critic_median if is_top_critic else self.median_agreement

    def get_calibration(self, name):
        """Get per-critic calibration data if available.

        Returns dict with optional keys:
            avg_fresh: avg tomatometer of movies this critic rated Fresh
            avg_rotten: avg tomatometer of movies this critic rated Rotten
            score_curve: dict mapping score buckets to avg tomatometer
        Returns None if no calibration data exists.
        """
        cal = self._calibration.get(name) or self._calibration.get(name.lower())
        if cal:
            return cal
        match = self._fuzzy_match(name)
        if match:
            return self._calibration.get(match)
        return None

    def get_implied_tomatometer(self, name, sentiment, rating_text=None):
        """Get this critic's implied tomatometer prediction for a review.

        Uses the critic's historical calibration: when this critic gives a Fresh
        review, what's the avg tomatometer of those movies? If they gave a numeric
        score, use the score calibration curve for finer granularity.

        Returns a 0-100 float, or None if no calibration data.
        """
        cal = self.get_calibration(name)
        if not cal:
            return None

        if rating_text and cal.get("score_curve"):
            from model.sentiment import parse_numeric_rating
            numeric = parse_numeric_rating(rating_text)
            if numeric is not None:
                bucket = self._score_bucket(numeric)
                if bucket in cal["score_curve"]:
                    return cal["score_curve"][bucket]

        if sentiment == "Fresh" and "avg_fresh" in cal:
            return cal["avg_fresh"]
        if sentiment != "Fresh" and "avg_rotten" in cal:
            return cal["avg_rotten"]
        return None

    @staticmethod
    def _score_bucket(numeric_0_1):
        """Map a 0-1 numeric rating to a score bucket string like '70-80'."""
        pct = int(numeric_0_1 * 100)
        bucket_low = (pct // 10) * 10
        bucket_high = bucket_low + 10
        return f"{bucket_low}-{bucket_high}"

    def _fuzzy_match(self, name, threshold=None):
        if threshold is None:
            threshold = config.FUZZY_MATCH_THRESHOLD
        name_lower = name.lower()
        best_score = 0
        best_match = None
        for known in self._names:
            score = difflib.SequenceMatcher(None, name_lower, known.lower()).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = known
        return best_match
