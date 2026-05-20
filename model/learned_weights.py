"""
ML-based critic weight and score prediction model.

Architecture:
  Layer 1: LightGBM predicts the final tomatometer from individual review features.
           Each review becomes a training example: (critic features, review features) -> movie score.
  Layer 2: At inference, each review's ML-predicted score replaces the crude
           "agreement_rate * intensity" signal. These predictions feed into the
           Beta-binomial framework for uncertainty quantification.

Why this beats hand-crafted weights:
  - Learns non-linear relationships (critic X is predictive above 70% but not below)
  - Captures interactions (certain critic combinations are more informative)
  - Automatically discovers which features matter most
  - Adapts to the actual distribution of Kalshi-relevant movies

Why we keep the Beta-binomial:
  - Correctly handles uncertainty with small review counts
  - Naturally shrinks toward prior with sparse data
  - Outputs calibrated probabilities at every threshold
  - Doesn't need thousands of Kalshi-specific examples to learn these properties
"""
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

import config

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "model" / "critic_weight_model.pkl"
FEATURE_STATS_PATH = ROOT / "model" / "feature_stats.json"


def build_training_data(critic_cal, movie_scores):
    """Build (features, target) pairs from historical critic review data.

    Each training example is one review of one movie:
      features = critic stats + review characteristics + movie context
      target = final tomatometer of the movie (0-100)

    We train the model to predict the final score from a single review's
    features. At inference, we average the predictions across all reviews
    of a new movie to get a score estimate.
    """
    # Pre-compute per-movie review counts
    movie_review_counts = defaultdict(int)
    for critic_name, cdata in critic_cal.items():
        for r in cdata.get("reviews", []):
            ems = r.get("movie_ems_id")
            if ems:
                movie_review_counts[ems] += 1

    features = []
    targets = []
    movie_ids = []

    for critic_name, cdata in critic_cal.items():
        reviews = cdata.get("reviews", [])
        if len(reviews) < 5:
            continue

        critic_features = _extract_critic_features(cdata, movie_review_counts, movie_scores)

        for r in reviews:
            ems = r.get("movie_ems_id")
            if not ems or ems not in movie_scores:
                continue

            score_info = movie_scores[ems]
            tomato = score_info.get("tomatometer")
            if tomato is None:
                continue

            review_features = _extract_review_features(r)
            context_features = _extract_context_features(ems, movie_review_counts)

            row = {**critic_features, **review_features, **context_features}
            features.append(row)
            targets.append(tomato)
            movie_ids.append(ems)

    return features, targets, movie_ids


def _extract_critic_features(cdata, movie_review_counts, movie_scores):
    """Features about the critic (static per critic)."""
    agreement = cdata.get("agreement_rate") or 0.5
    big_agreement = cdata.get("big_movie_agreement_rate")
    avg_fresh = cdata.get("avg_tomato_when_fresh")
    avg_rotten = cdata.get("avg_tomato_when_rotten")
    big_avg_fresh = cdata.get("big_avg_tomato_when_fresh")
    big_avg_rotten = cdata.get("big_avg_tomato_when_rotten")
    fresh_rate = cdata.get("fresh_rate", 0.5)
    n_reviews = len(cdata.get("reviews", []))

    spread = abs(avg_fresh - avg_rotten) if (avg_fresh and avg_rotten) else 0
    big_spread = abs(big_avg_fresh - big_avg_rotten) if (big_avg_fresh and big_avg_rotten) else spread

    # How many big movies has this critic reviewed?
    big_count = 0
    for r in cdata.get("reviews", []):
        ems = r.get("movie_ems_id")
        if ems and movie_review_counts.get(ems, 0) >= config.BIG_MOVIE_REVIEW_THRESHOLD:
            big_count += 1

    return {
        "critic_agreement": agreement,
        "critic_big_agreement": big_agreement if big_agreement is not None else agreement,
        "critic_fresh_rate": fresh_rate,
        "critic_n_reviews": min(n_reviews, 500),
        "critic_avg_fresh": (avg_fresh or 65) / 100.0,
        "critic_avg_rotten": (avg_rotten or 40) / 100.0,
        "critic_big_avg_fresh": (big_avg_fresh or avg_fresh or 65) / 100.0,
        "critic_big_avg_rotten": (big_avg_rotten or avg_rotten or 40) / 100.0,
        "critic_spread": spread / 100.0,
        "critic_big_spread": big_spread / 100.0,
        "critic_big_movie_count": big_count,
        "critic_selectivity": 1.0 - fresh_rate,
    }


def _extract_review_features(review):
    """Features about this specific review."""
    is_fresh = 1.0 if review.get("sentiment") == "POSITIVE" else 0.0

    # Parse numeric score if available
    from model.sentiment import parse_numeric_rating, score_intensity
    score_text = review.get("original_score", "")
    numeric = parse_numeric_rating(score_text)

    dummy_review = {
        "sentiment": "Fresh" if is_fresh else "Rotten",
        "rating_text": score_text,
        "quote": "",
    }
    intensity = score_intensity(dummy_review)

    return {
        "review_is_fresh": is_fresh,
        "review_numeric_score": numeric if numeric is not None else -1.0,
        "review_has_numeric": 1.0 if numeric is not None else 0.0,
        "review_intensity": intensity,
    }


def _extract_context_features(ems_id, movie_review_counts):
    """Features about the movie's review context."""
    total_reviews = movie_review_counts.get(ems_id, 0)
    return {
        "movie_total_reviews": min(total_reviews, 500),
        "movie_is_big": 1.0 if total_reviews >= config.BIG_MOVIE_REVIEW_THRESHOLD else 0.0,
    }


def train_model(critic_cal, movie_scores):
    """Train the LightGBM model on historical review data.

    Uses leave-one-movie-out cross-validation to estimate generalization,
    then trains final model on all data.
    """
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    print("Building training data...", flush=True)
    features, targets, movie_ids = build_training_data(critic_cal, movie_scores)

    if len(features) < 100:
        print(f"Only {len(features)} training examples. Need more data.")
        return None

    feature_names = sorted(features[0].keys())
    X = np.array([[row[f] for f in feature_names] for row in features])
    y = np.array(targets, dtype=np.float64)
    groups = np.array(movie_ids)

    print(f"Training data: {len(X)} reviews across {len(set(movie_ids))} movies", flush=True)
    print(f"Features: {feature_names}", flush=True)

    # Cross-validate with movie-level splits (no data leakage)
    gkf = GroupKFold(n_splits=5)
    cv_errors = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
        }

        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        preds = model.predict(X_val)
        fold_mae = np.mean(np.abs(preds - y_val))
        cv_errors.append(fold_mae)
        print(f"  Fold {fold+1}: MAE = {fold_mae:.2f}%", flush=True)

    mean_cv = np.mean(cv_errors)
    print(f"\nCross-validated MAE: {mean_cv:.2f}% (+/- {np.std(cv_errors):.2f}%)", flush=True)

    # Also compute naive baseline (just use fresh_rate as score prediction)
    naive_errors = []
    for feat, actual in zip(features, targets):
        naive_pred = feat["review_is_fresh"] * feat["critic_avg_fresh"] * 100 + \
                     (1 - feat["review_is_fresh"]) * feat["critic_avg_rotten"] * 100
        naive_errors.append(abs(naive_pred - actual))
    naive_mae = np.mean(naive_errors)
    print(f"Naive baseline MAE: {naive_mae:.2f}%", flush=True)
    print(f"ML improvement: {naive_mae - mean_cv:.2f}% ({(naive_mae - mean_cv)/naive_mae*100:.1f}% better)", flush=True)

    # Train final model on all data
    print("\nTraining final model on all data...", flush=True)
    full_data = lgb.Dataset(X, label=y, feature_name=feature_names)
    final_model = lgb.train(
        params, full_data,
        num_boost_round=500,
    )

    # Feature importance
    importance = dict(zip(feature_names, final_model.feature_importance(importance_type="gain")))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature importance (gain):")
    total_imp = sum(v for _, v in sorted_imp)
    for feat, imp in sorted_imp:
        pct = imp / total_imp * 100 if total_imp > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {feat:30s} {pct:5.1f}% {bar}")

    # Save model
    model_data = {
        "model": final_model,
        "feature_names": feature_names,
        "cv_mae": mean_cv,
        "naive_mae": naive_mae,
        "n_training": len(X),
        "n_movies": len(set(movie_ids)),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nModel saved to {MODEL_PATH}")

    # Save feature stats for inference fallbacks
    stats = {}
    for feat in feature_names:
        col = [row[feat] for row in features]
        stats[feat] = {"median": float(np.median(col)), "mean": float(np.mean(col))}
    with open(FEATURE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    return final_model, feature_names, mean_cv


class LearnedPredictor:
    """Wraps the trained LightGBM model for use at inference time.

    Given a set of reviews and critic data, predicts what each review
    implies about the final tomatometer, then aggregates into a
    Beta-binomial posterior.
    """

    def __init__(self):
        self._model = None
        self._feature_names = None
        self._stats = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return self._model is not None

        self._loaded = True
        if not MODEL_PATH.exists():
            return False

        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._feature_names = data["feature_names"]

        if FEATURE_STATS_PATH.exists():
            with open(FEATURE_STATS_PATH) as f:
                self._stats = json.load(f)

        return True

    @property
    def available(self):
        return self._load()

    def predict_review_scores(self, reviews, critic_db):
        """For each review, predict what it implies about the final tomatometer.

        Returns list of (predicted_score_0_to_1, confidence_weight) tuples.
        Higher confidence_weight means the model is more certain about this
        review's predictive value.
        """
        if not self._load():
            return None

        rows = []
        for r in reviews:
            name = r["critic_name"]
            critic = critic_db.get_critic(name)
            cal = critic_db.get_calibration(name)

            critic_features = self._critic_features(critic, cal, name, critic_db)
            review_features = _extract_review_features({
                "sentiment": "POSITIVE" if r["sentiment"] == "Fresh" else "NEGATIVE",
                "original_score": r.get("rating_text", ""),
            })
            context_features = {
                "movie_total_reviews": len(reviews),
                "movie_is_big": 1.0,
            }

            row = {**critic_features, **review_features, **context_features}
            # Fill any missing features with median from training
            for feat in self._feature_names:
                if feat not in row:
                    row[feat] = self._stats[feat]["median"] if self._stats and feat in self._stats else 0.0

            rows.append([row[f] for f in self._feature_names])

        X = np.array(rows)
        predictions = self._model.predict(X)

        results = []
        for i, pred in enumerate(predictions):
            score = max(0, min(100, pred)) / 100.0

            # Confidence based on whether we know this critic
            critic = critic_db.get_critic(reviews[i]["critic_name"])
            confidence = 1.0 if critic else 0.5

            results.append((score, confidence))

        return results

    def _critic_features(self, critic, cal, name, critic_db):
        """Build critic feature dict, with fallbacks for unknown critics."""
        if critic:
            agreement = critic["agreement_rate"]
            big_agreement = critic.get("big_movie_agreement_rate") or agreement
            fresh_rate = critic["fresh_rate"]
            n_reviews = critic["total_reviews"]
        else:
            agreement = critic_db.median_agreement
            big_agreement = agreement
            fresh_rate = 0.5
            n_reviews = 0

        if cal:
            avg_fresh = (cal.get("avg_fresh") or 65) / 100.0
            avg_rotten = (cal.get("avg_rotten") or 40) / 100.0
        else:
            avg_fresh = 0.65
            avg_rotten = 0.40

        spread = abs(avg_fresh - avg_rotten)

        return {
            "critic_agreement": agreement,
            "critic_big_agreement": big_agreement,
            "critic_fresh_rate": fresh_rate,
            "critic_n_reviews": min(n_reviews, 500),
            "critic_avg_fresh": avg_fresh,
            "critic_avg_rotten": avg_rotten,
            "critic_big_avg_fresh": avg_fresh,
            "critic_big_avg_rotten": avg_rotten,
            "critic_spread": spread,
            "critic_big_spread": spread,
            "critic_big_movie_count": 0,
            "critic_selectivity": 1.0 - fresh_rate,
        }


# Singleton
learned_predictor = LearnedPredictor()
