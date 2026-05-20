import config


def compute_sizing_multiplier(confidence, known_pct=0, settlement_date=None):
    """Compute the combined sizing multiplier from confidence and time-to-settlement.

    Returns a float multiplier applied to the base Kelly fraction.
    """
    from engine.alpha import _time_to_settlement_multiplier

    conf_mult = config.CONFIDENCE_KELLY_MULTIPLIER.get(confidence, 0.3)
    time_mult = _time_to_settlement_multiplier(settlement_date)

    return round(conf_mult * time_mult, 3)


def format_sizing(opportunity, bankroll=1000, known_pct=0, settlement_date=None):
    """Format position sizing recommendation for display."""
    size = opportunity["suggested_size"]
    kelly = opportunity["kelly_pct"]
    direction = opportunity["direction"]
    threshold = opportunity["threshold"]
    edge = opportunity["edge"]
    confidence = opportunity["confidence"]

    mult = compute_sizing_multiplier(confidence, known_pct, settlement_date)
    size *= mult

    return {
        "direction": direction,
        "threshold": threshold,
        "size": round(size, 2),
        "kelly_pct": kelly,
        "edge": edge,
        "sizing_mult": mult,
        "note": f"{mult:.2f}x (confidence={confidence})",
    }
