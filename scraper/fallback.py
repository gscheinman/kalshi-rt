import json


def load_reviews_file(path):
    """Load reviews from a JSON file."""
    with open(path) as f:
        return json.load(f)


def manual_entry():
    """Interactive mode for entering reviews manually."""
    print("\nManual Review Entry")
    print("Format: critic_name, Fresh/Rotten [, publication]")
    print("Type 'done' when finished.\n")

    reviews = []
    while True:
        try:
            line = input(f"  Review #{len(reviews) + 1} > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if line.lower() in ("done", "quit", "exit", "q", ""):
            break

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            print("    Format: critic_name, Fresh/Rotten [, publication]")
            continue

        sentiment = "Fresh" if parts[1].lower().startswith("f") else "Rotten"
        reviews.append({
            "critic_name": parts[0],
            "sentiment": sentiment,
            "publication": parts[2] if len(parts) > 2 else "",
            "top_critic": False,
        })
        print(f"    Added: {parts[0]} -> {sentiment}")

    return reviews
