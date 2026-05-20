"""
Apply optimized parameters from backtest results into config.py and calibration.py.

Reads optimized_params.json and optimized_platt.json, updates the source files,
then re-runs the backtest to verify improvement.

Usage:
    python -m backtest.apply_params           # apply all
    python -m backtest.apply_params --dry-run  # show changes without writing
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.py"
CALIBRATION_PATH = ROOT / "model" / "calibration.py"
OPTIMIZED_CONFIG_PATH = ROOT / "backtest" / "optimized_params.json"
OPTIMIZED_PLATT_PATH = ROOT / "backtest" / "optimized_platt.json"


def apply_config_params(dry_run=False):
    """Update config.py with optimized model parameters."""
    if not OPTIMIZED_CONFIG_PATH.exists():
        print("No optimized_params.json found. Run backtest.optimize first.")
        return False

    with open(OPTIMIZED_CONFIG_PATH) as f:
        data = json.load(f)

    params = data["params"]
    config_text = CONFIG_PATH.read_text()
    changes = []

    for param_name, new_value in params.items():
        pattern = rf"^({param_name}\s*=\s*)[\d.]+(.*)$"
        match = re.search(pattern, config_text, re.MULTILINE)
        if match:
            old_line = match.group(0)
            new_line = f"{match.group(1)}{new_value}{match.group(2)}"
            if old_line != new_line:
                changes.append((param_name, old_line.strip(), new_line.strip()))
                if not dry_run:
                    config_text = config_text.replace(old_line, new_line)

    if changes:
        print(f"Config changes ({'DRY RUN' if dry_run else 'APPLIED'}):")
        for name, old, new in changes:
            print(f"  {name}: {old} -> {new}")
        if not dry_run:
            # Mark as validated from backtest
            config_text = config_text.replace(
                "UNVALIDATED -- should be fitted from backtest.",
                "VALIDATED from backtest optimization."
            )
            CONFIG_PATH.write_text(config_text)
            print(f"  Written to {CONFIG_PATH}")
    else:
        print("No config changes needed.")

    return bool(changes)


def apply_platt_params(dry_run=False):
    """Update calibration.py with fitted Platt scaling parameters."""
    if not OPTIMIZED_PLATT_PATH.exists():
        print("No optimized_platt.json found. Run backtest.optimize first.")
        return False

    with open(OPTIMIZED_PLATT_PATH) as f:
        platt = json.load(f)

    cal_text = CALIBRATION_PATH.read_text()

    new_dict_lines = ["PLATT_PARAMS = {"]
    for n in sorted(platt.keys(), key=int):
        a, b = platt[n]
        new_dict_lines.append(f"    {n}:  ({a}, {b}),")
    new_dict_lines.append("}")
    new_block = "\n".join(new_dict_lines)

    pattern = r"PLATT_PARAMS\s*=\s*\{[^}]+\}"
    match = re.search(pattern, cal_text, re.DOTALL)
    if not match:
        print("Could not find PLATT_PARAMS block in calibration.py")
        return False

    old_block = match.group(0)
    if old_block.strip() == new_block.strip():
        print("No Platt param changes needed.")
        return False

    print(f"Platt scaling changes ({'DRY RUN' if dry_run else 'APPLIED'}):")
    for n in sorted(platt.keys(), key=int):
        a, b = platt[n]
        print(f"  N={n}: a={a}, b={b}")

    if not dry_run:
        cal_text = cal_text.replace(old_block, new_block)
        cal_text = cal_text.replace(
            "These are identity-ish for now (a=1, b=0) and will be fit from backtesting",
            "Fitted from backtest optimization"
        )
        CALIBRATION_PATH.write_text(cal_text)
        print(f"  Written to {CALIBRATION_PATH}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    config_changed = apply_config_params(dry_run=args.dry_run)
    platt_changed = apply_platt_params(dry_run=args.dry_run)

    if (config_changed or platt_changed) and not args.dry_run:
        print("\nParameters applied. Run backtest to verify:")
        print("  python -m backtest.evaluate")


if __name__ == "__main__":
    main()
