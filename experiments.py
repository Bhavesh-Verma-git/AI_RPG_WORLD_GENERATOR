"""
experiments.py — Parameter Experiment Runner
=============================================
Runs controlled generation experiments by varying:
  • temperature (creativity vs structure)
  • max_length (output completeness)
  • theme (different world types)

Each experiment is saved to outputs/experiments/ for comparison.
Run this after checking generate.py works correctly.

Usage:
    python experiments.py
"""

import os
import time
from generate import generate_world

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT DEFINITIONS
# Each experiment is a dict of parameters to pass to generate_world().
# We use a fixed seed=1 so that parameter effects (not randomness) explain differences.
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # ── Experiment 1: Low temperature — structured, deterministic output
    {
        "id": "exp1_low_temperature",
        "description": "Low temperature (0.3) — tests structured, deterministic generation",
        "params": {"theme": "fantasy", "temperature": 0.3, "max_length": 200, "seed": 1},
    },
    # ── Experiment 2: Medium temperature — balanced creativity
    {
        "id": "exp2_medium_temperature",
        "description": "Medium temperature (0.8) — balanced creativity and structure",
        "params": {"theme": "fantasy", "temperature": 0.8, "max_length": 200, "seed": 1},
    },
    # ── Experiment 3: High temperature — more creative, less predictable
    {
        "id": "exp3_high_temperature",
        "description": "High temperature (1.2) — high creativity, may lose structure",
        "params": {"theme": "fantasy", "temperature": 1.2, "max_length": 200, "seed": 1},
    },
    # ── Experiment 4: Short output — quick, truncated world
    {
        "id": "exp4_short_output",
        "description": "Short max_length (80) — effect of limiting token budget",
        "params": {"theme": "fantasy", "temperature": 0.8, "max_length": 80, "seed": 1},
    },
    # ── Experiment 5: Long output — richer world detail
    {
        "id": "exp5_long_output",
        "description": "Long max_length (400) — richer, more detailed world generation",
        "params": {"theme": "fantasy", "temperature": 0.8, "max_length": 400, "seed": 1},
    },
    # ── Experiment 6: Dark fantasy theme
    {
        "id": "exp6_dark_fantasy",
        "description": "Dark fantasy theme at medium temperature",
        "params": {"theme": "dark_fantasy", "temperature": 0.8, "max_length": 250, "seed": 1},
    },
    # ── Experiment 7: Desert theme
    {
        "id": "exp7_desert",
        "description": "Desert theme at medium temperature",
        "params": {"theme": "desert", "temperature": 0.8, "max_length": 250, "seed": 1},
    },
    # ── Experiment 8: Arctic theme
    {
        "id": "exp8_arctic",
        "description": "Arctic theme at medium temperature",
        "params": {"theme": "arctic", "temperature": 0.8, "max_length": 250, "seed": 1},
    },
    # ── Experiment 9: Sci-fi theme
    {
        "id": "exp9_sci_fi",
        "description": "Sci-fi theme at medium temperature",
        "params": {"theme": "sci_fi", "temperature": 0.8, "max_length": 250, "seed": 1},
    },
    # ── Experiment 10: Different seeds — shows variability
    {
        "id": "exp10_seed_variation",
        "description": "Fantasy, same parameters, different seed — shows output variability",
        "params": {"theme": "fantasy", "temperature": 0.8, "max_length": 200, "seed": 99},
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(output_dir: str = "outputs/experiments"):
    """
    Iterate over EXPERIMENTS, generate worlds, and save results.
    Prints a summary table at the end.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  🔬  AI RPG WORLD GENERATOR — EXPERIMENT RUNNER")
    print("=" * 65)
    print(f"  Running {len(EXPERIMENTS)} experiments...\n")

    summary_rows = []

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        desc = exp["description"]
        params = exp["params"]

        print(f"▶  {exp_id}")
        print(f"   {desc}")
        print(f"   Params: {params}")

        start = time.time()
        try:
            world_text = generate_world(**params)
            elapsed = time.time() - start
            status = "✅ OK"
        except Exception as e:
            world_text = f"[ERROR] {e}"
            elapsed = time.time() - start
            status = "❌ FAILED"

        # Save result
        out_path = os.path.join(output_dir, f"{exp_id}.txt")
        header = (
            f"EXPERIMENT: {exp_id}\n"
            f"DESCRIPTION: {desc}\n"
            f"PARAMETERS: {params}\n"
            f"TIME: {elapsed:.1f}s\n"
            f"{'=' * 60}\n\n"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header + world_text)

        print(f"   {status} — saved to {out_path} ({elapsed:.1f}s)\n")
        summary_rows.append((exp_id, params.get("temperature"), params.get("max_length"), params.get("theme"), status))

    # Print summary table
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'ID':<30} {'Temp':>5} {'Len':>5} {'Theme':<12} {'Status'}")
    print(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*12} {'-'*8}")
    for row in summary_rows:
        print(f"  {row[0]:<30} {str(row[1]):>5} {str(row[2]):>5} {str(row[3]):<12} {row[4]}")
    print(f"\n  All results saved to: {output_dir}/\n")


if __name__ == "__main__":
    run_experiments()
