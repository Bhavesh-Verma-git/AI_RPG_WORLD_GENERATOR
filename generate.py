"""
generate.py — AI RPG World Generator
======================================
Generates procedural RPG world scenarios using DistilGPT2 from HuggingFace.
All inference runs on CPU — no GPU required.

APPROACH: Field-by-field generation.
Instead of asking the model to generate all 6 fields at once (which is hard
for a small model on CPU), we generate each field individually using a
targeted prompt. This ensures all fields are always populated and readable.

Usage:
    python generate.py --theme fantasy
    python generate.py --theme dark_fantasy --temperature 0.9 --max_length 200
    python generate.py --theme desert --seed 42
"""

import argparse
import os
import random
import time

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# THEME CONFIGURATIONS
# Each theme defines context phrases that prime the model for that world flavour.
# ─────────────────────────────────────────────────────────────────────────────

THEME_CONFIGS = {
    "fantasy": {
        "adjectives": "high-fantasy, magical, ancient, elven, mystical",
        "setting": "a world of ancient elven forests, glowing rivers, and arcane towers",
        "name_example": "Eldenmoor",
        "npc_prefix": "a mysterious mage or mystical warrior",
        "quest_type": "ancient artefact or magical ritual",
        "weapon_type": "enchanted blade or arcane staff",
        "lore_hook": "a long-forgotten kingdom of elves and mages",
    },
    "dark_fantasy": {
        "adjectives": "dark, cursed, grim, gothic, haunted",
        "setting": "a cursed land of red skies, undead hordes, and fallen heroes",
        "name_example": "Ashenveil",
        "npc_prefix": "a fallen paladin or cursed necromancer",
        "quest_type": "dark ritual or demonic threat",
        "weapon_type": "cursed axe or soul-bound blade",
        "lore_hook": "a kingdom destroyed by forbidden dark magic",
    },
    "desert": {
        "adjectives": "sun-scorched, ancient, nomadic, arid, sandy",
        "setting": "a vast desert of endless dunes and buried civilisations",
        "name_example": "Sandscorch Expanse",
        "npc_prefix": "a nomadic mercenary or desert oracle",
        "quest_type": "lost oasis or ancient desert tomb",
        "weapon_type": "flaming bow or sandstorm spear",
        "lore_hook": "a great ancient civilisation swallowed by the sands",
    },
    "arctic": {
        "adjectives": "frozen, icy, blizzard-swept, ancient, isolated",
        "setting": "a frozen valley of glaciers, ancient ruins, and ice magic",
        "name_example": "Frosthold Valley",
        "npc_prefix": "an aged ice guardian or frost shaman",
        "quest_type": "lost ice relic or frozen dungeon",
        "weapon_type": "lightning sword or ice-infused spear",
        "lore_hook": "a powerful mage kingdom consumed by endless winter",
    },
    "sci_fi": {
        "adjectives": "futuristic, dystopian, neon-lit, corporate-controlled, cyberpunk",
        "setting": "a decaying orbital station or cyberpunk megacity ruled by corporations",
        "name_example": "Neon Drift Station",
        "npc_prefix": "a rogue android or rebel hacker",
        "quest_type": "hacking a corporate AI or escaping a space prison",
        "weapon_type": "plasma cannon or mono-filament blade",
        "lore_hook": "humanity's last refuge turned into a corporate prison",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# Loads model once and caches it so repeated calls don't reload from disk.
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def load_model(model_name: str = "distilgpt2"):
    """
    Load DistilGPT2 tokenizer and model from HuggingFace.
    The first call downloads the model (~80MB). Subsequent calls use the cache.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"[INFO] Loading model '{model_name}'... (first run downloads ~82 MB)")
        _tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        _model = GPT2LMHeadModel.from_pretrained(model_name)
        _model.eval()  # Set to inference mode — no gradient computation
        print("[INFO] Model loaded successfully.\n")
    return _tokenizer, _model


# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATION PRIMITIVE
# Generates a short completion for a given prompt.
# ─────────────────────────────────────────────────────────────────────────────

def _complete(
    prompt: str,
    tokenizer,
    model,
    temperature: float,
    max_new_tokens: int,
) -> str:
    """
    Generate a short text completion for the given prompt.
    Returns only the newly generated text (not the prompt itself).
    """
    # Encode input; suppress attention_mask warning by encoding without padding
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,   # Generate at most this many NEW tokens
            do_sample=True,                  # Stochastic sampling for variety
            temperature=max(temperature, 0.1),
            top_p=0.92,                      # Nucleus sampling
            top_k=50,                        # Limit sampling pool
            repetition_penalty=1.3,         # Penalise repeated phrases
            pad_token_id=50256,              # GPT2's EOS token ID (no padding needed)
        )

    # Slice off the input tokens → only the newly generated portion
    new_tokens = output_ids[0][input_ids.shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Return the first line only (or up to a period/newline) to keep it clean
    completion = completion.strip()
    # Cut at the first occurrence of a field header or double newline
    for stop in ["\n\n", "World Name:", "Description:", "NPC:", "Quest:", "Reward Item:", "Lore:"]:
        if stop in completion:
            completion = completion[:completion.index(stop)].strip()
    return completion if completion else "(not generated)"


# ─────────────────────────────────────────────────────────────────────────────
# FIELD PROMPT BUILDERS
# Each function builds a targeted prompt for one RPG world field.
# ─────────────────────────────────────────────────────────────────────────────

def _build_name_prompt(cfg: dict) -> str:
    return (
        f"A {cfg['adjectives']} RPG location name (like \"{cfg['name_example']}\"):\n"
        f"Place Name:"
    )


def _build_description_prompt(cfg: dict, world_name: str) -> str:
    return (
        f"Describe a {cfg['adjectives']} RPG world called \"{world_name}\" "
        f"set in {cfg['setting']}. Write 1-2 sentences:\n"
        f"Description:"
    )


def _build_npc_prompt(cfg: dict, world_name: str) -> str:
    return (
        f"Create a named {cfg['adjectives']} RPG character who lives in \"{world_name}\". "
        f"They should be {cfg['npc_prefix']}. Format: Name — description.\n"
        f"NPC:"
    )


def _build_quest_prompt(cfg: dict, world_name: str) -> str:
    return (
        f"Write a short {cfg['adjectives']} RPG quest involving a {cfg['quest_type']} "
        f"in the world of \"{world_name}\". One sentence.\n"
        f"Quest:"
    )


def _build_reward_prompt(cfg: dict) -> str:
    return (
        f"Name a {cfg['adjectives']} RPG weapon or item reward. "
        f"It should be a {cfg['weapon_type']}. Format: Item Name — short description.\n"
        f"Reward Item:"
    )


def _build_lore_prompt(cfg: dict, world_name: str) -> str:
    return (
        f"Write a short lore fact (1 sentence) about the history of \"{world_name}\". "
        f"It should mention {cfg['lore_hook']}.\n"
        f"Lore:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# WORLD GENERATOR
# Core function — generates all 6 fields and formats the world scenario.
# ─────────────────────────────────────────────────────────────────────────────

def generate_world(
    theme: str = "fantasy",
    temperature: float = 0.8,
    max_length: int = 250,
    seed: int = None,
) -> str:
    """
    Generate a complete RPG world scenario using field-by-field generation.

    Parameters
    ----------
    theme       : World theme — one of THEME_CONFIGS keys.
    temperature : Controls creativity. Low (0.3) = structured. High (1.2) = creative.
    max_length  : Controls tokens per field (total budget ÷ 6 fields).
    seed        : Random seed for reproducibility. None = random each time.

    Returns
    -------
    str : Formatted RPG world scenario string.
    """

    # Validate theme
    if theme not in THEME_CONFIGS:
        available = ", ".join(THEME_CONFIGS.keys())
        raise ValueError(f"Unknown theme '{theme}'. Available: {available}")

    # Set seed for reproducibility if requested
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    tokenizer, model = load_model()
    cfg = THEME_CONFIGS[theme]

    # Per-field token budget: distribute max_length across the 6 fields
    # Name gets fewer tokens (it's short); description/lore get more
    tokens_per_field = max(20, max_length // 6)

    # ── Generate each field ────────────────────────────────────────────────────
    world_name = _complete(_build_name_prompt(cfg), tokenizer, model, temperature, tokens_per_field)
    # Clean up world name — take only the first line
    world_name = world_name.split("\n")[0].strip().strip("-—").strip()
    if not world_name:
        world_name = cfg["name_example"]  # Fallback to the example from config

    description = _complete(_build_description_prompt(cfg, world_name), tokenizer, model, temperature, tokens_per_field * 2)
    npc = _complete(_build_npc_prompt(cfg, world_name), tokenizer, model, temperature, tokens_per_field * 2)
    quest = _complete(_build_quest_prompt(cfg, world_name), tokenizer, model, temperature, tokens_per_field)
    reward = _complete(_build_reward_prompt(cfg), tokenizer, model, temperature, tokens_per_field)
    lore = _complete(_build_lore_prompt(cfg, world_name), tokenizer, model, temperature, tokens_per_field * 2)

    # ── Format the final output ────────────────────────────────────────────────
    world_text = _format_world(theme, world_name, description, npc, quest, reward, lore, temperature, max_length, seed)
    return world_text


def _format_world(
    theme: str,
    world_name: str,
    description: str,
    npc: str,
    quest: str,
    reward: str,
    lore: str,
    temperature: float,
    max_length: int,
    seed,
) -> str:
    """Format all six fields into a clean, readable RPG world scenario."""
    lines = [
        "=" * 60,
        f"  🌍  RPG WORLD — {theme.upper().replace('_', ' ')}  🌍",
        "=" * 60,
        "",
        f"  World Name:    {world_name}",
        "",
        "  Description:",
        f"    {description}",
        "",
        "  NPC:",
        f"    {npc}",
        "",
        "  Quest:",
        f"    {quest}",
        "",
        "  Reward Item:",
        f"    {reward}",
        "",
        "  Lore:",
        f"    {lore}",
        "",
        "=" * 60,
        f"  Generated with: theme={theme} | temperature={temperature} | max_length={max_length} | seed={seed}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SAVER
# ─────────────────────────────────────────────────────────────────────────────

def save_output(text: str, theme: str, output_dir: str = "outputs") -> str:
    """Save the generated world scenario to a timestamped file in outputs/."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"world_{theme}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI RPG World Generator — powered by DistilGPT2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --theme fantasy
  python generate.py --theme dark_fantasy --temperature 0.9 --max_length 200
  python generate.py --theme arctic --seed 42 --no-save
        """,
    )

    parser.add_argument(
        "--theme",
        type=str,
        default="fantasy",
        choices=list(THEME_CONFIGS.keys()),
        help="World theme to generate (default: fantasy)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature — controls creativity (default: 0.8). "
             "Range: 0.1 (deterministic) to 1.5 (very creative).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=250,
        help="Total token budget for generating all world fields combined (default: 250).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible outputs (default: random).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If set, do not save output to the outputs/ folder.",
    )

    args = parser.parse_args()

    print(f"\n🎲 Generating RPG world: theme='{args.theme}' | temp={args.temperature} | "
          f"max_length={args.max_length} | seed={args.seed}\n")

    world = generate_world(
        theme=args.theme,
        temperature=args.temperature,
        max_length=args.max_length,
        seed=args.seed,
    )

    print(world)

    if not args.no_save:
        path = save_output(world, args.theme)
        print(f"\n✅ Saved to: {path}")


if __name__ == "__main__":
    main()
