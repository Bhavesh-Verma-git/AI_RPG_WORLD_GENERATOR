# AI RPG World Generator - Complete Project Explanation Guide

**Prepared for: Interview Preparation**
**Project Status:** Recruiter Task (Accepted for Interview)
**Technology Stack:** Python, PyTorch, HuggingFace Transformers, FastAPI

---

## 📚 TABLE OF CONTENTS
1. [Executive Summary](#executive-summary)
2. [Project Architecture](#project-architecture)
3. [File-by-File Breakdown](#file-by-file-breakdown)
4. [Technology Stack & Choices](#technology-stack--choices)
5. [How to Explain to Recruiters](#how-to-explain-to-recruiters)
6. [Interview Q&A](#interview-qa)
7. [Key Concepts & Focus Areas](#key-concepts--focus-areas)
8. [Interview Tips](#interview-tips)

---

## EXECUTIVE SUMMARY

### What is this project?
An **AI-powered procedural RPG world generator** that uses a lightweight language model (DistilGPT2) to generate complete fantasy game worlds with:
- **World Name** (e.g., "Frosthold Valley")
- **Description** (story/setting)
- **NPC** (non-player character)
- **Quest** (adventure objective)
- **Reward Item** (loot/weapon)
- **Lore** (world history/mythology)

### Why it's impressive:
✅ **No GPU needed** - runs entirely on CPU
✅ **No training required** - uses prompt engineering on pre-trained model
✅ **Production-ready** - includes CLI tool, FastAPI backend, and automated experiments
✅ **Demonstrable** - includes sample outputs and parameter sensitivity analysis

### Technical Takeaway:
This project shows you understand:
- Machine learning fundamentals (model loading, inference)
- Prompt engineering (the modern approach to NLP without fine-tuning)
- Software engineering (clean code, CLI design, API architecture)
- Experimental methodology (parameter tuning, reproducibility)

---

## PROJECT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    User/Recruiter                            │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  ┌──────────┐      ┌──────────┐
  │ CLI Tool │      │ FastAPI  │
  │generate. │      │api.py    │
  │py        │      │          │
  └────┬─────┘      └────┬─────┘
       │                 │
       └────────┬────────┘
                │
                ▼
        ┌──────────────────────┐
        │ Core Generator       │
        │ generate_world()     │
        │ in generate.py       │
        └────────┬─────────────┘
                 │
        ┌────────┴──────────────────┐
        │                           │
        ▼                           ▼
   ┌─────────┐           ┌──────────────────┐
   │ Tokenizer│           │  DistilGPT2 Model│
   │in        │           │ (82M parameters) │
   │generate. │           │ HuggingFace      │
   │py        │           │                  │
   └─────────┘           └──────────────────┘
   (Line 92)             (Lines 84-96)
        │                       │
        └───────────┬───────────┘
                    │
            ┌───────▼────────┐
            │ DistilGPT2     │
            │ Model Cache    │
            │ (~80 MB)       │
            │ Downloaded on  │
            │ first run      │
            └────────────────┘
```

### Data Flow:
1. **User provides parameters** → theme, temperature, max_length, seed
2. **Theme config selected** → pulls adjectives, examples, prompts
3. **Prompts built** → five separate prompts (one per field)
4. **Model completion** → each prompt fed to DistilGPT2 for generation
5. **Text cleaning** → regex post-processing to remove artifacts
6. **Formatting** → assembled into final world structure
7. **Output saved** → written to `outputs/world_*.txt`

---

## FILE-BY-FILE BREAKDOWN

### 1. **generate.py** (415 lines)
**Purpose:** Core world generation engine

#### Key Sections:

**A) Theme Configuration (Lines 26-72)**
```python
THEME_CONFIGS = {
    "fantasy": {
        "adjectives": "high-fantasy, magical, ancient, elven, mystical",
        "setting": "a world of ancient elven forests...",
        "name_example": "Eldenmoor",
        ...
    },
    ...
}
```
- Stores 5 different theme templates
- Each theme has 7 attributes (adjectives, setting, name_example, npc_prefix, quest_type, weapon_type, lore_hook)
- **Why separate themes?** Allows model to stay contextually relevant without fine-tuning
- **Why dictionary?** Scalable - adding a theme is just one entry

**B) Model Loader (Lines 84-96)**
```python
def load_model(model_name: str = "distilgpt2"):
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"[INFO] Loading model '{model_name}'...")
        _tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        _model = GPT2LMHeadModel.from_pretrained(model_name)
        _model.eval()
        print("[INFO] Model loaded successfully.\n")
    return _tokenizer, _model
```
- **Global caching** (lines 80-81): Loads model once, reuses for all generations (CRITICAL for performance)
- **from_pretrained()** (lines 92-93): Downloads from HuggingFace hub
- **eval() mode** (line 94): Disables training layers (dropout, batch norm) for inference
- **Why DistilGPT2?**
  - 82M parameters (vs GPT-2's 124M)
  - 80 MB download size
  - Runs on CPU in seconds
  - Maintains coherent output quality
- **Alternatives considered:**
  - **GPT-2 Medium (355M)** → Too slow on CPU, overkill for this task
  - **FLAN-T5 (250M)** → Better instruction-following, but larger
  - **LLaMA-7B** → Requires GPU for reasonable speed
  - **Mistral-7B** → Even larger
  - **Fine-tuned models** → Require labeled RPG data (time/cost prohibitive)

**C) Completion Function (Lines 101-155)**
```python
def _complete(prompt: str, tokenizer, model,
              temperature: float, max_new_tokens: int) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():  # Line 112: Disable gradient computation
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Line 116: Use sampling, not greedy decoding
            temperature=max(temperature, 0.1),
            top_p=0.92,      # Nucleus sampling parameter
            top_k=50,        # Limit to top 50 tokens
            repetition_penalty=1.3,
            pad_token_id=50256,
        )

    # Lines 125-155: Aggressive post-processing
    new_tokens = output_ids[0][input_ids.shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # ... cuts at double newlines, field headers, 120 char limit, cleans artifacts
```
- **Key parameters explained:**
  - `do_sample=True`: Use probabilistic sampling (vs greedy, which always picks highest prob token)
  - `temperature`: Scaling factor for probability distribution (0.1=deterministic, 2.0=chaotic)
  - `top_p=0.92`: Nucleus sampling - only sample from tokens whose cumsum prob < 0.92
  - `top_k=50`: Limits pool to top 50 most likely tokens
  - `repetition_penalty=1.3`: Penalizes tokens already used (reduces loops)
- **Why torch.no_grad()?** (line 112) Disables automatic differentiation during inference (saves memory, speed)
- **Post-processing (lines 131-175):**
  - Cuts at field boundaries and newlines
  - Removes parenthetical stats `(5)`, `(HP: 10/10)`
  - Removes mathematical expressions `2x3 - 1`
  - Cleans up punctuation artifacts

**D) Prompt Builders (Lines 180-247)**
Five functions build theme-specific prompts using **few-shot prompting**:

```python
def _build_name_prompt(cfg: dict) -> str:
    return (
        f"Examples of short {cfg['adjectives']} RPG world names:\n"
        f"- {cfg['name_example']}\n"
        f"- Thornveil Reach\n"
        f"One new {cfg['adjectives']} RPG world name (2-4 words only):\n"
        f"World Name:"
    )
```
- **Few-shot prompting:** Shows 2+ examples, then asks for more
- **Why?** Model learns pattern without gradient updates
- **Alternative:** Zero-shot ("Generate a fantasy world name") - less reliable output format
- Each of 5 functions follows same pattern with domain-specific examples

**E) Main Generator (Lines 252-292)**
```python
def generate_world(theme: str = "fantasy",
                   temperature: float = 0.8,
                   max_length: int = 250,
                   seed: int = None) -> str:

    if theme not in THEME_CONFIGS:
        raise ValueError(...)

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    tokenizer, model = load_model()
    cfg = THEME_CONFIGS[theme]

    tokens_per_field = max(20, max_length // 6)

    # Generate each of 6 fields separately
    world_name = _complete(_build_name_prompt(cfg), ..., tokens_per_field)
    description = _complete(_build_description_prompt(cfg, world_name), ..., tokens_per_field * 2)
    # ... NPC, Quest, Reward, Lore

    world_text = _format_world(...)
    return world_text
```
- **Token budget distribution:** Divides `max_length` across 6 fields
  - Name: 1x budget (short)
  - Description, NPC, Lore: 2x budget (longer fields)
  - Quest, Reward: 1x budget (structured)
- **Why separate prompts per field?** Better control + cleaner outputs than one giant prompt
- **Why call load_model() inside?** Allows reuse across API and CLI

**F) Output Saver (Lines 339-347)**
```python
def save_output(text: str, theme: str, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"world_{theme}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath
```
- **Timestamped filenames:** Prevents overwriting previous outputs
- **UTF-8 encoding:** Handles special characters (é, ü, etc.)

**G) CLI Interface (Lines 352-416)**
```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--theme", choices=list(THEME_CONFIGS.keys()), ...)
    parser.add_argument("--temperature", type=float, default=0.8, ...)
    parser.add_argument("--max_length", type=int, default=250, ...)
    parser.add_argument("--seed", type=int, default=None, ...)
    parser.add_argument("--no-save", action="store_true", ...)

    args = parser.parse_args()
    world = generate_world(...)
    print(world)
    if not args.no_save:
        path = save_output(world, args.theme)
```
- **Argparse library:** Standard Python CLI tool
- **Why argparse vs Click?** Argparse is built-in (no extra dependency)
- **Alternatives:**
  - **Click:** More elegant, used in popular tools (Flask, Black)
  - **Typer:** Modern, type-hint based
  - **Hydra:** Config file based (overkill for this project)

**Command examples:**
```bash
python generate.py --theme fantasy
python generate.py --theme dark_fantasy --temperature 0.9 --max_length 200
python generate.py --theme arctic --seed 42  # Reproducible
```

---

### 2. **api.py** (109 lines)
**Purpose:** ExposeGenerator as REST API (FastAPI backend)

#### Code Walkthrough:

```python
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from generate import generate_world, THEME_CONFIGS

app = FastAPI(
    title="AI RPG World Generator API",
    description=(...),
    version="1.0.0",
)

class WorldResponse(BaseModel):  # Lines 33-39: Pydantic schema
    theme: str
    temperature: float
    max_length: int
    seed: int | None
    world: str
```

**Endpoints:**

**A) Health Check (Lines 44-52)**
```python
@app.get("/", summary="Health check")
def root():
    return {
        "status": "online",
        "message": "AI RPG World Generator API is running.",
        "docs": "/docs",
        "available_themes": list(THEME_CONFIGS.keys()),
    }
```

**B) Generate World (Lines 55-102)**
```python
@app.get("/generate_world", response_model=WorldResponse)
def generate_world_endpoint(
    theme: str = Query(
        default="fantasy",
        enum=list(THEME_CONFIGS.keys()),
    ),
    temperature: float = Query(default=0.8, ge=0.1, le=2.0),
    max_length: int = Query(default=250, ge=50, le=500),
    seed: int | None = Query(default=None),
):
    try:
        world_text = generate_world(
            theme=theme,
            temperature=temperature,
            max_length=max_length,
            seed=seed,
        )
        return WorldResponse(
            theme=theme,
            temperature=temperature,
            max_length=max_length,
            seed=seed,
            world=world_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
```
- **Why FastAPI?**
  - Auto-generates interactive Swagger UI (`/docs`)
  - Built-in request validation (Pydantic)
  - Fast (async support)
  - Type hints → docs automatically generated
- **Query parameters:** Constraints handled by FastAPI
  - `ge=0.1, le=2.0` → validates temperature is in range
  - `enum=...` → only allows specific themes
- **Error handling:** 400 for client errors, 500 for server errors

**C) List Themes (Lines 105-108)**
```python
@app.get("/themes", summary="List available world themes")
def list_themes():
    return {"themes": list(THEME_CONFIGS.keys())}
```

**Usage:**
```bash
uvicorn api:app --reload
# Visit http://127.0.0.1:8000/docs for interactive Swagger
# GET http://127.0.0.1:8000/generate_world?theme=dark_fantasy&temperature=0.9
```

**Why FastAPI over Flask?**
| Feature | FastAPI | Flask |
|---------|---------|-------|
| Auto-docs | ✅ Swagger + ReDoc | ❌ Requires Flasgger |
| Validation | ✅ Built-in (Pydantic) | ❌ Manual |
| Async | ✅ Native | ❌ Added later |
| Type hints | ✅ Required → docs | ❌ Optional |
| Performance | ✅ Fastest in Python | ✅ Good |
| Learning curve | ⭕ Moderate | ✅ Easy |

---

### 3. **experiments.py** (150 lines)
**Purpose:** Automated parameter sensitivity analysis

#### Structure:

```python
EXPERIMENTS = [
    {
        "id": "exp1_low_temperature",
        "description": "Low temperature (0.3) — tests structured, deterministic generation",
        "params": {"theme": "fantasy", "temperature": 0.3, "max_length": 200, "seed": 1},
    },
    # ... 9 more experiments
]

def run_experiments(output_dir: str = "outputs/experiments"):
    os.makedirs(output_dir, exist_ok=True)

    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        params = exp["params"]

        start = time.time()
        world_text = generate_world(**params)
        elapsed = time.time() - start

        # Save with header metadata
        out_path = os.path.join(output_dir, f"{exp_id}.txt")
        with open(out_path, "w") as f:
            f.write(f"EXPERIMENT: {exp_id}\n...")
            f.write(world_text)

        print(f"✅ {exp_id} — saved ({elapsed:.1f}s)")

    # Print summary table
    print("SUMMARY TABLE...")
```

**10 experiments test:**
1. **Low temperature (0.3)** → Deterministic output
2. **Medium temperature (0.8)** → Balanced
3. **High temperature (1.2)** → Creative
4. **Short output (80 tokens)** → Truncated worlds
5. **Medium output (200 tokens)** → Baseline
6. **Long output (400 tokens)** → Detailed worlds
7-11. **Five themes** at baseline parameters

**Why important?**
- Shows you can **validate hypotheses** with code
- Demonstrates understanding of **hyperparameter tuning**
- Provides empirical evidence of choices (why 0.8 temperature is default)
- Professional touch: "I didn't just build it, I tested it"

---

### 4. **experiments.md** (192 lines)
**Purpose:** Write-up of experiment findings

**Structure:**
```markdown
# Parameter Experiments — AI RPG World Generator

## Experiment Setup
- Model: distilgpt2
- Fixed seed: 1

## Experiment Axis 1 — Temperature
[3 temperature levels tested]
- Observations for each
- Insights

## Experiment Axis 2 — Output Length
[3 length levels tested]

## Experiment Axis 3 — Theme Variation
[5 themes tested]

## Overall Conclusions
[Summary table with best practices]
```

**Key findings documented:**
- **Best temperature for legibility:** 0.7-0.8
- **Best max_length for complete world:** 200-300 tokens
- **Most reliable theme:** fantasy (best pre-training coverage)
- **Least reliable theme:** sci_fi (needs fine-tuning)

**Why document experiments?**
- Shows **scientific rigor** (hypothesis → test → document)
- Gives recruiters **confidence** in your choices
- Demonstrates communication skills (explain findings clearly)

---

### 5. **requirements.txt** (21 lines)
**Purpose:** Dependency management

```
# ── Core ML ────────────────────────────────────────────────────────────────────
# HuggingFace transformers library — provides GPT2 tokenizer and model
transformers>=4.40.0

# PyTorch — the deep learning backend (CPU-only version is fine)
torch>=2.2.0

# ── API ────────────────────────────────────────────────────────────────────────
# FastAPI — modern, fast web framework for building APIs
fastapi>=0.111.0

# Uvicorn — ASGI server for running FastAPI locally
uvicorn>=0.29.0

# Pydantic — data validation used internally by FastAPI
pydantic>=2.7.0
```

**Dependencies breakdown:**

| Package | Version | Why |
|---------|---------|-----|
| **transformers** | 4.40.0+ | Loads DistilGPT2, tokenizer, text generation utilities |
| **torch** | 2.2.0+ | Backend for model inference (CPU-only build OK) |
| **fastapi** | 0.111.0+ | REST API framework |
| **uvicorn** | 0.29.0+ | ASGI server to run FastAPI |
| **pydantic** | 2.7.0+ | Request/response validation in FastAPI |

**Why minimal dependencies?**
- Reduces complexity
- Easier deployment
- Smaller footprint on CPU systems

**Installation:**
```bash
python -m venv rpg_env
# Windows:
rpg_env\Scripts\activate
# Mac/Linux:
source rpg_env/bin/activate

pip install -r requirements.txt
# First run downloads ~80 MB DistilGPT2 model
```

---

### 6. **README.md** (245 lines)
**Purpose:** Project documentation

**Sections:**
1. **Project Overview** (lines 8-23)
2. **Model Justification** (lines 27-46) → Why DistilGPT2
3. **Prompt Engineering Explanation** (lines 50-72) → Few-shot prompting
4. **World Themes** (lines 77-85) → 5 theme descriptions
5. **Quick Start** (lines 89-121) → Install + run
6. **CLI Reference** (lines 125-137) → All command options
7. **FastAPI Backend** (lines 141-165) → How to run API
8. **Parameter Experiments** (lines 169-177) → How to run
9. **Project Structure** (lines 181-198) → File tree
10. **Parameter Explanations** (lines 202-211) → Deep dive into each param
11. **Future Improvements** (lines 215-235) → Roadmap

**Why well-documented README?**
- Shows **professionalism**
- Allows **anyone** to run the project
- Demonstrates **communication** (explain complex ideas simply)

---

### 7. **.gitignore** (implicit)
```
rpg_env/
outputs/
.venv/
__pycache__/
*.pyc
.env
```
- **rpg_env/** → Virtual environment (recreatable, shouldn't be committed)
- **outputs/** → Generated files (can be large/temporary)
- **__pycache__/** → Python bytecode (auto-generated)

---

## TECHNOLOGY STACK & CHOICES

### Core ML Stack

#### 1. **PyTorch** (vs TensorFlow, JAX)
**Choice:** PyTorch 2.2.0+

**Why PyTorch?**
- ✅ Pythonic, intuitive API
- ✅ Dominant in research + NLP community
- ✅ Excellent debuggability (imperative style)
- ✅ Strong HuggingFace integration
- ✅ Smaller memory footprint on CPU

**Alternatives:**
- **TensorFlow 2.x**: More mature, better mobile support, but heavier
- **JAX**: Functional approach, slower to learn, for research only

```python
# PyTorch usage in project:
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

with torch.no_grad():  # Critical for inference (disables gradients)
    output = model.generate(input_ids, max_new_tokens=250, ...)
```

#### 2. **HuggingFace Transformers** (vs other model APIs)
**Choice:** transformers>=4.40.0

**Why HuggingFace?**
- ✅ 1-line model loading
- ✅ Model zoo of 100k+ pretrained models
- ✅ Consistent API across models
- ✅ Active community support
- ✅ Battle-tested in production

**Alternatives:**
- **OpenAI API**: Requires API key, cloud-based, costs $
- **LocalLLM (Ollama)**: No code library integration
- **Replicate**: Cloud-based, slower
- **LiteLLM**: Wrapper over APIs, adds latency

```python
# HuggingFace usage:
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("distilgpt2")  # One line!
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
```

#### 3. **DistilGPT2 Model** (vs other models)
**Choice:** DistilGPT2 (82M parameters)

**Why DistilGPT2?**
- ✅ Runs on CPU in seconds
- ✅ ~80 MB download (small)
- ✅ No GPU required (accessible)
- ✅ Good English knowledge from pretraining
- ✅ Sufficient for creative tasks

**Model Comparison:**

| Model | Size | Speed (CPU) | Quality | Use Case |
|-------|------|-----------|---------|----------|
| **DistilGPT2** | 82M | ✅ 5-15s | ⭐⭐⭐ | Lightweight, CPU |
| **GPT2** | 124M | ⭐⭐ 15-30s | ⭐⭐⭐⭐ | General purpose |
| **GPT2-Medium** | 355M | ❌ 60+ sec | ⭐⭐⭐⭐⭐ | Quality over speed |
| **GPT-2 Large** | 774M | ❌ GPU only | ⭐⭐⭐⭐⭐⭐ | High quality |
| **T5-Base** | 220M | ❌ Slow | ⭐⭐⭐ | Task-specific |
| **LLaMA-7B** | 7B | ❌ GPU | ⭐⭐⭐⭐⭐⭐⭐ | Overkill |
| **Mistral-7B** | 7B | ❌ GPU | ⭐⭐⭐⭐⭐⭐⭐ | Overkill |

**For this project:** DistilGPT2 is **optimal**.

---

### Web Framework

#### **FastAPI** (vs Flask, Django)
**Choice:** FastAPI 0.111.0+

**Why FastAPI?**
- ✅ Auto-generated Swagger UI (`/docs`)
- ✅ Built-in request validation (Pydantic)
- ✅ Type hints → automatic documentation
- ✅ Async support by default
- ✅ Modern (async/await syntax)
- ✅ Fast routing, low overhead

**Code comparison:**

```python
# FastAPI (simple, powerful)
from fastapi import FastAPI, Query

@app.get("/generate_world")
def generate_world_endpoint(
    theme: str = Query(default="fantasy", enum=["fantasy", "dark_fantasy"]),
    temperature: float = Query(default=0.8, ge=0.1, le=2.0)
):
    return {"world": generate_world(theme, temperature)}
    # Automatically validates theme ∈ enum, 0.1 ≤ temperature ≤ 2.0
    # Generates Swagger docs automatically
```

```python
# Flask (manual, verbose)
from flask import Flask, request

@app.route("/generate_world")
def generate_world_endpoint():
    theme = request.args.get("theme", "fantasy")
    if theme not in ["fantasy", "dark_fantasy"]:
        return {"error": "Invalid theme"}, 400

    temperature = request.args.get("temperature", 0.8, type=float)
    if not (0.1 <= temperature <= 2.0):
        return {"error": "Temperature out of range"}, 400

    return {"world": generate_world(theme, temperature)}
    # Must manually write Swagger docs
```

**Alternatives:**
- **Flask**: Simpler for tiny projects, but manual validation
- **Django**: Overkill for this project (built for full web apps)
- **Quart**: Async Flask, less mature than FastAPI
- **Starlette**: Lower level, FastAPI is built on this

---

### CLI Tool

#### **argparse** (vs Click, Typer)
**Choice:** argparse (built into Python stdlib)

**Why argparse?**
- ✅ Standard library (no external dependency)
- ✅ Sufficient for simple CLIs
- ✅ Familiar to most Python developers
- ✅ Flexible argument types and validation

**Code in project (lines 352-416 of generate.py):**
```python
parser = argparse.ArgumentParser(...)
parser.add_argument("--theme", choices=list(THEME_CONFIGS.keys()), ...)
parser.add_argument("--temperature", type=float, default=0.8, ...)
parser.add_argument("--max_length", type=int, default=250, ...)
parser.add_argument("--seed", type=int, default=None, ...)
parser.add_argument("--no-save", action="store_true", ...)
```

**Alternatives:**
- **Click**: More elegant decorators, used by Flask/pip (`@click.command()`)
- **Typer**: Modern, type-hint based, great docs
- **Hydra**: Config file based (overkill here)

```python
# Click version (more elegant)
@click.command()
@click.option("--theme", default="fantasy", type=click.Choice([...]))
@click.option("--temperature", default=0.8, type=float)
def main(theme, temperature):
    world = generate_world(theme, temperature)
    print(world)
```

For **learners/interviews**, argparse shows you know standard library.

---

## HOW TO EXPLAIN TO RECRUITERS

### The Elevator Pitch (30 seconds)
```
"I built an AI system that procedurally generates complete RPG worlds
using prompt engineering. It runs entirely on CPU using DistilGPT2,
takes 5-15 seconds per world, and includes a CLI tool, FastAPI backend,
and automated parameter validation. I submitted it as a recruiter task
and it demonstrates understanding of ML fundamentals, modern NLP techniques,
and good software engineering practices."
```

### The Complete Explanation (5-10 minutes)

**1. Start with the problem:**
```
"A lot of RPG games need procedurally generated content — NPCs, quests, lore,
items. Typically, you'd either hardcode templates or hire writers. I wanted
to show how modern language models can automate this without requiring GPU
compute or massive datasets."
```

**2. Walk through the architecture:**
```
"The system has three main components:

1. Core Generator (generate.py):
   - Loads DistilGPT2, a lightweight language model with 82M parameters
   - Uses few-shot prompt engineering to generate 6 world fields:
     name, description, NPC, quest, reward item, lore
   - Each field gets its own prompt with examples to guide the model
   - Output is cleaned with regex to remove artifacts
   - Results are saved with timestamp

2. Web API (api.py):
   - FastAPI backend with 3 endpoints
   - /generate_world takes theme, temperature, max_length as parameters
   - Auto-validates inputs and generates Swagger documentation
   - Shows how to integrate ML into a production service

3. Experimentation (experiments.py + experiments.md):
   - Systematically tests how temperature, output length, and themes
     affect quality
   - Documents findings with concrete observations
   - Shows 0.8 temperature is the sweet spot for this task
```

**3. Explain the technology choices:**
```
"Why these choices?

- DistilGPT2 instead of GPT-2: Same knowledge, 40% smaller, runs on consumer CPU
- PyTorch instead of TensorFlow: Pythonic and dominant in NLP
- HuggingFace instead of OpenAI API: Free, runs locally, reproducible
- Prompt engineering instead of fine-tuning: No data needed, fast iteration
- FastAPI: Auto-generates docs and validates inputs with minimal code
- Argparse: Standard library, no extra dependencies
"
```

**4. Discuss results:**
```
"The system generates coherent, thematically consistent worlds. For example:

FANTASY world might produce:
  - World Name: Silvantium Vale
  - Description: An ancient elven lands where magic flows through crystalline rivers
  - NPC: Lysara the Starweaver — a mage who communes with celestial spirits
  - Quest: Recover the Moonstone Scepter from the Temple of Echoes
  - Reward Item: Skybound Cloak — grants the wearer command over winds and clouds
  - Lore: The vale was once the capital of an immortal kingdom until a curse darkened the stars

DARK FANTASY world might produce:
  - Cursed setting, undead themes, high stakes

The system is deterministic with seeds (reproducible) and
randomized by default (infinite variety).
"
```

**5. Highlight engineering quality:**
```
"From a software engineering perspective, the code shows:
- Clean separation of concerns (CLI, API, core logic)
- Proper error handling with try/except and HTTPException
- Reproducibility (seed management, version pinning in requirements.txt)
- Documentation (README covers every aspect)
- Testing (experiments with controlled parameters)
- Extensibility (adding themes is just one config entry)
"
```

**6. Discuss alternatives considered:**
```
"I chose DistilGPT2 and CPU because:

Alternative: Fine-tune on RPG data
  - Pro: Higher quality outputs
  - Con: Needs labeled data (100+ examples), training time (hours), infrastructure
  - My choice: Not time-effective for a demonstration

Alternative: Use OpenAI API
  - Pro: State-of-the-art quality
  - Con: Requires API key, costs money, not reproducible locally
  - My choice: Wrong for showing system design skills

Alternative: Use larger model (Mistral-7B)
  - Pro: Better quality
  - Con: Requires GPU, 15 GB memory, slow even on GPU
  - My choice: Defeats 'runs on consumer hardware' goal

I optimized for the recruiter's likely constraints: quick demo,
no cloud costs, runs on their laptop.
"
```

### Common Interviewer Questions & Answers

**Q: Why did you choose prompt engineering over fine-tuning?**
```
A: "Fine-tuning requires labeled RPG data, training time, and GPU compute.
Prompt engineering achieves 80% of the quality with 5% of the effort.
For a demo showing technical breadth, prompt engineering is the right tool.
In production with a large labeled dataset, fine-tuning would be worth it."
```

**Q: How would you improve this for production?**
```
A: "Several avenues:

1. Fine-tune on fantasy literature + D&D datasets (better domain relevance)
2. Use a larger model if budget allows (GPT-2 Medium, LLaMA-7B with quantization)
3. Add RAG (Retrieval-Augmented Generation): fetch relevant lore snippets
   from a knowledge base before generation → more coherent worlds
4. Implement constrained decoding (outlines library) to guarantee all
   6 fields are populated in valid JSON
5. Add a Web UI (React + FastAPI) for interactive generation
6. Add evaluation metrics (BLEU, diversity scores, user ratings)
"
```

**Q: How does temperature work exactly?**
```
A: "Temperature scales the logits (raw model outputs) before softmax normalization.
   Lower temperature = sharper distribution = model picks safer words
   Higher temperature = flatter distribution = model takes more risks

   Mathematically: softmax(logits / T)
   T=0.3: Distribution becomes peaks and valleys → deterministic
   T=0.8: Natural probability distribution
   T=1.5: Flattened distribution → very creative but less coherent

   For this project, 0.8 balances coherence with creativity.
"
```

**Q: What happens if the model generates an incomplete world?**
```
A: "The _complete() function has aggressive post-processing:
   - Cuts at field boundaries (if it generates 'Quest: X NPC: Y',
     we extract just the quest)
   - Removes stat artifacts like '(5/10)' that the model sometimes adds
   - Caps at 120 characters per field
   - Trims to the first complete sentence

   If a field is still empty (rare), _complete() returns '(not generated)'
   and we show the user a partial world. This is acceptable for the demo —
   in production, we'd use constrained decoding to guarantee all fields.
"
```

**Q: Why not use GPT-3 or GPT-4?**
```
A: "OpenAI's models require API keys and cost money:
   - GPT-4: $0.03 per 1K tokens, so ~$0.01 per world
   - To run this hundreds of times during development would cost $10s

   DistilGPT2 is free, runs locally, and is sufficient for creative tasks.
   In production, if OpenAI integration made sense, we'd use their API.
   But this demo prioritizes showing system design, not cutting-edge results.
"
```

---

## INTERVIEW Q&A

### Technical Questions

#### Q1: Walk us through the data flow when a user calls `generate.py --theme fantasy --temperature 0.9`

**A:**
```
1. argparse parses CLI arguments (lines 362-395)
   → theme = "fantasy", temperature = 0.9, max_length = 250, seed = None

2. main() calls generate_world() with these params (line 401)

3. generate_world() validates theme exists (lines 261-263)

4. load_model() is called (line 270)
   → Returns cached tokenizer/model (first call downloads ~80 MB)

5. THEME_CONFIGS["fantasy"] config is loaded (line 271)

6. Token budget is calculated: 250 // 6 ≈ 41 tokens/field (line 275)

7. For each of 6 fields, _complete() is called with a targeted prompt:
   a. _build_name_prompt() → generates "Eldenmoor"
   b. _build_description_prompt() → generates world description
   c. _build_npc_prompt() → generates NPC with name and description
   d. _build_quest_prompt() → generates quest objective
   e. _build_reward_prompt() → generates item/weapon
   f. _build_lore_prompt() → generates mythical history

8. Each call to _complete():
   - Encodes prompt to token IDs (line 110)
   - Calls model.generate() with temperature=0.9 (lines 112-122)
   - Decodes output tokens to text (line 126)
   - Post-processes to clean artifacts (lines 131-155)

9. All fields assembled into final world text (line 291)

10. World saved to outputs/world_fantasy_<timestamp>.txt (line 412)

11. Printed to terminal and user sees complete world
```

---

#### Q2: Why do you call model.generate() with `do_sample=True` instead of greedy decoding?

**A:**
```
Greedy decoding always picks the highest probability token at each step.
This produces safe, repeated output.

Sampling (do_sample=True) randomly picks from the probability distribution.
This produces varied, creative output.

For creative tasks like world generation, sampling is essential.

In the code (line 116):
    do_sample=True,              # Use sampling
    temperature=0.9,              # Control distribution shape
    top_p=0.92,                   # Nucleus sampling (only top cumsum=0.92)
    top_k=50,                     # Limit to top 50 tokens
    repetition_penalty=1.3        # Penalize repeated tokens

Together these prevent:
- Very unlikely, nonsense tokens (top_k, top_p)
- Repetitive loops (repetition_penalty)
- While still allowing creative variance (sampling)

With greedy decoding, you'd get deterministic output every time.
With sampling + temperature, you get variety controlled by temperature.
```

---

#### Q3: Explain the few-shot prompting approach vs zero-shot

**A:**
```
ZERO-SHOT (doesn't work well):
    "Generate a fantasy world name:\n"

Without examples, the model might output:
    - Just gibberish
    - Something not fantasy-themed
    - Multiple words when you wanted one name
    - Non-English text

FEW-SHOT (used in this project):
    "Examples of short, high-fantasy RPG world names:\n"
    "- Eldenmoor\n"
    "- Thornveil Reach\n"
    "One new high-fantasy RPG world name (2-4 words only):\n"
    "World Name:"

The model sees the pattern (2-4 word fantasy names with capitals)
and continues it. Output is much more reliable.

In generate.py (lines 180-188, _build_name_prompt):
    return (
        f"Examples of short {cfg['adjectives']} RPG world names:\n"
        f"- {cfg['name_example']}\n"
        f"- Thornveil Reach\n"
        f"One new {cfg['adjectives']} RPG world name (2-4 words only):\n"
        f"World Name:"
    )

This is called "in-context learning" — the model learns patterns
from the prompt itself, not from gradient updates.
```

---

#### Q4: If you wanted better output quality, what would you do?

**A:**
```
Priority 1 (Best ROI): Fine-tune DistilGPT2 on RPG domain data
  - Collect 500+ examples from: D&D sourcebooks, fantasy wikis, game scripts
  - Fine-tune for 3-5 epochs (∼30 mins on GPU, or hours on CPU)
  - Would dramatically improve thematic consistency and creativity
  - Cost: ~$5-10 in cloud compute for fine-tuning

Priority 2: Use a larger base model
  - GPT-2 Medium (355M params) instead of DistilGPT2 (82M)
  - Better knowledge, but slower (∼30s per world on CPU)
  - Still free, just slower
  - Or quantize LLaMA-7B to run on CPU (4-bit quantization)

Priority 3: Add Retrieval-Augmented Generation (RAG)
  - Store snippets of lore/quests in a vector database (Pinecone, Weaviate)
  - Before generation, retrieve relevant snippets based on theme
  - Inject them into the prompt → more coherent, fact-consistent worlds
  - Cost: ~$10-30/month for vector DB

Priority 4: Use structured/constrained decoding
  - Force output to follow a strict schema (all 6 fields always populated)
  - Use outlines library to constrain generation to valid JSON/regex
  - Eliminates post-processing hacks

Priority 5: Ensemble multiple models
  - Generate 3 candidates, pick the best via scoring function
  - Slower but higher perceived quality

For this recruiter task, the current approach is appropriate.
Fine-tuning would be the next step for production.
```

---

#### Q5: What's the purpose of the `experiments.py` file?

**A:**
```
Demonstrates scientific rigor and hypothesis testing.

It runs 10 controlled experiments varying:
  1. Temperature (0.3, 0.8, 1.2) → creativity vs structure
  2. Max_length (80, 200, 400) → output completeness
  3. Themes (fantasy, dark_fantasy, desert, arctic, sci_fi) → domain coverage

Each uses fixed seed=1 so parameter differences, not randomness, explain results.

Findings documented in experiments.md:
  - temp=0.8 is sweet spot (balanced creativity + structure)
  - max_length=200-300 completes all fields without being slow
  - Fantasy/dark_fantasy most reliable (better pre-training coverage)
  - Sci-fi least reliable (would need fine-tuning)

Why this matters to recruiters:
  - Shows you don't just build, you validate
  - Shows you can run experiments with code
  - Provides empirical justification for defaults
  - Demonstrates methodical problem-solving

In code (experiments.py):
    EXPERIMENTS = [
        {"id": "exp1_low_temperature", "params": {...}, ...},
        ...
    ]

    for exp in EXPERIMENTS:
        world = generate_world(**exp["params"])
        # Save and time the result

    # Print summary table

This is a professional touch that many junior projects lack.
```

---

### Behavioral/Design Questions

#### Q6: Why does the code split the world into 6 separate fields instead of generating everything in one prompt?

**A:**
```
Good question. There are tradeoffs:

OPTION A: Six separate prompts (CHOSEN)
  Pro:
    - Fine-grained control over output format
    - If one field fails, others still succeed
    - Can give different token budgets (description gets 2x, name gets 1x)
    - Easier to post-process and format
    - Cleaner code structure
  Con:
    - Slower (6 forward passes instead of 1)
    - Some fields might be inconsistent
      (NPC might reference a quest that contradicts the quest field)
    - Higher token cost if using paid API

OPTION B: One giant prompt
  Pro:
    - Faster (1 forward pass)
    - More context for consistency
    - Lower token cost on paid APIs
  Con:
    - Harder to constrain format (might output 3 NPCs instead of 1)
    - Post-processing more complex (extract 6 fields from one block)
    - If generation fails, lose everything
    - Can't give different budgets to fields

CHOSEN: Six separate prompts because:
  - Recruiter demo: clarity > speed
  - CPU runs (5-15s is acceptable)
  - Each field gets tailored prompt + examples
  - Easier to debug if something goes wrong

If I were optimizing for latency or API cost, I'd switch to Option B.
```

---

#### Q7: How does caching the model in global variables improve performance?

**A:**
```
Loading a language model from disk is expensive (~2-3 seconds for DistilGPT2).

Without caching (BAD):
    for i in range(100):
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")  # 2-3s each time
        world = generate_world(...)
    # Total: 200-300 seconds

With caching (GOOD):
    _model, _tokenizer = None, None

    def load_model():
        global _model, _tokenizer
        if _model is None:  # Only on first call
            _tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            _model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        return _tokenizer, _model

    for i in range(100):
        tokenizer, model = load_model()  # No disk I/O after first call
        world = generate_world(...)
    # Total: ~2-3 seconds (load) + 100 * 0.1s (inference) = ~12 seconds

100x faster!

In the code (lines 80-96 of generate.py):
    _model = None
    _tokenizer = None

    def load_model(model_name: str = "distilgpt2"):
        global _model, _tokenizer
        if _model is None or _tokenizer is None:
            print(f"[INFO] Loading model '{model_name}'...")
            _tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            _model = GPT2LMHeadModel.from_pretrained(model_name)
            _model.eval()
        return _tokenizer, _model

This is loaded in both generate.py (CLI) and api.py
(FastAPI endpoint gets same cached model across requests).
```

---

#### Q8: What would break if you removed the `with torch.no_grad():` block?

**A:**
```
This block (line 112-122 in generate.py) disables gradient computation during inference.

WITHOUT torch.no_grad() (BAD):
    output_ids = model.generate(input_ids, ...)

    PyTorch will:
    - Compute gradients for every operation during inference
    - Store computation graphs for backpropagation
    - Use 2-3x more memory
    - Run 2-3x slower

    Example: DistilGPT2 would use ~500MB memory instead of ~150MB
    And take 20-30 seconds instead of 5-10 seconds

WITH torch.no_grad() (GOOD):
    with torch.no_grad():
        output_ids = model.generate(input_ids, ...)

    PyTorch will:
    - Skip gradient computation (we don't need backprop for inference)
    - Skip storing computation graphs
    - Use minimal memory
    - Run at maximum speed
    - Example: Uses ~150MB, takes 5-10 seconds

Bottom line: torch.no_grad() tells PyTorch
"we're not training, not updating weights, just using the model".
This is essential for efficient inference on any model.

It's like telling a car engine:
"Don't store diagnostic logs, just run the engine" → saves resources.
```

---

### Questions They Might Ask

#### Q9: How would you deploy this to production?

**A:**
```
Approach 1: Cloud Function (Recommended for low traffic)
  - Docker containerize the FastAPI app
  - Deploy to Google Cloud Run / AWS Lambda / Azure Container Instances
  - Scales automatically, pay per request
  - Drawback: Cold starts (3-5 sec loading model)
  - Cost: ~$1-5/month for light usage

Approach 2: Kubernetes Cluster
  - Deploy FastAPI in Docker on K8s
  - Auto-scaling based on requests
  - Add request queuing, load balancing
  - Deployed to Google GKE / AWS EKS
  - Cost: ~$50-200/month minimum
  - Overkill unless you have 1000s of requests/day

Approach 3: Traditional VPS + Gunicorn
  - Deploy FastAPI on a Linux VPS (Hetzner, Linode, DigitalOcean)
  - Run behind Nginx reverse proxy
  - Simple, cheap (~$5-10/month), good for 10s of simultaneous users
  - Manual scaling

Approach 4: Serverless with pre-warming
  - Use Cloud Functions with pre-warmed instances
  - Keep model in memory across invocations
  - Eliminates cold-start problem

My recommendation for THIS project:
  - Start with Approach 3 (VPS + Nginx + Gunicorn)
  - If traffic grows, migrate to Approach 1 (Cloud Run)
  - Kubernetes only when you have 1000+ daily users

Deployment steps:
  1. pip freeze > requirements.txt (pin versions)
  2. Create Dockerfile (Docker: FROM python:3.11, RUN pip install -r...)
  3. Build image: docker build -t rpg-generator .
  4. Push to registry (Docker Hub, GCR, ECR)
  5. Deploy to your platform
  6. Set up CI/CD (GitHub Actions) for auto-deploy on push
```

---

#### Q10: How would you test this system?

**A:**
```
Unit Tests (test_generate.py):
  - Test theme config loading (all 5 themes exist, have all 7 fields)
  - Test prompt builders return non-empty strings
  - Test _clean_text() removes specific artifacts
  - Test seed reproducibility (seed=42 produces same output)
  - Test temperature validation (0.1 ≤ temp ≤ 2.0)

  Example:
    def test_seed_reproducibility():
        world1 = generate_world(theme="fantasy", seed=42)
        world2 = generate_world(theme="fantasy", seed=42)
        assert world1 == world2  # Same seed = same output

Integration Tests (test_api.py):
  - Test GET /generate_world endpoint returns 200
  - Test response matches WorldResponse schema
  - Test invalid theme returns 400
  - Test temperature out of range returns 400

  Example:
    def test_api_generate_world():
        response = client.get("/generate_world?theme=fantasy")
        assert response.status_code == 200
        assert response.json()["theme"] == "fantasy"

Acceptance/Smoke Tests:
  - Generate 100 worlds, check all have 6 fields
  - Manually verify outputs are coherent and on-theme
  - Check generation time < 30 seconds

Quality Tests (manual or automated scoring):
  - Check world names are 2-4 words (run regex)
  - Check descriptions are single sentences (run NER to count periods)
  - Check no stat artifacts like (5), (HP), etc. remain
  - Check diversity: no two worlds are similar

Example test with pytest:
    import pytest
    from generate import generate_world, THEME_CONFIGS

    def test_all_themes_work():
        for theme in THEME_CONFIGS.keys():
            world = generate_world(theme=theme, seed=1)
            assert "World Name:" in world
            assert "Description:" in world
            assert "NPC:" in world
            # ... check all 6 fields

    def test_temperature_affects_output():
        world_low = generate_world(theme="fantasy", temperature=0.3, seed=1)
        world_high = generate_world(theme="fantasy", temperature=1.5, seed=1)
        assert world_low != world_high  # Different temps should produce different output

    def test_seed_reproducibility():
        world1 = generate_world(seed=42)
        world2 = generate_world(seed=42)
        assert world1 == world2

Run tests:
    pytest tests/ -v
```

---

## KEY CONCEPTS & FOCUS AREAS

The recruiter will likely test your understanding of these:

### 1. **Prompt Engineering** ✅ CRITICAL
What it is: Designing input text to a language model to get desired outputs, without changing model weights.

Why it matters: Modern NLP is moving from fine-tuning to prompt engineering (faster, cheaper, more flexible).

In your project:
```python
# This is few-shot prompt engineering:
prompt = """
Examples of short high-fantasy RPG world names:
- Eldenmoor
- Thornveil Reach

One new high-fantasy RPG world name (2-4 words only):
World Name:
"""
# Model learns the pattern and generates a name in the same format
```

**Be ready to explain:**
- Zero-shot vs few-shot vs chain-of-thought prompting
- How examples guide the model
- Why "World Name:" signals where to start generating
- How adjectives (magical, cursed, frozen) prime the model

---

### 2. **Sampling vs Greedy Decoding** ✅ CRITICAL
What it is: Two strategies for selecting the next token at each step.

**Greedy:** Always pick highest probability token
  ```
  Model outputs: [0.6, 0.3, 0.05] for tokens ["the", "a", "an"]
  Greedy: Always pick "the"
  Output: "the the the..." (boring, repetitive)
  ```

**Sampling:** Pick according to probability distribution
  ```
  Same probabilities
  Sampling: Pick "the" 60% of time, "a" 30%, "an" 5%
  Output: "the apple", "a tree", "an orange" (varied, natural)
  ```

In your code (line 116):
```python
do_sample=True  # Use sampling instead of greedy
```

**Be ready to explain:**
- Why sampling is better for creative tasks
- How temperature controls "how much" we sample
- That sampling is probabilistic (non-deterministic without seed)

---

### 3. **Temperature & Probability Distributions** ✅ IMPORTANT
Mathematical insight they might drill:

```
Temperature divides the logits before softmax:

P(next token) = softmax(logits / temperature)

Temperature = 0.3 (low):
  logits = [-2, 0, 1]
  scaled = [-6.67, 0, 3.33]  (more extreme)
  softmax = [0.001, 0.01, 0.989]  (almost deterministic)

Temperature = 1.0 (neutral):
  logits = [-2, 0, 1]
  scaled = [-2, 0, 1]
  softmax = [0.09, 0.24, 0.67]  (natural distribution)

Temperature = 1.5 (high):
  logits = [-2, 0, 1]
  scaled = [-1.33, 0, 0.67]  (more uniform)
  softmax = [0.21, 0.34, 0.45]  (flattened)
```

Lower temp → sharper distribution → deterministic
Higher temp → flatter distribution → creative

---

### 4. **Sequence-to-Sequence Models & Tokenization** ✅ IMPORTANT
How DistilGPT2 works:

```
1. Tokenizer converts text to integers:
   "Retrieve the Crystal" → [24484, 262, 6968]

2. Embedding layer converts to vectors:
   [24484, 262, 6968] → [[0.2, -0.1, ...], [0.5, 0.3, ...], ...]

3. Transformer layers process and summarize:
   Layer 1: Apply attention, combine info across tokens
   Layer 2: Learn richer representations
   ...
   Layer 6: Output contextualized vectors

4. Language modeling head predicts next token:
   Input: "Retrieve the"
   Output: Probability distribution over 50,000 tokens
   argmax → "Crystal" (or sample)
```

DistilGPT2 has 6 layers (vs GPT-2's 12 layers), making it faster.

---

### 5. **Generation Parameters & Their Effects** ✅ IMPORTANT

| Parameter | Effect | Use Case |
|-----------|--------|----------|
| `temperature` | Controls distribution sharpness | 0.7-0.8 for most tasks |
| `top_k` | Limit to top K tokens | Prevents random gibberish |
| `top_p` | Nucleus sampling (cumsum < p) | Adaptive token limits |
| `repetition_penalty` | Penalize repeated tokens | Prevents loops "the the the" |
| `max_new_tokens` | Token budget | Controls output length |
| `seed` | RNG seed | Reproducibility |

Your code uses all of them (lines 113-122):
```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    do_sample=True,
    temperature=max(temperature, 0.1),
    top_p=0.92,
    top_k=50,
    repetition_penalty=1.3,
    pad_token_id=50256,
)
```

**Be ready to:**
- Explain why you chose these specific values
- Discuss tradeoffs (top_k vs top_p, temperature vs repetition_penalty)
- Predict what would happen if you changed them

---

### 6. **Caching & Performance** ✅ IMPORTANT
Your caching strategy (lines 80-96) is professional:

```python
_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        # Load from disk (slow, 2-3 seconds)
        _tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        _model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return _tokenizer, _model
```

**Benefits:**
- First generation: 2s (load) + 5s (inference) = 7s total
- Subsequent generations: 5s (no load overhead)
- In a FastAPI server: All requests share the same model in memory

**Be ready to:**
- Explain why this is necessary (model loading is expensive)
- Discuss alternatives (singleton pattern, dependency injection)
- Explain memory implications (model in RAM permanently)

---

### 7. **API Design & REST Best Practices** ✅ IMPORTANT
In api.py, you follow REST principles:

```python
@app.get("/generate_world")
def generate_world_endpoint(
    theme: str = Query(..., enum=[...]),  # Validated enum
    temperature: float = Query(..., ge=0.1, le=2.0),  # Range validation
    max_length: int = Query(..., ge=50, le=500),
    seed: int | None = Query(None),  # Optional
):
    # Handle errors gracefully
    try:
        world = generate_world(...)
        return WorldResponse(...)  # Structured response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # Client error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")  # Server error
```

**REST principles shown:**
- GET for read-only operations (not POST, since no server state changes)
- Query parameters for inputs
- Enum validation (theme ∈ {fantasy, dark_fantasy, ...})
- Range validation (0.1 ≤ temperature ≤ 2.0)
- Proper HTTP status codes (200 success, 400 client error, 500 server error)
- Documented response schema (Pydantic BaseModel → Swagger docs)

---

### 8. **Reproducibility & Seeding** ✅ IMPORTANT
Your seed handling (lines 266-268):

```python
if seed is not None:
    torch.manual_seed(seed)
    random.seed(seed)
```

**Why:**
- Without seed: Every run generates a different world (useful for players)
- With seed: Same seed always produces same world (useful for testing, demos)
- Both PyTorch AND Python random must be seeded (model outputs + potential python randomness)

**Be ready to:**
- Explain why you seed both torch and random
- Discuss reproducibility challenges (floating point precision, hardware differences)
- Show how seeds are used in experiments.py (seed=1 for all tests)

---

### 9. **Error Handling & User Experience** ✅ IMPORTANT
Your code handles errors gracefully:

```python
# CLI (generate.py):
try:
    world = generate_world(...)
    print(world)
except Exception as e:
    print(f"Error: {e}")

# API (api.py):
try:
    world_text = generate_world(...)
    return WorldResponse(...)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
```

**Good practices:**
- Validate inputs early (theme ∈ THEME_CONFIGS)
- Catch specific exceptions (ValueError for validation, broad Exception for bugs)
- Return meaningful error messages
- Use appropriate HTTP status codes (400 for client errors, 500 for server errors)

---

### 10. **Extensibility & Maintainability** ✅ IMPORTANT
Your code is easy to extend:

**Adding a new theme:** Just one dict entry (lines 26-72):
```python
THEME_CONFIGS = {
    ...existing themes...,
    "steampunk": {
        "adjectives": "industrial, clockwork, steam-powered",
        "setting": "a world of gears and brass",
        "name_example": "Cogsworth Citadel",
        "npc_prefix": "a rogue inventor or airship captain",
        "quest_type": "retrieve lost blueprints or fix a broken machine",
        "weapon_type": "steam-powered rifle or gear-infused sword",
        "lore_hook": "an ancient civilization powered by steam engines",
    }
}
```

No code changes needed! Just add config.

**Why this is good:**
- Themes are data, not hardcoded logic
- New themes don't require testing existing code
- Non-programmers could add themes (if given the template)

---

## INTERVIEW TIPS

### Before the Interview

1. **Rehearse your elevator pitch:**
   - Practice explaining the project in 30 seconds, 2 minutes, 5 minutes
   - Record yourself and listen back
   - Aim for: what problem it solves, how it works, why you're proud of it

2. **Prepare for the deep dive:**
   - Know every file line by line
   - Be able to explain any function's purpose
   - Have examples ready: "If you look at line 116..."
   - Don't just memorize; understand the reasoning

3. **Prepare specific talking points:**
   - Why DistilGPT2 over larger models
   - Why few-shot prompting over fine-tuning
   - Why FastAPI over Flask
   - Why you tested with experiments.py
   - What you'd do differently with more time/resources

4. **Practice answering these questions:**
   - "Walk me through the data flow when a user generates a world"
   - "Why did you choose [technology]?"
   - "How would you improve this?"
   - "What would break if you removed [code]?"
   - "How would you deploy this?"
   - "What would you test?"

5. **Prepare to live-code:**
   - They might ask you to extend the project
   - Example: "Can you add a theme for medieval worlds?"
   - Example: "Can you add logging to track generation time?"
   - Solution: Show you understand the architecture by confidently making changes

6. **Have a mental model:**
   - Be able to draw/sketch the architecture on whiteboard
   - Understand the control flow (CLI → generate_world → model → output)
   - Know where each component fits

---

### During the Interview

1. **Start with confidence:**
   - "This is a procedural RPG world generator using language models."
   - Show you understand what you built, not just that you built it

2. **Explain, don't just list:**
   - Bad: "I used PyTorch, HuggingFace, FastAPI, argparse"
   - Good: "I used PyTorch as the ML backend because it's intuitive and has great NLP integration via HuggingFace. I used FastAPI because it auto-generates API documentation and validates inputs with Pydantic."

3. **Use the code as reference:**
   - "If you look at line 116, I have do_sample=True..."
   - "In the _build_name_prompt function (lines 180-188)..."
   - This shows you know your code intimately

4. **Discuss tradeoffs:**
   - "I chose DistilGPT2 for CPU compatibility, but there's a quality tradeoff vs GPT-2 Medium"
   - "I call generate_world 6 times (once per field) which is slower than generating all at once, but it gives me better control"
   - Sophisticated developers discuss tradeoffs

5. **If you don't know, say so:**
   - "That's a great question. Let me think... I haven't considered that angle before, but I could see [approach]"
   - Never make up technical details
   - Showing you think through problems is valuable

6. **Ask clarifying questions:**
   - "When you ask about deployment, are you thinking cloud (AWS, GCP) or on-premises?"
   - "For testing, do you want unit tests, integration tests, or quality metrics?"
   - Shows you think systematically

7. **Be enthusiastic but honest:**
   - "I'm really proud of the experiment design — it shows how different parameters affect output"
   - "If I had more time, I'd add RAG to ground generations in consistent lore"
   - Genuine enthusiasm is more attractive than fake hype

---

### Red Flags to Avoid

❌ **Don't claim to know things you don't:**
   - "I'm an expert in LLMs" (unless you really are)
   - "DistilGPT2 is the best model" (context matters)

❌ **Don't over-engineer your explanations:**
   - Use examples but keep them focused
   - Don't explain everything 5 times

❌ **Don't shy away from limitations:**
   - "The sci_fi theme needed more work to stay on-brand"
   - "Post-processing with regex isn't perfect"
   - Acknowledging limitations shows maturity

❌ **Don't be defensive:**
   - If they criticize your choice (e.g., "Why not use GPT-2 Medium?")
   - Say: "Great point. I chose DistilGPT2 for speed, but GPT-2 Medium would give [benefits]"
   - Show you've thought about it

❌ **Don't forget about the experiments:**
   - "I ran 10 experiments and documented findings"
   - This is actually impressive — most projects don't do this
   - Talk about it! It shows rigor.

---

### If They Ask About Weaknesses

**Q: "What don't you like about your implementation?"**

Good answers:
- "The sci_fi theme quality is lower because the model hasn't seen much sci-fi RPG text during pretraining. Fine-tuning would fix this."
- "My post-processing with regex is brittle. Using constrained decoding (outlines library) would be more robust."
- "I cache the model globally, which means it stays in RAM. For a serverless deployment, I'd need to rethink this."
- "I call generate_world 6 times (once per field), which means I can't ensure consistency between fields (e.g., NPC might contradict the quest)."

Bad answers:
- "Nothing, it's perfect!" (Show naivety)
- "I don't know, I didn't think about that." (Lack of reflection)
- Criticizing fundamentals you can't fix

---

### If They Ask You to Extend It

**Scenario: "Can you add a JSON export feature?"**

Approach:
1. Clarify: "You want the world output in JSON format instead of text?"
2. Sketch: "I'd modify _format_world() to return a dict instead of formatted string"
3. Code:
   ```python
   def export_world_json(world_name, description, npc, quest, reward, lore):
       return {
           "world_name": world_name,
           "description": description,
           "npc": npc,
           "quest": quest,
           "reward_item": reward,
           "lore": lore,
       }
   ```
4. Test: "I'd test this returns valid JSON, handles special characters, etc."

This shows you understand the codebase and can extend it.

---

### Post-Interview

- **Send a thank-you message** within 24 hours
- **Reference specific things you discussed:**
  - "Thank you for the great feedback onusing few-shot prompting..."
  - "I enjoyed discussing how to improve the sci_fi theme with fine-tuning..."
- **Share the project if they ask:**
  - Have the GitHub link ready
  - Make sure README is polished
  - Include a RESULTS.md with sample outputs

---

## FINAL CHECKLIST FOR INTERVIEW

✅ Rehearse 30-second pitch
✅ Know every line of code
✅ Understand each technology choice
✅ Prepare 3 alternatives for each major choice
✅ Know all parameters and their effects
✅ Practice explaining data flow
✅ Discuss tradeoffs confidently
✅ Have GitHub repo clean and documented
✅ Understand experiments and findings
✅ Know what you'd do with more time/resources
✅ Practice answering "walk me through..." questions
✅ Prepare to live-code small extensions
✅ Have questions prepared for them
✅ Know your weaknesses and how to address them
✅ Be ready to discuss deployment, testing, scaling

---

## CONCLUSION

This project demonstrates:
1. **ML Knowledge:** Understanding of models, prompts, sampling, tokens
2. **Software Engineering:** Clean code, API design, CLI tools, error handling
3. **Problem Solving:** Using lightweight models for CPU-friendly inference
4. **Experimental Thinking:** Validating choices with controlled tests
5. **Communication:** Well-documented code and methodology

The recruiter gave you this task to see if you could build something end-to-end. You did. Now in the interview, show them you understand *why* every choice was made.

Good luck! 🚀

---

*Last updated: March 2026*
*For questions, review the code comments or re-read this guide's explanations.*
