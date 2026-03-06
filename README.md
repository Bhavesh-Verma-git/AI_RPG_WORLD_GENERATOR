# AI RPG World Generator 🌍⚔️

> **Generate procedural RPG worlds and game assets using a lightweight AI language model.**
> Powered by DistilGPT2 · CPU-friendly · No training required · GitHub-ready

---

## 📋 Project Overview

This project is a **procedural RPG content generator** that uses a pretrained language model (DistilGPT2) to generate complete RPG world scenarios through **prompt engineering**.

Instead of generating a single game asset, the system produces a **rich, structured world scenario** containing:

| Field | Example |
|-------|---------|
| World Name | Frosthold Valley |
| Description | A frozen valley where ancient ice magic lingers |
| NPC | Eldric the Keeper — a guardian with frost in his veins |
| Quest | Retrieve the Crystal of Permafrost |
| Reward Item | Stormfang Sword — stuns enemies with lightning |
| Lore | The valley was a mage kingdom lost overnight |

The generator supports **multiple world themes**, **configurable generation parameters**, and includes a **FastAPI backend** to demonstrate integration into a game backend service.

---

## 🤖 Model: DistilGPT2

| Property | Detail |
|----------|--------|
| Model | `distilgpt2` (HuggingFace) |
| Parameters | 82 million |
| Architecture | 6-layer distilled transformer (GPT-2 small) |
| Size (~download) | ~80 MB |
| Hardware | CPU-only ✅ |
| Training | Pretrained on WebText — no fine-tuning required |

### Why DistilGPT2?

- **Lightweight**: 82M parameters, runs in seconds on a laptop CPU.
- **No GPU needed**: Runs entirely on CPU without sacrificing coherent output.
- **Pretrained**: Comes with broad knowledge of English, including fantasy vocabulary, names, and storytelling patterns.
- **No training required**: We use **prompt engineering** to guide outputs — zero data collection or training time.
- **HuggingFace integration**: One-line model loading with a well-documented API.

This makes it ideal for a demo/internship project that should be **fully runnable locally** without cloud compute.

---

## ⚙️ How Prompt Engineering Works

Prompt engineering is the technique of **crafting the input text to a language model** so that it generates what you want — without modifying the model's weights.

### Our approach — Few-Shot Prompting

We provide the model with **one complete example world** inside the prompt, then ask it to generate another:

```
Create a high-fantasy RPG world scenario.

World Name: Eldenmoor
Description: A lush magical land filled with ancient elven forests.
NPC: Syla the Wandering Mage — a scholar who seeks forgotten spells.
Quest: Recover the Shattered Staff of Aeons from the Temple of Echoes.
Reward Item: Starblade Dagger — a silver blade that glows under moonlight.
Lore: Legends say Eldenmoor was split apart by a war between gods.

Create another high-fantasy RPG world scenario.

World Name:   ← MODEL COMPLETES FROM HERE
```

The model **learns the pattern** from the example and continues it, generating the next world in the same structured format. This is called **few-shot prompting** — no gradient-based training, just clever input design.

---

## 🗺️ World Themes

| Theme | Description |
|-------|-------------|
| `fantasy` | Classic high-fantasy — elves, magic, ancient temples |
| `dark_fantasy` | Cursed lands, demons, fallen heroes, undead |
| `desert` | Ancient civilisations, sandstorms, lost oases |
| `arctic` | Ice magic, frozen valleys, permafrost ruins |
| `sci_fi` | Orbital stations, rogue AIs, corporate dystopias |

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
python --version   # Python 3.10 or 3.11 recommended
```

### 2. Clone & Set Up

```bash
git clone https://github.com/YOUR_USERNAME/AI-RPG-WORLD-GENERATOR.git
cd AI-RPG-WORLD-GENERATOR

# Create and activate virtual environment
python -m venv rpg_env
rpg_env\Scripts\activate          # Windows
# source rpg_env/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

> ⚠️ The first run downloads DistilGPT2 (~80 MB). Subsequent runs use the local cache.

### 3. Generate a World

```bash
python generate.py --theme fantasy
python generate.py --theme dark_fantasy
python generate.py --theme arctic --temperature 0.9 --max_length 300
python generate.py --theme desert --seed 42          # Reproducible output
python generate.py --theme sci_fi --no-save          # Don't save to file
```

---

## 🖥️ CLI Reference

```
usage: generate.py [-h] [--theme THEME] [--temperature TEMPERATURE]
                   [--max_length MAX_LENGTH] [--seed SEED] [--no-save]

Options:
  --theme         World theme: fantasy, dark_fantasy, desert, arctic, sci_fi
  --temperature   Creativity (0.1 = structured, 1.5 = creative)  [default: 0.8]
  --max_length    Number of new tokens to generate               [default: 250]
  --seed          Integer seed for reproducibility               [default: random]
  --no-save       Skip saving output to outputs/
```

---

## 🌐 FastAPI Backend (Optional)

### Start the server

```bash
uvicorn api:app --reload
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/generate_world` | Generate a world scenario |
| GET | `/themes` | List available themes |

### Example request

```
GET http://127.0.0.1:8000/generate_world?theme=dark_fantasy&temperature=0.9&max_length=200
```

### Swagger UI

Visit `http://127.0.0.1:8000/docs` — interactive API documentation auto-generated by FastAPI.

---

## 🔬 Parameter Experiments

The `experiments.py` script runs a battery of controlled tests varying temperature, max_length, and theme.

```bash
python experiments.py
```

Results are saved to `outputs/experiments/`. See [`experiments.md`](experiments.md) for a full write-up of findings.

---

## 📁 Project Structure

```
AI-RPG-WORLD-GENERATOR/
│
├── generate.py          ← Core world generator + argparse CLI
├── api.py               ← FastAPI backend
├── experiments.py       ← Parameter sweep runner
├── experiments.md       ← Experiment write-up & observations
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── .gitignore
│
└── outputs/
    ├── world_example_1.txt   ← Sample: Fantasy world
    ├── world_example_2.txt   ← Sample: Dark fantasy world
    └── world_example_3.txt   ← Sample: Desert world
```

---

## 🧪 Understanding Parameters

| Parameter | Effect |
|-----------|--------|
| **temperature** | Scales the probability distribution over next tokens. Low (0.3) = picks high-probability tokens → structured output. High (1.2) = flattens distribution → creative but less predictable. |
| **max_length** | Total tokens in output. More tokens = richer world detail but slower generation. |
| **top_p (0.92)** | Nucleus sampling — only samples from the smallest set of tokens whose cumulative probability exceeds 0.92. Prevents very unlikely (weird) tokens. |
| **top_k (50)** | Limits sampling pool to the 50 most probable tokens at each step. |
| **repetition_penalty (1.2)** | Penalises tokens that were already generated, reducing repetitive loops. |
| **seed** | Fixes the random number generator for reproducible outputs. |

---

## 🔮 Future Improvements

With more time and compute, the system could be significantly enhanced:

1. **Fine-tune on RPG datasets**: The most impactful improvement would be to fine-tune DistilGPT2 (or a larger model like GPT-2 medium, LLaMA-3, or Mistral-7B) on domain-specific datasets containing fantasy lore, NPC dialogue, quest descriptions, and item flavour text. Sources like D&D sourcebooks, fantasy wikis, and game script dumps would dramatically improve the domain relevance and structural consistency of generated worlds.

2. **Game engine integration**: The generator API could be connected to a **Unity or Unreal Engine** plugin, enabling real-time procedural world generation during gameplay. Each new area the player enters could have a unique AI-generated name, backstory, and quest.

3. **Structured generation with Grammars / Constrained Decoding**: Use libraries like `outlines` or `guidance` to enforce the output to follow a strict JSON schema, guaranteeing that every field (Name, NPC, Quest, etc.) is always populated without relying on post-processing heuristics.

4. **Evaluation metrics**: Implement automated quality metrics such as:
   - **Perplexity** — measures how surprising the generated text is relative to the model's own distribution.
   - **BLEU / ROUGE** — measures overlap with reference RPG worlds.
   - **Diversity metrics** — ensures generated worlds don't repeat themes verbatim.

5. **Web UI**: Build an interactive front-end (e.g., with React + FastAPI) where users can select a theme, adjust sliders for temperature and max_length, and see the generated world rendered in a styled game-like interface.

6. **Retrieval-Augmented Generation (RAG)**: Augment the generator with a vector database of RPG lore snippets. Before generation, retrieve semantically relevant lore and inject it into the prompt, grounding the output in consistent world-building facts.

---

## 📖 How to Explain This Project to a Recruiter

### How does the language model generate text?

DistilGPT2 is a **transformer-based language model** trained to predict the next word in a sequence. Given the tokens in the prompt, the model outputs a probability distribution over the entire vocabulary for the next token. We sample from this distribution, append the chosen token, and repeat — producing text token by token. All of this happens through a **forward pass** of the neural network; no training occurs at inference time.

### How does prompt engineering control output structure?

By providing the model with a **few-shot example** of a complete RPG world in the prompt, we show it the exact format we expect. The model has learned during pretraining that text often continues in the same pattern as what it has seen. Our structured prompt exploits this to produce consistently formatted outputs — no fine-tuning required.

### How does temperature affect creativity?

Temperature **scales the logits** (raw scores) before the softmax that converts them into probabilities. Low temperature (e.g., 0.3) sharpens the distribution — the model almost always picks the most probable token, producing conservative, predictable text. High temperature (e.g., 1.2) flattens the distribution — less probable tokens get a real chance, producing more surprising, creative, but sometimes less coherent text.

### Why were parameter experiments performed?

Different games need different styles of content. A dungeon generator wants **structured, predictable** descriptions (low temperature). A random encounter generator benefits from **surprising, creative** events (high temperature). By systematically varying parameters and recording outputs, we can recommend the best settings for each use case — this is the same principle as hyperparameter tuning in supervised learning.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | >=4.40.0 | HuggingFace model loading & inference |
| `torch` | >=2.2.0 | PyTorch deep learning backend |
| `fastapi` | >=0.111.0 | REST API framework |
| `uvicorn` | >=0.29.0 | ASGI server for FastAPI |
| `pydantic` | >=2.7.0 | Data validation (FastAPI dependency) |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using HuggingFace Transformers and FastAPI.*
