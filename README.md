# Radio Astronomy in the Era of Vision–Language Models

Code for the paper:

**Radio Astronomy in the Era of Vision-Language Models: Prompt Sensitivity and Adaptation**  
Mariia Drozdova, Erica Lastufka, Vitaliy Kinakh, Taras Holotyak, Daniel Schaerer, Slava Voloshynovskiy  
arXiv: [2509.02615](https://arxiv.org/abs/2509.02615)

This repo consists of two parts:
1) **Evaluate pretrained VLMs** with different prompting strategies (zero-/few-shot, diagram) using `evaluate_test.py`.  
2) **Train** a lightweight **LoRA** adaptation of Qwen2-VL using `train_sft.py`.

---

## Repo structure
```
application_VLM_to_astronomy/
├─ evaluate_test.py       # main evaluation script (prompted VLMs, zero/few-shot)
├─ train_sft.py           # LoRA fine-tuning for Qwen2-VL
├─ frcartoon.png          # schematic FR-I/FR-II diagram used in "theory-shot"
└─ src/
   ├─ all_predict.py      # model adapters + prompting logic
   ├─ data.py             # MiraBest data loading helpers (auto-download)
   ├─ retriever.py        # CLIP-based neighbor retrieval for few-shot prompts
   └─ utils.py            # parsing + metrics helpers
```

---

## Installation

```bash
# Python 3.10+ recommended
# Create a new conda env
conda create -n vlm_astronomy python=3.10 -y
conda activate vlm_astronomy

# Install dependencies
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
pip install -U transformers accelerate peft pillow datasets scikit-learn tenacity tqdm wandb
pip install -U google-genai openai  # Gemini & GPT APIs
```

### API keys (Gemini & GPT)
If you want to use proprietary models, you should create `environment.py` in the project root:

```python
# environment.py
GEMINI_FLASH_2_API_KEY = "YOUR_GEMINI_FLASH_2_API_KEY"
GPT_API_KEY = "YOUR_OPENAI_GPT_API_KEY"
```

Follow instructions from Gemini and OpenAI to get the keys.

---

## Data

MiraBest (confident split) is auto-downloaded to `./data/mirabest` on first run.  
No extra setup needed.

---

## 1) Evaluation: prompted VLMs

`evaluate_test.py` runs **zero-shot**, **diagram (“theory-shot”)**, and **few-shot** regimes.  
Prompts are bundled inside and selected via `--index`.

**Common flags**
- `--model_id` one of:
  - Open models: `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`, `llava-hf/llava-v1.6-mistral-7b-hf`, etc.
- `--sampling_regime` ∈ `{zero-shot, theory-shot, few-shot, few-shot-most-closest-neighbors, few-shot-balanced-closest-neighbors}`
- `--index` prompt template index (per regime)
- `--image_first` place image before text in the prompt
- `--temperature` decoding temperature (float)
- `--n_reps` repeat evaluations (default 10)
- `--use_wandb` log to Weights & Biases (project: `mira-best-test`)

### Zero-shot (text-only)
```bash
python evaluate_test.py \
  --sampling_regime zero-shot \
  --index 0 \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --temperature 0 \
  --n_reps 1
```

### Diagram / “theory-shot” (uses `frcartoon.png`)
```bash
python evaluate_test.py \
  --sampling_regime theory-shot \
  --index 0 \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --temperature 0 \
  --n_reps 1
```

### Few-shot — fixed exemplars
Uses `--fixed_indexes` (defaults provided in the script).
```bash
python evaluate_test.py \
  --sampling_regime few-shot \
  --index 0 \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --fixed_indexes 105 102 3 100 \
  --temperature 0
```

### Few-shot — kNN retrieved exemplars
Retrieve nearest neighbors with CLIP (global or balanced per class).
```bash
# Global top-k
python evaluate_test.py \
  --sampling_regime few-shot-most-closest-neighbors \
  --index 1 \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --nearest_neighbors 5

# Balanced (use an even k)
python evaluate_test.py \
  --sampling_regime few-shot-balanced-closest-neighbors \
  --index 1 \
  --model_id gemini-2.5-flash \
  --nearest_neighbors 6
```

> Tips:
> - Toggle `--image_first` and `--temperature` to probe prompt sensitivity.
> - Set `--device cuda:0` to select a GPU.

---

## 2) Training: LoRA adaptation of Qwen2-VL

`train_sft.py` fine-tunes **~15M** LoRA parameters on MiraBest.  
(Training prompt matches a zero-shot template to avoid train/test prompt mismatch.)

**Minimal run (Qwen2-VL-7B)**
```bash
python train_sft.py \
  --model_id Qwen/Qwen2-VL-7B-Instruct \
  --dataset_name mirabest \
  --epochs 100 \
  --batch_size 4 \
  --lora_r 16 \
  --lora_alpha 64 \
  --lora_dropout 0.3 \
  --lr 5e-5 \
  --use_wandb
```

**Notes**
- Checkpoints are written to a temporary `./ckpt_<RUNID>` during training.
- By default, the script cleans up the checkpoint directory at the end (adjust if you want to persist).

---

## Reproducing paper experiments

- Use the **provided prompt indices** (`--index`) for each `--sampling_regime` to reproduce reported variants.  
- The script already contains the **Text**, **Diagram**, **Fixed-Imgs**, **kNN-Imgs**, and **kNN-Balanced** templates and logic.

---

## Citation

```bibtex
@misc{drozdova2025radioastronomyeravisionlanguage,
  title={Radio Astronomy in the Era of Vision-Language Models: Prompt Sensitivity and Adaptation},
  author={Mariia Drozdova and Erica Lastufka and Vitaliy Kinakh and Taras Holotyak and Daniel Schaerer and Slava Voloshynovskiy},
  year={2025},
  eprint={2509.02615},
  archivePrefix={arXiv},
  primaryClass={astro-ph.IM},
  url={https://arxiv.org/abs/2509.02615},
}
```
