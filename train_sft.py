import argparse
import logging
import random
import shutil, uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

from src.data import (
    load_gmnist_data,
    load_mirabest_data,
    load_radiogalaxynet_dataset,
)
from qwen_vl_utils import process_vision_info
from src.utils import extract_last_class, report_results
def parse_args():
    parser = argparse.ArgumentParser(description="VLM SFT training")

    parser.add_argument("--use_wandb", action="store_true", help="log to Weights & Biases")

    parser.add_argument("--device", type=str, default="cuda:0", help="torch device")
    parser.add_argument("--extra_line", type=str, default="tmp", help="suffix for save path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--part_to_train", type=str, default="LoRA")
    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["linear","cosine","constant"])
    parser.add_argument("--dataset_name", type=str, default="mirabest",
                        choices=["mirabest", "gmnist", "radiogalaxynetdataset"])

    parser.add_argument("--val_size", type=int, default=70)
    parser.add_argument("--train_size", type=int, default=656)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--lora_dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_id):
    min_pixels = 224*224
    max_pixels = 2048*2048

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map=None,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2",
    ).to(device)
    
    processor = Qwen2VLProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "left"
    return model, processor

def zero_shot_predict(
        model, processor, test_dataset,
        query_idx,
        device="cuda",
        print_text=False,
        temperature=0.0,
        reps=1,
        system_message="",
        query_text="",
        max_new_tokens=512,
        all_labels=["FR-I", "FR-II"]

):
    # assert system_message != "" and query_text != ""
    imgq = test_dataset[query_idx]["image"]

    msgs = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
    ]
    # msgs.extend([
    #    {"role":"user",      "content":[{"type":"image","image":imgq}, {"type":"text","text":query_text}]},
    # ])
    msgs.extend([
        {"role": "user", "content": [{"type": "text", "text": query_text}, {"type": "image", "image": imgq}]},
    ])

    text_input = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)

    if print_text:
        print("Prompt:")
        print(text_input)

    inputs = processor(text=[text_input], images=image_inputs, return_tensors="pt").to(device)

    answers = Counter()
    raw_answers = []
    out_ids = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             pad_token_id=processor.tokenizer.pad_token_id,
                             num_return_sequences=reps,
                             )[:, inputs.input_ids.shape[-1]:]
    raw_decoded = processor.batch_decode(out_ids, skip_special_tokens=True)
    for raw in raw_decoded:
        raw = raw.strip()
        if print_text:
            print("Answer:")
            print(raw)
        raw_answers.append(raw)
        answer = extract_last_class(raw, all_labels=all_labels)
        answers[answer] += 1
    if print_text:
        print(answers)

    final_answer = answers.most_common(1)[0][0]

    res = {"final_answer": final_answer, "answers": answers, "raw_answers": raw_answers}

    return res


def evaluate_score(
        model, processor,
        test_dataset,
        system_message, query_text,
):
    test_indexes = range(len(test_dataset))
    y_true_majority = []
    y_pred_majority = []
    print_text = True

    for i in tqdm(test_indexes, desc="Gathering preds"):
        gt = test_dataset[i]["label"]  # e.g. "FR-I" or "FR-II"
        ans = zero_shot_predict(
            model, processor,
            test_dataset,
            i,
            device=device, print_text=print_text, reps=1, temperature=0,
            max_new_tokens=200, system_message=system_message, query_text=query_text,
            all_labels=all_labels
        )
        if i == 0:
            log.info(ans)
            print_text = False
        pred = ans["final_answer"]
        if pred is None:
            pred = ""
        if pred.lower().startswith("answer"):
            pred = pred.split(":", 1)[1].strip()
        y_true_majority.append(gt)
        y_pred_majority.append(pred)

    score = report_results(y_true_majority, y_pred_majority, True, all_labels=all_labels)
    return score


class RadioGalaxyDataset(Dataset):
    def __init__(self, hf_dataset, processor,):
        self.ds = hf_dataset
        self.proc = processor
        self.system_msg = system_message
        self.user_msg = query_text

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex['image']
        label = ex['label']

        # Build prompt
        system_prompt = f"<|im_start|>system\n{self.system_msg}\n<|im_end|>\n"
        user_prompt = (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{self.user_msg}\n"
            "<|im_end|>\n"
        )
        assistant_prompt = "<|im_start|>assistant\n"
        prompt = system_prompt + user_prompt + assistant_prompt

        # Process text+image
        out = self.proc(
            text=prompt,
            images=img,
            return_tensors="pt",
            truncation=True,
        )
        input_ids = out.input_ids.squeeze(0)
        attention_mask = out.attention_mask.squeeze(0)
        pixel_values = out.pixel_values.squeeze(0)
        image_grid_thw = out.image_grid_thw.squeeze(0)

        # Tokenize label
        target = label + self.proc.tokenizer.eos_token
        target_ids = self.proc.tokenizer(
            target,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        # Create labels array
        labels = torch.full(
            (input_ids.size(0) + target_ids.size(0),),
            -100,
            dtype=torch.long,
        )
        labels[input_ids.size(0):] = target_ids

        # Concatenate for LM
        input_ids = torch.cat([input_ids, target_ids], dim=0)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(target_ids)
        ], dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'labels': labels
        }


def _left_pad(seqs, pad_value):
    length = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), length), pad_value, dtype=seqs[0].dtype, device=seqs[0].device)
    for i, s in enumerate(seqs):
        out[i, -s.size(0):] = s
    return out


def collate_fn(batch):
    pad_id = processor.tokenizer.pad_token_id
    input_ids = _left_pad([b['input_ids'] for b in batch], pad_id)
    attention_mask = _left_pad([b['attention_mask'] for b in batch], 0)
    labels = _left_pad([b['labels'] for b in batch], -100)

    pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0)
    image_grid_thw = torch.stack([b['image_grid_thw'] for b in batch], dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'labels': labels
    }

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

args = parse_args()

device = args.device
use_wandb = args.use_wandb
extra_line = args.extra_line
model_id = args.model_id
val_size = args.val_size
train_size = args.train_size
epochs = args.epochs
part_to_train = args.part_to_train
batch_size = args.batch_size
lora_r = args.lora_r
lora_alpha = args.lora_alpha
lora_dropout = args.lora_dropout
lr = args.lr
lr_schedule = args.lr_schedule
dataset_name = args.dataset_name
seed = args.seed
num_workers = args.num_workers

set_seed(seed)

lora_layers = [
            "attn.qkv",
            "attn.proj",
            "q_proj",
            "v_proj",
            "mlp.fc1",      # vision‐side FFN
            "mlp.fc2",
        ]

system_message = (
            "You are an expert radio galaxy classifier. "
            "FR-I: bright core, gradually fading jets. "
            "FR-II: faint core, jets end in hotspots. "
            "Consider core brightness and jet termination."
            )
query_text = "Respond only FR-I or FR-II."

if dataset_name == "radiogalaxynetdataset":
    system_message = (
        "You are an expert radio galaxy classifier. "
        "Classify each source based on its morphology:\n"
        "- FR-I: Bright central core, jets gradually fade, edge-darkened.\n"
        "- FR-II: Faint core, jets terminate in bright hotspots, edge-brightened.\n"
        "- FR-X: Uncertain cases where morphology is ambiguous between FR-I and FR-II.\n"
        "- R: Barely resolved sources with only one visible jet peak outside the central component.\n"
        "Consider both the core brightness and the jet termination structure when deciding. "
        "Respond strictly with one of: FR-I, FR-II, FR-X, or R."
    )
    query_text = "Classify the source into FR-I, FR-II, FR-X, or R."
if dataset_name == "gmnist":
    system_message = (
        "You are an expert galaxy morphologist working with 64×64 grayscale DECaLS cutouts (GalaxyMNIST). "
        "Classify each image into exactly one of: edge_on_disk, smooth_cigar, smooth_round, unbarred_spiral.\n"
        "\n"
        "Definitions:\n"
        "- edge_on_disk: very thin, highly elongated disk seen edge-on; strong axial ratio; possible central bulge and/or dust lane; "
        "  no visible spiral arms due to edge-on view.\n"
        "- smooth_cigar: smooth, elongated elliptical (no disk/arms); light profile declines smoothly; "
        "  lacks thin midplane or dust lane; thicker appearance than an edge-on disk.\n"
        "- smooth_round: smooth, nearly circular elliptical/lenticular; no arms, bar, or dust lane.\n"
        "- unbarred_spiral: disk galaxy with visible spiral structure or clumpy arms; no strong central bar; "
        "  not extremely edge-on.\n"
        "\n"
        "Consider axial ratio and thickness, presence/absence of a thin midplane or dust lane, smoothness vs clumpiness, "
        "and any spiral features. Ignore foreground stars, borders, and noise. Be decisive."
    )

    query_text = "Respond only with one label: edge_on_disk, smooth_cigar, smooth_round, or unbarred_spiral."


train_system_message = system_message
train_query_text = query_text
examples_message, query_text_example, summary_after_examples_text = "", "", ""

if use_wandb:
    import wandb
    wandb.init(
        project=dataset_name+"-sft-VLM",
        config={
            "device": device,
            "extra_line": extra_line,
            "model_id": model_id,
            "val_size": val_size,
            "train_size": train_size,
            "epochs": epochs,
            "part_to_train": part_to_train,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_layers": lora_layers,
            "batch_size": batch_size,
            "lr": lr,
            "system_message": system_message,
            "query_text": query_text,
            "train_system_message": train_system_message,
            "train_query_text": train_query_text,
            "lr_schedule": lr_schedule,
            "dataset_name" : dataset_name
        }
    )

assert model_id in [
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]

model, processor = load_model(model_id)

# Attach adapter
train_dataset, test_dataset = load_mirabest_data()
all_labels = ["FR-I", "FR-II"]
if dataset_name == "gmnist":
    train_dataset, test_dataset = load_gmnist_data()
    all_labels = ["edge_on_disk", "smooth_cigar", "smooth_round", "unbarred_spiral"]


train_subset = train_dataset
if dataset_name == "radiogalaxynetdataset":
    train_subset, val_subset, test_dataset = load_radiogalaxynet_dataset()
    all_labels=["FR-I", "FR-II", "FR-X", "R",]

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"{trainable_params}/{total_params}")
losses = []

if part_to_train == "LoRA":
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_layers,
    )
    model = get_peft_model(model, lora_config).to(device)
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Before vision off {trainable_params}/{total_params}")

elif part_to_train == "vision_only":
    # Freeze all parameters, then unfreeze only a small part of the vision encoder (e.g., patch embed + last 2 blocks + merger)
    for p in model.parameters():
        p.requires_grad = False
    
    # Vision submodules
    vision = model.model.visual
    num_total_blocks = len(vision.blocks)
    num_unfreeze = 4  # adjust: how many of the last blocks to train
    
    # Unfreeze patch embedding
    for p in vision.patch_embed.parameters():
        p.requires_grad = True
    
    # Unfreeze the last `num_unfreeze` vision blocks
    for idx, block in enumerate(vision.blocks):
        if idx >= num_total_blocks - num_unfreeze:
            for p in block.parameters():
                p.requires_grad = True
    
    # Unfreeze the patch merger
    for p in vision.merger.parameters():
        p.requires_grad = True

elif part_to_train == "vision_only+lora":
    for p in model.parameters():
        p.requires_grad = False
        
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.3,
        target_modules=["q_proj", "v_proj", ]
    )
    model = get_peft_model(model, lora_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"After peft {trainable_params}/{total_params}")
    # Vision submodules
    vision = model.model.visual
    num_total_blocks = len(vision.blocks)
    num_unfreeze = 2
    
    # Unfreeze patch embedding
    for p in vision.patch_embed.parameters():
        p.requires_grad = True
    
    # Unfreeze the last `num_unfreeze` vision blocks
    for idx, block in enumerate(vision.blocks):
        if idx >= num_total_blocks - num_unfreeze:
            for p in block.parameters():
                p.requires_grad = True
    
    # Unfreeze the patch merger
    for p in vision.merger.parameters():
        p.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Adding vision {trainable_params}/{total_params}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"{trainable_params}/{total_params}")

# 4) DataLoaders
train_loader = DataLoader(
    RadioGalaxyDataset(train_subset, processor),
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=num_workers,
)

num_update_steps_per_epoch = len(train_loader)
max_train_steps = num_update_steps_per_epoch * (epochs + 1)
num_warmup_steps = int(0.1 * max_train_steps)

scheduler_type = lr_schedule
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
if scheduler_type == "linear":
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
elif scheduler_type == "cosine":
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
elif scheduler_type == "constant":
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps
    )
else:
    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

RUN_ID = uuid.uuid4().hex[:8]
CKPT_DIR = Path(f"./ckpt_{RUN_ID}")
MILESTONE_EVERY = 10
EPS = 1e-8

best_train_loss = float("inf")
best_epoch = None
best_test_metrics = None
improved_since_last_milestone = False

losses = []

try:
    global_step = 0
    for epoch in range(0, epochs + 1):
        model.train()
        total_train = 0.0

        for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            optimizer.zero_grad()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values'],
                image_grid_thw=batch['image_grid_thw'],
                labels=batch['labels']
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train += float(loss.item())
            global_step += 1

        train_average_loss = total_train / max(1, len(train_loader))
        if use_wandb:
            wandb.log({"epoch": epoch, "train_average_loss": train_average_loss})
        log.info(f"Epoch {epoch} Train Loss: {train_average_loss:.6f}")

        if train_average_loss + EPS < best_train_loss:
            best_train_loss = train_average_loss
            best_epoch = epoch
            improved_since_last_milestone = True

            if CKPT_DIR.exists():
                shutil.rmtree(CKPT_DIR, ignore_errors=True)
            CKPT_DIR.mkdir(parents=True, exist_ok=True)

            try:
                model.save_pretrained(str(CKPT_DIR))
                log.info(f"[epoch {epoch}] Saved new-best checkpoint -> {CKPT_DIR}")
            except Exception as e:
                log.warning(f"Could not save checkpoint: {e}")

        if (epoch + 1) % MILESTONE_EVERY == 0:
            if improved_since_last_milestone:
                model.eval()
                with torch.no_grad():
                    best_test_metrics = evaluate_score(
                        model, processor, test_dataset, system_message, query_text
                    )
                if use_wandb and isinstance(best_test_metrics, dict):
                    for k, v in best_test_metrics.items():
                        wandb.log({"epoch": epoch, f"test_{k}": v})
                log.info(f"[Test metrics @ epoch {epoch}] {best_test_metrics}")
                improved_since_last_milestone = False
            else:
                log.info(f"[epoch {epoch}] No new best since last milestone — skipping TEST eval.")

        f1_test = (best_test_metrics or {}).get("f1", float('nan'))
        losses.append([train_average_loss, float('nan'), float('nan'), f1_test])

except KeyboardInterrupt:
    losses = np.array(losses)
    log.info(losses)

log.info(f"Best train loss {best_train_loss:.6f} at epoch {best_epoch}")

if CKPT_DIR.exists():
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    log.info("Deleted checkpoint directory to keep the run clean.")

losses = np.array(losses)
log.info(losses)
if use_wandb:
    wandb.finish()