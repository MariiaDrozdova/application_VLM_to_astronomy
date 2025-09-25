import argparse
import logging
import sys

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import random_split

from src.utils import report_results
from src.data import load_mirabest_data
from src.all_predict import zero_shot_predict_func, few_shot_predict_func, theory_shot_predict_func

def parse_args():
    parser = argparse.ArgumentParser(description="VLM SFT training")

    # booleans
    parser.add_argument("--use_wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--image_first", dest="image_first", action="store_true", help="put image before text")
    parser.add_argument("--force_repeat", action="store_true",
                        help="Ignore previous completed runs (default: False)")

    # strings
    parser.add_argument("--device", type=str, default="cuda:0", help="torch device")
    parser.add_argument("--extra_line", type=str, default="tmp", help="suffix for save path")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--part_to_train", type=str, default="LoRA")

    # ints
    parser.add_argument("--val_size", type=int, default=70)
    parser.add_argument("--train_size", type=int, default=656)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sampling_regime", type=str, default="zero-shot")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--fixed_indexes",
        type=int,
        nargs="+",  # accepts one or more values
        default=[105, 102, 3, 100],
        help="List of fixed indexes"
    )
    parser.add_argument("--nearest_neighbors", type=int, default=5)

    return parser.parse_args()

def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_id):
    if "gemini" in model_id:
        from google.genai import Client
        from environment import GEMINI_FLASH_2_API_KEY

        client = Client(api_key=GEMINI_FLASH_2_API_KEY)

        model = client
        processor = model_id
    elif "gpt" in model_id:
        from openai import OpenAI
        from environment import GPT_API_KEY
        client = OpenAI(api_key=GPT_API_KEY)

        model = client
        processor = model_id
    elif "Qwen2-" in model_id:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        min_pixels = 224 * 224
        max_pixels = 2048 * 2048
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=None,
            dtype=dtype,
            # attn_implementation="flash_attention_2",
        ).to(device)

        processor = Qwen2VLProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    elif model_id.startswith("Qwen/Qwen2.5-VL-"):
        from transformers import AutoProcessor
        import torch
        processor = AutoProcessor.from_pretrained(model_id)

        try:
            from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, dtype=torch.bfloat16,
            ).to(device)
        except Exception:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)
        return model, processor
    elif model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        import torch

        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                  dtype=torch.float16, low_cpu_mem_usage=True)
        model.to(device)
    elif model_id == "HuggingFaceM4/idefics2-8b-chatty":
        from transformers import Idefics2ForConditionalGeneration, Idefics2Processor
        import torch
        processor = Idefics2Processor.from_pretrained(model_id)
        model = Idefics2ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        ).to(device)
        return model, processor

    elif model_id == "OpenGVLab/InternVL2-1B":
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,  # or float16 if you prefer
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        return model, processor

    elif model_id == "openbmb/MiniCPM-V-2_6":
        from transformers import AutoProcessor, AutoModel
        import torch
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        return model, processor

    elif model_id == "microsoft/Phi-3.5-vision-instruct":
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        return model, processor
    elif model_id == "llava-hf/llava-1.5-7b-hf":
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)

    elif model_id == "Qwen/Qwen3-8B":
        from transformers import AutoModelForCausalLM, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto"
        )
    elif model_id == "deepseek-ai/deepseek-vl2-tiny" or model_id == "deepseek-ai/deepseek-vl2-small":
        import os
        os.environ["XFORMERS_DISABLE_MEMORY_EFFICIENT_ATTENTION"] = "1"
        from transformers import AutoModelForCausalLM
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        import torch

        model_path = model_id
        model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )
    return model, processor


def evaluate_score(
        model, processor,
        train_dataset, test_dataset,
        system_message, query_text,
        examples_message, query_text_example, summary_after_examples_text
):
    test_indexes = range(len(test_dataset))
    y_true_majority = []
    y_pred_majority = []

    for i in tqdm(test_indexes, desc="Gathering preds"):
        if "few-shot" in sampling_regime:
            if "most-closest-neighbors" in sampling_regime:
                indexes = clip_retriever.retrieve_global(test_dataset[i]["image"], nearest_neighbors)
            elif "balanced-closest-neighbors" in sampling_regime:
                supports = clip_retriever.retrieve_per_class(test_dataset[i]["image"], nearest_neighbors // 2)
                indexes = [idx for indices in supports.values() for idx in indices]
            elif "fixed-neighbors" in sampling_regime:
                indexes = fixed_indexes
        else:
            indexes = None
        gt = test_dataset[i]["label"]  # e.g. "FR-I" or "FR-II"
        ans = shot_predict(
            model, processor, train_dataset, test_dataset,
            indexes,
            i,
            device=device,
            print_text=verbose,
            temperature=temperature,
            reps=reps,
            system_message=system_message,
            examples_message=examples_message,
            query_text_example=query_text_example,
            query_text=query_text,
            summary_after_examples_text=summary_after_examples_text,
            max_new_tokens=2048,
            image_first=image_first,
        )
        if i == 0:
            print(ans)
        pred = ans["final_answer"]
        if pred is None:
            pred = ""
        if pred.lower().startswith("answer"):
            pred = pred.split(":", 1)[1].strip()
        y_true_majority.append(gt)
        y_pred_majority.append(pred)

    score = report_results(y_true_majority, y_pred_majority, True)
    score["y_true_majority"] = y_true_majority
    score["y_pred_majority"] = y_pred_majority
    return score

def run_already_done(project, config):
    api = wandb.Api()
    runs = api.runs(f"tespoir/{project}")

    filtered_config = {k: v for k, v in config.items() if k != "device" and k!= "extra_line" and k!= "n_reps" and k!= "nearest_neighbors" and k!= "reps"}
    log.info(filtered_config)

    normalized_config = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in filtered_config.items()
    }
    log.info(normalized_config)
    log.info(runs)

    for r in runs:
        filtered_config = {k: v for k, v in r.config.items() if k != "device" and k!= "extra_line" and k!= "n_reps" and (k != "nearest_neighbors") and k!= "reps"}
        run_config = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in filtered_config.items()
        }
        if run_config == normalized_config and r.state == "finished":
            return True
    log.info("This config was not explored yet")
    return False

args = parse_args()

# Example usage:
device = args.device
use_wandb = args.use_wandb
extra_line = args.extra_line
model_id = args.model_id
val_size = args.val_size
train_size = args.train_size
n_reps = args.n_reps
index = args.index
image_first = args.image_first
sampling_regime=args.sampling_regime
temperature = args.temperature
reps=args.reps
fixed_indexes = args.fixed_indexes
nearest_neighbors=args.nearest_neighbors
verbose=args.verbose
seed_nb=args.seed

train_dataset, test_dataset = load_mirabest_data()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

messages_pot = []
#0
system_message = (
            "You are an expert radio galaxy classifier. "
            "FR-I: bright core, gradually fading jets. "
            "FR-II: faint core, jets end in hotspots. "
            "Consider core brightness and jet termination."
            )
query_text = (
            "Core: Bright or Faint? Jets: Fading or Hotspots? "
            "Classify based on core and jet properties: FR-I (bright core, fading jets) "
            "or FR-II (faint core, hotspots). Make a selection."
            "FR-I or FR-II?"
        )
query_text = "Respond only FR-I or FR-II."
examples_message, query_text_example, summary_after_examples_text = "", "", ""

messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])
#1
system_message='You are an expert at classifying radio galaxies as either Fanaroff-Riley Type I (FR-I) or Type II (FR-II). FR-I galaxies have central brightness and edge-darkened lobes. FR-II galaxies have edge-brightened lobes and hotspots at lobe ends. Jet characteristics can also aid in classification (common jets for FR-I, often one-sided jets for FR-II). Focus on lobe brightness distribution and hotspot presence.'
query_text='Describe the lobe brightness distribution (edge-brightened or edge-darkened) and the presence and location of hotspots in the radio galaxy image. Classify the galaxy as either FR-I or FR-II. Respond with: <answer>FR-I/FR-II</answer>.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])
#2
system_message='You are an expert at classifying radio galaxies as either Fanaroff-Riley Type I (FR-I) or Type II (FR-II). FR-I galaxies have central brightness and edge-darkened lobes. FR-II galaxies have edge-brightened lobes and hotspots at lobe ends. Jet characteristics can also aid in classification (common jets for FR-I, often one-sided jets for FR-II). Focus on lobe brightness distribution and hotspot presence.'
query_text='Respond only with: <answer>FR-I/FR-II</answer>.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#3
system_message='You are an expert at classifying radio galaxies as either Fanaroff-Riley Type I (FR-I) or Type II (FR-II). FR-I galaxies have central brightness and edge-darkened lobes. FR-II galaxies have edge-brightened lobes and hotspots at lobe ends. Jet characteristics can also aid in classification (common jets for FR-I, often one-sided jets for FR-II). Focus on lobe brightness distribution and hotspot presence.'
query_text='Describe the lobe brightness distribution (edge-brightened or edge-darkened) and the presence and location of hotspots in the radio galaxy image. Classify the galaxy as either FR-I or FR-II. Finish your response with: <answer>FR-I/FR-II</answer>.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#4
system_message='You are an expert at classifying radio galaxies as either Fanaroff-Riley Type I (FR-I) or Type II (FR-II). FR-I galaxies have central brightness and edge-darkened lobes. FR-II galaxies have edge-brightened lobes and hotspots at lobe ends. Jet characteristics can also aid in classification (common jets for FR-I, often one-sided jets for FR-II). Focus on lobe brightness distribution and hotspot presence.'
query_text='Describe the lobe brightness distribution (edge-brightened or edge-darkened) and the presence and location of hotspots in the radio galaxy image. Classify the galaxy as either FR-I or FR-II. Respond with: <answer>FR-I/FR-II</answer>.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#5
system_message='''
You are an astronomer tasked with classifying morphologies of radio galaxies.  
There are two classes:
• FR-I: bright toward the center, fainter at the lobes’ edges (edge-darkened), steep spectra, common jets, ram-pressure distortions in rich X-ray clusters.  
• FR-II: edge-brightened, more luminous, bright hotspots at lobe ends, one-sided jets.'''
query_text='Define a morphology class of this image. Respond only FR-I or FR-II.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#6
system_message='''
You are an astronomer tasked with classifying real images of radio galaxies.
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.'''
query_text='Define a morphology class of this image. Respond only FR-I or FR-II.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#7
system_message='''
You are an astronomer tasked with classifying morphologies of radio galaxies.  
There are two classes:
• FR-I: bright toward the center, fainter at the lobes’ edges (edge-darkened), common jets, ram-pressure distortions in rich X-ray clusters.  
• FR-II: edge-brightened, more luminous, bright hotspots at lobe ends, one-sided jets.'''
query_text='''Define a morphology class of this image. First analyze the features with respect to the described classes. Conclude if FR-I or FR-II.
Respond to the previous questions in the following format:
<think>...</think>
<answer>...</answer>
'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#8
system_message='''
You are an astronomer tasked with classifying morphologies of radio galaxies.  
There are two classes:
• FR-I: bright toward the center, fainter at the lobes’ edges (edge-darkened), common jets, ram-pressure distortions in rich X-ray clusters.  
• FR-II: edge-brightened, more luminous, bright hotspots at lobe ends, one-sided jets.'''
query_text='''Define a morphology class of this image. First analyze the features with respect to the described classes. Conclude if FR-I or FR-II.
Respond to the previous questions in the following format:
<think>...</think>
<answer>...</answer>
'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])
#9
system_message='''
You are an astronomer tasked with classifying real images of radio galaxies.
    
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.

    Now you will get the real image of a galaxy. Respond in the following format: <think>...</think> <answer>...</answer>
'''
query_text='''First, list the features you need to identify the class, then analyze them. Estimate the probability from 0 to 1 to be each class. Conclude if FR-I or FR-II.'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#10
system_message='''
You are an astronomer tasked with classifying morphologies of radio galaxies.  
There are two classes:
• FR-I: bright toward the center, fainter at the lobes’ edges (edge-darkened), common jets, ram-pressure distortions in rich X-ray clusters.  
• FR-II: edge-brightened, more luminous, bright hotspots at lobe ends, one-sided jets.
'''
query_text='''Define a morphology class of this image. First analyze the features with respect to the described classes. Conclude which of FR-I or FR-II is more probable (you must choose one class as the answer, you cannot ask for more information or say that you do not know).
Respond to the previous questions in the following format:
    <think>...</think>
    <answer>...</answer>'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#11
system_message=(
            "Core: Bright or Faint? Jets: Fading or Hotspots? "
            "<think>Classify based on core and jet properties: FR-I (bright core, fading jets) "
            "or FR-II (faint core, hotspots). Make a selection.</think> "
            "<answer>FR-I or FR-II?</answer>"
)
query_text=(
            "Respond only"
            "<answer>FR-I or FR-II?</answer>"
        )
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#12
system_message='You are an expert at classifying radio galaxies as either Fanaroff-Riley Type I (FR-I) or Type II (FR-II). FR-I galaxies have central brightness and edge-darkened lobes. FR-II galaxies have edge-brightened lobes and hotspots at lobe ends. Jet characteristics can also aid in classification (common jets for FR-I, often one-sided jets for FR-II). Focus on lobe brightness distribution and hotspot presence.'
query_text='Describe the lobe brightness distribution (edge-brightened or edge-darkened) and the presence and location of hotspots in the radio galaxy image. Classify the galaxy as either FR-I or FR-II. Finish your response with: <answer>FR-I/FR-II</answer>.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

cartoon_messages_pot = []

#0
system_message='''
You are an astronomer tasked with classifying real images of radio galaxies based on this diagram.
    The image above is a diagram compairing two radio galaxy morphologies. Ignore it for classification, remember just features.
    
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.'''
query_text='Define a morphology class of this image. Respond only FR-I or FR-II.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

system_message='''
You are an astronomer tasked with classifying real images of radio galaxies.
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.'''
query_text='Define a morphology class of this image. Respond only FR-I or FR-II.'
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#2
system_message='''You are an astronomer tasked with classifying real images of radio galaxies based on the diagram compairing two radio galaxy morphologies based on their features. Ignore it for classification, remember just features.
    
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.
    
    When studying the query image, analyze each listed feature separately.'''
query_text='''Define a morphology class of this image. 
    Analyze the features with respect to the described classes. 
    Conclude if FR-I or FR-II.
    Respond to the previous questions in the following format: <think>...</think> <answer>...</answer>'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#3
system_message='''You are an astronomer tasked with classifying real images of radio galaxies.
    This diagram roughly shows the features of two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.
    '''
query_text='''Define a morphology class of this image. First analyze the features with respect to the described classes. Conclude if FR-I or FR-II.
Respond to the previous questions in the following format:
<think>...</think>
<answer>...</answer>'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#4
system_message='''You are an astronomer tasked with classifying real images of radio galaxies.
    
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.

    Now you will get the real image of a galaxy. Respond in the following format: <think>...</think> <answer>...</answer>'''
query_text='''List the features you need to identify the class. Conclude if FR-I or FR-II.'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#5
system_message='''
You are an astronomer tasked with classifying real images of radio galaxies.
    
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.

    Now you will get the real image of a galaxy. Respond in the format: <think>...</think> <answer>...</answer>. '''
query_text='''First, list the features you need to identify the class, then analyze them. Estimate the probability from 0 to 1 to be each class. Conclude if FR-I or FR-II.'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#6
system_message='''
You are an astronomer tasked with classifying real images of radio galaxies based on the diagram compairing two radio galaxy morphologies based on their features. Ignore it for classification, remember just features.
    
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.
    
    When studying the query image, analyze each listed feature separately.'''
query_text='''First, list the features you need to identify the class, then analyze them. Carefully consider each feature.Keep the reasoning short. After always classify the last image into one of the two classes. Conclude your final answer as:
<answer>FR-I</answer>
or
<answer>FR-II</answer>'''
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

#7
system_message = """You are an astronomer tasked with classifying real images of radio galaxies based on this diagram.
    The image above is a diagram compairing two radio galaxy morphologies.
    
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.
    
    Respond with only FR-I or FR-II."""
query_text=''''''
examples_message=""
query_text_example=""
summary_after_examples_text=""
cartoon_messages_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

fewshot_pot = []

system_message='''
You are an astronomer tasked with classifying morphologies of radio galaxies.  
There are two classes:
• FR-I: bright toward the center, fainter at the lobes’ edges (edge-darkened), steep spectra, common jets, ram-pressure distortions in rich X-ray clusters.  
• FR-II: edge-brightened, more luminous, bright hotspots at lobe ends, one-sided jets.'''
query_text='Define a morphology class of this image. Respond only FR-I or FR-II.'
examples_message=""
query_text_example='Define a morphology class of this image. Respond only FR-I or FR-II.'
summary_after_examples_text=""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

system_message='''
You are an astronomer tasked with classifying real images of radio galaxies.
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.
   
'''
query_text='Define a morphology class of this image. Respond only FR-I or FR-II.'
examples_message="Here are examples:"
query_text_example='Define a morphology class of this image. Respond only FR-I or FR-II.'
summary_after_examples_text=""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

system_message='''

 You are an astronomer classifying the morthology of real galaxies. 
            
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.

    Look carefully through examples of radio galaxies:'''
query_text='''
    Explain the previous examples. Then classify the following galaxy image using the format <think>..</think><answer>..</answer>.  Explain your reasoning and motivation. '''
examples_message=""
query_text_example='Example: Classify this galaxy image.'
summary_after_examples_text=""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

system_message='''You are an astronomer classifying the morthology of real galaxies. 
            
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.

    Format your answer as <think></think><answer></answer>. 
 Classify the galaxy image. '''
query_text=''' '''
examples_message=""
query_text_example='Example: Classify this galaxy image.'
summary_after_examples_text=""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

system_message='''You are an astronomer classifying the morthology of real galaxies. 
            
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.

    Format your answer as <think></think><answer></answer>. 
 Classify the galaxy image. '''
query_text='''Explain how to correctly classify radio galaxies as FR-I or FR-II. Classify the galaxy image. '''
examples_message=""
query_text_example='Classify this galaxy image.'
summary_after_examples_text=""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

system_message='''You are an astronomer classifying the morthology of real galaxies. 
            
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.

    Format your answer as <think></think><answer></answer>. 
 Classify the galaxy image. '''
query_text='''Classify the last image. Use the format <think>..</think><answer>..</answer>.  In your think block explain thoroughly your class prediction.'''
examples_message=""
query_text_example='Classify this galaxy image.'
summary_after_examples_text=""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])

system_message = (
            "You are an expert radio galaxy classifier. "
            "FR-I: bright core, gradually fading jets. "
            "FR-II: faint core, jets end in hotspots. "
            "Consider core brightness and jet termination."
            )
query_text = (
            "Core: Bright or Faint? Jets: Fading or Hotspots? "
            "Classify based on core and jet properties: FR-I (bright core, fading jets) "
            "or FR-II (faint core, hotspots). Make a selection."
            "FR-I or FR-II?"
        )
query_text_example = "Respond only FR-I or FR-II."
examples_message,  summary_after_examples_text = "", ""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])


system_message='''You are an astronomer tasked with classifying real images of radio galaxies.
    There are two classes:
    - FR-I: bright toward the center, fainter at the lobes' edges, often shows jets, etc.
    - FR-II: edge-brightened, luminous lobes, bright hotspots at ends.
    
    '''
query_text='Define a morphology class of this image. Respond only FR-I or FR-II.'
examples_message=""
query_text_example=query_text
summary_after_examples_text=""
fewshot_pot.append([system_message, query_text, examples_message, query_text_example, summary_after_examples_text])



log.info(f"total stats:\n\tzero-shot : {len(messages_pot)}\n\ttheory-shot : {len(cartoon_messages_pot)}\n\tfew-shot : {len(fewshot_pot)}")
if sampling_regime == "zero-shot":
    assert 0 <= index < len(messages_pot), f"--index must be < {len(messages_pot)}"
    system_message, query_text, examples_message, query_text_example, summary_after_examples_text = messages_pot[index]
    shot_predict = zero_shot_predict_func(model_id)
elif sampling_regime == "theory-shot":
    assert 0 <= index < len(cartoon_messages_pot), f"--index must be < {len(cartoon_messages_pot)}"
    system_message, query_text, examples_message, query_text_example, summary_after_examples_text = cartoon_messages_pot[index]
    shot_predict = theory_shot_predict_func(model_id)
elif "few-shot" in sampling_regime:
    assert 0 <= index < len(fewshot_pot), f"--index must be < {len(fewshot_pot)}"
    system_message, query_text, examples_message, query_text_example, summary_after_examples_text = fewshot_pot[index]
    shot_predict = few_shot_predict_func(model_id)
else:
    log.info(f"Not defined option : {sampling_regime}")
    sys.exit(0)

config_dict = {
    "device": device,
    "extra_line": extra_line,
    "model_id": model_id,
    "val_size": val_size,
    "train_size": train_size,
    "system_message": system_message,
    "query_text": query_text,
    "examples_message": examples_message,
    "query_text_example": query_text_example,
    "summary_after_examples_text": summary_after_examples_text,
    "n_reps": n_reps,
    "index": index,
    "image_first": image_first,
    "sampling_regime": sampling_regime,
    "temperature": temperature,
    "reps": reps,
    "fixed_indexes": fixed_indexes,
    "nearest_neighbors": nearest_neighbors,
}

if use_wandb:
    if run_already_done("mira-best-test", config_dict) and not args.force_repeat:
        log.info("Run already completed — skipping.")
        exit(0)

    wandb.init(
        project="mira-best-test",
        config=config_dict
    )

model, processor = load_model(model_id)

n_total = len(train_dataset)
rest = n_total - val_size - train_size
generator = torch.Generator().manual_seed(42)
if rest <= 0:
    train_subset, _ = random_split(
        train_dataset,
        [n_total - val_size, val_size],
        generator=generator
    )
    train_subset = train_dataset
else:
    train_subset, _, _ = random_split(
        train_dataset,
        [train_size, val_size, rest],
        generator=generator
    )

if "most-closest-neighbors" in sampling_regime or "balanced-closest-neighbors" in sampling_regime:
    from src.retriever import CLIPRetriever
    clip_retriever = CLIPRetriever(train_dataset, device=device)

set_seed(seed_nb)

with torch.no_grad():
    for i in range(n_reps):
        res_val = evaluate_score(
            model, processor,
            train_dataset, test_dataset,
            system_message, query_text,
            examples_message, query_text_example, summary_after_examples_text)
        if use_wandb:
            for key in res_val:
                wandb.log({f"test{i}_"+key: res_val[key]})



