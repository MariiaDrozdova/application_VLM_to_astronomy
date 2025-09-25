import os
import PIL
import json
import numpy as np
import torch

from src.datasets import MiraBest_full
from datasets import Dataset, Features, Value, Image
from functools import partial

#from galaxy_mnist import GalaxyMNIST
from torchvision.transforms.functional import to_pil_image

fr1_list = [0, 1, 2]
fr2_list = [5, 6]

def gen(ds):
    for img, orig_lbl in ds:
        if orig_lbl in fr1_list:
            lbl = "FR-I"
        elif orig_lbl in fr2_list:
            lbl = "FR-II"
        else:
            continue
        yield {
            "image":            img,    
            "query":            "",    
            "label":            lbl,   
            "human_or_machine": "" 
        }

def load_mirabest_data(root = "./data/mirabest"):
    train_ds = MiraBest_full(
        root=root,
        train=True,
        transform=None,          
        target_transform=None,
        download=True
    )
    test_ds = MiraBest_full(
        root=root,
        train=False,
        transform=None,
        target_transform=None,
        download=True
    )
    features = Features({
        "image":            Image(decode=True),
        "query":            Value("string"),
        "label":            Value("string"),
        "human_or_machine": Value("string"),
    })
    
    train_dataset = Dataset.from_generator(partial(gen, train_ds), features=features)
    test_dataset = Dataset.from_generator(partial(gen, test_ds), features=features)
    return train_dataset, test_dataset

def _to_pil(img):
    if isinstance(img, torch.Tensor):
        t = img
    else:
        arr = np.asarray(img)
        t = torch.from_numpy(arr)

    if t.ndim == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)  # HWC -> CHW

    t = t.contiguous()
    return to_pil_image(t)

def _gen_gmnist(ds):
    class_names = list(ds.classes)  # e.g. ["smooth-round", "smooth-cigar", "edge-on-disk", "unbarred-spiral"]

    images, targets = ds.data, ds.targets

    for img, y in zip(images, targets):
        if isinstance(y, torch.Tensor):
            y_idx = int(y.item())
        else:
            y_idx = int(y)

        lbl = class_names[y_idx]
        yield {
            "image":            _to_pil(img),
            "query":            "",
            "label":            lbl,   # string class, like the MiraBest "FR-I"/"FR-II"
            "human_or_machine": ""
        }

def load_gmnist_data(root):
    train_raw = GalaxyMNIST(
        root=root,
        download=False,
        train=True
    )
    test_raw = GalaxyMNIST(
        root=root,
        download=False,
        train=False
    )

    features = Features({
        "image":            Image(decode=True),
        "query":            Value("string"),
        "label":            Value("string"),
        "human_or_machine": Value("string"),
    })

    train_dataset = Dataset.from_generator(partial(_gen_gmnist, train_raw), features=features)
    test_dataset  = Dataset.from_generator(partial(_gen_gmnist, test_raw),  features=features)

    return train_dataset, test_dataset

def make_radio_galaxy_generator(root_dir, split):
    ann_path = os.path.join(root_dir, "annotations", f"{split}.json")
    image_dir = os.path.join(root_dir, split)

    with open(ann_path, "r") as f:
        coco = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    image_id_to_category_id = {}
    for ann in coco["annotations"]:
        image_id_to_category_id[ann["image_id"]] = ann["category_id"]

    category_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    for image_id, file_name in image_id_to_filename.items():
        if image_id in image_id_to_category_id:
            label_name = category_id_to_name[image_id_to_category_id[image_id]]
            yield {
                "image_path": os.path.join(image_dir, file_name),
                "label": label_name
            }
            
def load_radio_galaxy_dataset(root_dir, split):
    gen = lambda: make_radio_galaxy_generator(root_dir, split)
    return Dataset.from_generator(gen)

def add_image_column(example):
    example["image"] = PIL.Image.open(example["image_path"]).convert("RGB")
    return example

def load_radiogalaxynet_dataset(path="/home/drozdova/projects/fewshot_rag_mirabest/data/RadioGalaxyNET/"):
    train_dataset = load_radio_galaxy_dataset(path, "train")
    train_dataset = train_dataset.map(add_image_column)
    val_dataset = load_radio_galaxy_dataset(path, "val")
    val_dataset = val_dataset.map(add_image_column)
    test_dataset = load_radio_galaxy_dataset(path, "test")
    test_dataset = test_dataset.map(add_image_column)
    return train_dataset, val_dataset, test_dataset

