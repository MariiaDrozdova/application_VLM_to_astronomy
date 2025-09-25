import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

class CLIPRetriever:
    def __init__(self, train_dataset, model_name="openai/clip-vit-base-patch32", device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model = CLIPModel.from_pretrained(model_name).eval().to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        all_embs = []
        self.labels = []
        self.class_embs = {}
        self.class_idxs = {}

        for idx, example in enumerate(train_dataset):
            label = example["label"]
            img = example["image"]
            inputs = self.clip_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feat = self.clip_model.get_image_features(**inputs)
                feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            emb = feat.cpu().numpy()[0]

            all_embs.append(emb)
            self.labels.append(label)

            self.class_embs.setdefault(label, []).append(emb)
            self.class_idxs.setdefault(label, []).append(idx)

        self.global_embs = np.stack(all_embs).astype("float32")

        for label, emb_list in self.class_embs.items():
            self.class_embs[label] = np.stack(emb_list).astype("float32")

    def return_text_embedding(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        return features.squeeze(0).cpu().numpy()

    def retrieve_global(self, query_img, k=5):
        # Retrieves the top-k nearest neighbors from the entire dataset.
        inputs = self.clip_processor(images=query_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_feat = self.clip_model.get_image_features(**inputs)
            q_feat = (q_feat / q_feat.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()[0]
        sims = self.global_embs @ q_feat
        topk = sims.argsort()[-k:][::-1]
        return topk.tolist()

    def retrieve_per_class(self, query_img, k_per_class=1):
        # Retrieve the top-k nearest neighbors within each class.
        inputs = self.clip_processor(images=query_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_feat = self.clip_model.get_image_features(**inputs)
            q_feat = (q_feat / q_feat.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()[0]
        
        result = {}
        for label, emb_array in self.class_embs.items():
            sims = emb_array @ q_feat
            topk = sims.argsort()[-k_per_class:][::-1]
            original_idxs = [self.class_idxs[label][i] for i in topk]
            result[label] = original_idxs
        return result

# Usage example:
# retriever = CLIPRetriever(train_dataset)
# global_neighbors = retriever.retrieve_global(query_image, k=5)
# per_class_neighbors = retriever.retrieve_per_class(query_image, k_per_class=1)
