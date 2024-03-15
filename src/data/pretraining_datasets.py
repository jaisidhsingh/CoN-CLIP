import torch
from torch.utils.data import Dataset
from PIL import Image
import clip


class PretrainingDataset(Dataset):
    def __init__(self, args, transform):
        self.args = args
        data = torch.load("/workspace/datasets/cc3m/preprocessed_data_final.pt")
        self.transform = transform
        
        self.image_paths = data["test"]["image_paths"]
        self.annotations = data["labels"]
    
        # storing important things here
        self.not_index = 783
    
    def find_not_index(self, text):
        tokens = clip.tokenize([text])
        tokens = tokens.tolist()[0]
        index = tokens.index(self.not_index)
        return index
    
    def text_transform(self, text_list):
        tokens = clip.tokenize(text_list)
        return tokens
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        annos = self.annotations[idx]
        caption = annos["json"]["caption"]
        negative_prompt = annos["sop_data"]["negative-prompt"]
        concept_present = annos["sop_data"]["exclusion-prompt"]
        concept_absent = annos["sop_data"]["sop_chosen_object"]
        
        idx = self.find_not_index(negative_prompt)
        texts_tokenized = self.text_transform([
            caption, negative_prompt, concept_present, concept_absent
        ])
        
        return image, texts_tokenized, idx
        
        
        
        