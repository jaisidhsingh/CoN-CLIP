import torch
from torch.utils.data import Dataset
import torchvision.datasets as dsets
from PIL import Image
import clip
from configs import configs


class FineTuningDataset(Dataset):
    def __init__(self, transform=None):
        data = torch.load(configs.finetuning_dataset_path)
        self.transform = transform
        self.split_size = len(data["image_paths"]) - configs.num_ccneg_eval_samples

        self.image_paths = [path.replace("/workspace/datasets/cc3m/", configs.ccneg_root_folder+"/") for path in data["image_paths"]]
        self.annotations = data["annotations"]

        self.image_paths = self.image_paths[:self.split_size]
        self.annotations = self.annotations[:self.split_size]

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
        caption = self.text_transform([caption])

        negative_prompt = annos["sop_data"]["negative-prompt"]
        negative_prompt = negative_prompt.replace(",", "")
        negative_prompt = self.text_transform([negative_prompt])

        return image, caption, negative_prompt


class FineTuningDatasetWithNegatives(Dataset):
    def __init__(self, transform=None):
        data = torch.load(configs.finetuning_dataset_path)
        self.transform = transform
        self.split_size = len(data["image_paths"]) - configs.num_ccneg_eval_samples

        self.image_paths = [path.replace("/workspace/datasets/cc3m/", configs.ccneg_root_folder+"/") for path in data["image_paths"]]
        self.annotations = data["annotations"]

        self.image_paths = self.image_paths[:self.split_size]
        self.annotations = self.annotations[:self.split_size]

        self.negatives_dataset = dsets.CocoCaptions(
        root=configs.negative_image_dataset_root,
        annFile=configs.negative_image_dataset_annotations_path,
        transform=self.transform)
        self.negatives_mapping_cc3m_to_coco = torch.load(configs.negative_image_ft_mapping_path)

    def text_transform(self, text_list):
        tokens = clip.tokenize(text_list)
        return tokens

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        
        topk_indices = self.negatives_mapping_cc3m_to_coco[idx]
        top1_index = topk_indices[0]
        negative_image = self.negatives_dataset[top1_index][0]

        if self.transform is not None:
            image = self.transform(image)

        annos = self.annotations[idx]
        caption = annos["json"]["caption"]
        caption = self.text_transform([caption])

        negative_prompt = annos["sop_data"]["negative-prompt"]
        negative_prompt = negative_prompt.replace(",", "")
        negative_prompt = self.text_transform([negative_prompt])

        return image, negative_image, caption, negative_prompt
