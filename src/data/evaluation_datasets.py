from PIL import Image
import torch
import clip
from torch.utils.data import Dataset, DataLoader

from configs.data_configs import configs


class ZeroShotClassificationDataset(Dataset):
	def __init__(self, args, transform=None):
		self.args = args
		self.transform = transform

		# self.data = torch.load(f"/workspace/datasets/{args.eval_dataset}/preprocessed_data.pt")
		self.data = torch.load(configs.zeroshot_data_loader(args.eval_dataset))
		self.class_names = [x.lower().replace("_", " ") for x in self.data["class_list"]]
		self.image_paths = self.data["test"]["image_paths"]
		self.labels = self.data["test"]["labels"] 
	
	def __len__(self):
		return len(self.labels)
	
	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image = Image.open(image_path)

		if self.transform is not None:
			image = self.transform(image)
		
		label = self.labels[idx]

		return image, label


class CCNegEvalDataset(Dataset):
	def __init__(self, args, transform=None):
		self.args = args
		data = torch.load("/workspace/datasets/ccneg/preprocessed_data.pt")
		self.transform = transform
		self.split_size = configs.num_ccneg_eval_samples

		N = len(data["image_paths"])

		self.image_paths, self.annotations = None, None

		if self.args.num_ops == -1: # take all prompts 
			self.image_paths = data["image_paths"]
			self.annotations = data["annotations"]
			
			# keep only the evaluation set (last ``self.split_size`` samples)
			self.image_paths = self.image_paths[-self.split_size:]
			self.annotations = self.annotations[-self.split_size:]

		else: # take the prompts which have ``args.num_ops`` object-predicate pairs and make sure they are in the eval split
			self.image_paths = data["image_paths"]
			self.annotations = data["annotations"]

			indices = data["num_ops"][self.args.num_ops]

			M = self.split_size if len(indices) >= self.split_size else len(indices)

			self.image_paths = [self.image_paths[i] for i in indices[-M:] if N - configs.num_ccneg_eval_samples <= i < N]
			self.annotations = [self.annotations[i] for i in indices[-M:] if N - configs.num_ccneg_eval_samples <= i < N]

		
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
		negative_prompt = negative_prompt.replace(",", "")

		return image, [caption], [negative_prompt]


class CCNeg2EvalDataset(Dataset):
	def __init__(self, args, transform=None):
		self.args = args
		# data = torch.load("/workspace/datasets/ccneg_eval/generated_datasets/annotations_ccnegeval500/preprocessed_data_final_modded_2.pt")
		# data = torch.load("/workspace/datasets/ccneg2/generated_datasets/annotations_ccnegeval500/preprocessed_data_final_modded_2.pt")
		data = torch.load("/workspace/datasets/ccneg2/final/preprocessed_data.pt")
		self.transform = transform

		# N = len(data["labels"])

		self.image_paths, self.annotations = None, None

		self.image_paths_positive = data["image_paths_positive"]
		self.image_paths_negative = data["image_paths_negative"]
		self.image_paths_negated  = data["image_paths_negated"]
		self.annotations = data["labels"]
			
		
	def text_transform(self, text_list):
		tokens = clip.tokenize(text_list)
		return tokens

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		image_pathlist_positive = self.image_paths_positive[idx]
		image_pathlist_negative = self.image_paths_negative[idx]
		image_pathlist_negated  = self.image_paths_negated [idx]
		image_path_positive = image_pathlist_positive[0]
		image_path_negative = image_pathlist_negative[0]
		image_path_negated  = image_pathlist_negated [0]
		
		image_positive = Image.open(image_path_positive)
		image_negative = Image.open(image_path_negative)
		image_negated  = None
		# image_negated  = Image.open(image_path_negated) # not needed for now!
		
		if self.transform is not None:
			if image_positive is not None:
				image_positive = self.transform(image_positive)
			if image_negative is not None:
				image_negative = self.transform(image_negative)
			if image_negated  is not None:
				image_negated = self.transform(image_negated)

		annos = self.annotations[idx]
		positive_caption = annos["positive_prompt"]
		negative_caption = annos["negative_prompt"]
		negated_caption  = annos["negated_prompt"]

		positive_caption, negative_caption, negated_caption = [
			p.replace(',', '') 
			for p in [
				positive_caption, 
				negative_caption, 
				negated_caption
			]
		]

		return image_positive, image_negative, positive_caption, negated_caption #negative_caption

	def collate(self, batch):
		images = [item[0] for item in batch]
		negative_images = [item[1] for item in batch]
		captions = [item[2] for item in batch]
		negative_captions = [item[3] for item in batch]
		return (images, negative_images, captions, negative_captions)


def test():
	def collate(batch):
		images = [item[0] for item in batch]
		captions = [item[1][0] for item in batch]
		negative_captions = [item[2][0] for item in batch]
		return (images, captions, negative_captions)

	args = None
	dataset = CCNegEvalDataset(args)
	print(len(dataset))
	loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
	(images, captions, negative_captions) = next(iter(loader))

	print(images)
	print(captions)
	print(negative_captions)


if __name__ == "__main__":
	test()