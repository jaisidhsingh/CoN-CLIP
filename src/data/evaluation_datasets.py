from PIL import Image
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from configs import configs


class CCNegEvalDataset(Dataset):
	def __init__(self, transform=None):
		data = torch.load(configs.finetuning_dataset_path)
		self.transform = transform
		self.split_size = configs.num_ccneg_eval_samples

		N = len(data["image_paths"])

		self.image_paths, self.annotations = None, None

		self.image_paths = [path.replace("/workspace/datasets/cc3m/", configs.ccneg_root_folder+"/") for path in data["image_paths"]]
		self.annotations = data["annotations"]
		
		self.image_paths = self.image_paths[-self.split_size:]
		self.annotations = self.annotations[-self.split_size:]

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
