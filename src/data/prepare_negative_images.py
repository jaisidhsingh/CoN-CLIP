import time
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dsets
from PIL import Image
import os
import clip
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


# prepare I' from COCO Dataset
class FineTuningDatasetHelper(Dataset):
    def __init__(self, args, transform=None):
        self.args = args
        data = torch.load("/workspace/datasets/cc3m/preprocessed_data_final.pt")
		# print(data.keys())
        self.transform = transform

        # self.image_paths = data["test"]["image_paths"]
        self.annotations = data["test"]["labels"]

    def text_transform(self, text_list):
        tokens = clip.tokenize(text_list)
        return tokens

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annos = self.annotations[idx]
        # caption = annos["json"]["caption"]
        # negative_prompt = annos["sop_data"]["negative-prompt"]
        concept_present = annos["sop_data"]["exclusion-prompt"]
        concept_absent = annos["sop_data"]["sop_chosen_object"]

        texts_tokenized = self.text_transform([
            concept_present, concept_absent
        ])

        return texts_tokenized.unsqueeze(0)


def setup_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--negative-images-dataset", type=str, default="coco")
	parser.add_argument("--batch-size", type=int, default=2048)
	parser.add_argument("--clip-model-name", type=str, default="ViT-B/16")
	parser.add_argument("--coco-split", type=str, default="train")
	parser.add_argument("--topk", type=int, default=3)

	args = parser.parse_args()
	return args

def custom_collate(batch):
	images = [item[0] for item in batch]
	images = torch.stack(images, dim=0)
	return images

def get_data(args, transform=None):
	loader, dataset = None, None

	if args.negative_images_dataset == "coco":
		dataset = dsets.CocoCaptions(
			root="/workspace/datasets/coco/coco_train2017/train2017",
			annFile="/workspace/datasets/coco/coco_ann2017/annotations/captions_train2017.json",
			transform=transform
		)

		loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=custom_collate)
	
	return loader, dataset

@torch.no_grad()
def encode_coco_images(args):
	model, preprocess = clip.load(args.clip_model_name, device=args.device)

	all_features = []
	loader, dataset = get_data(args, preprocess)
	bar = tqdm(total=len(loader))
	num_images_done = 0

	for images in loader:
		batch_size = images.shape[0]
		num_images_done += batch_size

		images = images.to(args.device)

		image_features = model.encode_image(images)
		image_features = image_features / image_features.norm(dim=-1, keepdim=True)

		all_features.append(image_features.cpu())
		
		bar.update(1)
		bar.set_postfix({"images_encodes": num_images_done})

	all_features = torch.cat(all_features, dim=0)
	save_path = f"/workspace/datasets/coco/coco_{args.coco_split}2017/coco_{args.coco_split}2017_image_features.pt"
	torch.save(all_features, save_path)
	print("Done!")


@torch.no_grad()
def encode_texts(args):
	model, preprocess = clip.load(args.clip_model_name, device=args.device)

	dataset = FineTuningDatasetHelper(args)
	loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

	all_features = []
	num_done = 0
	bar = tqdm(total=len(loader))

	for tokens in loader:
		batch_size = tokens.shape[0]
		dims = tokens.shape[-1]

		tokens = tokens.to(args.device) # shape [B, 2, 77]
		tokens = tokens.view(batch_size*2, dims)

		s_op_features = model.encode_text(tokens)
		new_dims = s_op_features.shape[-1]

		s_op_features = s_op_features / s_op_features.norm(dim=-1, keepdim=True)
		s_op_features = s_op_features.view(batch_size, 2, new_dims)

		all_features.append(s_op_features.cpu())
		num_done += batch_size
		bar.update(1)
		bar.set_postfix({"num_done": num_done})

	all_features = torch.cat(all_features, dim=0)
	save_path = f"/workspace/datasets/cc3m/final_conclip/s_op_cc3m_text_features.pt"
	torch.save(all_features, save_path)
	print("Done!")

def get_scores(args):
	image_features_path = f"/workspace/datasets/coco/coco_{args.coco_split}2017/coco_{args.coco_split}2017_image_features.pt"
	s_op_features_path = f"/workspace/datasets/cc3m/final_conclip/s_op_cc3m_text_features.pt"

	image_features = torch.load(image_features_path)
	s_op_features = torch.load(s_op_features_path)
	s_features = s_op_features[:, 0, :].float().to(args.device)
	op_features = s_op_features[:, 1, :].float().to(args.device)

	N = image_features.shape[0]
	step_size = 5000
	i = 0
	c = 1

	for j in tqdm(range(step_size, N, step_size)):
		img_features = image_features[i:j, :].float().to(args.device)
		scores = (img_features @ s_features.T) - (img_features @ op_features.T)
		values, indices = scores.topk(k=1, dim=0)
		results = {"chunk": c, "values": values.cpu(), "indices": indices.cpu()}
		torch.save(results, f"/workspace/datasets/cc3m/final_conclip/negative_images_scores_chunked/chunk_{c}.pt")

		# print(indices.shape)

		i += step_size
		c += 1

	print("loop done")

	img_features = image_features[i:N, :].float().to(args.device)
	scores = (img_features @ s_features.T) - (img_features @ op_features.T)
	values, indices = scores.topk(k=1, dim=0)
	results = {"chunk": c, "values": values.cpu(), "indices": indices.cpu()}
	torch.save(results, f"/workspace/datasets/cc3m/final_conclip/negative_images_scores_chunked/chunk_{c}.pt")
	print("done!")
	# print(indices.shape)

def merge_topk(args):
	num_chunks = 24
	step_size = 5000

	chunk_folder = "/workspace/datasets/cc3m/final_conclip/negative_images_scores_chunked"
	chunks = [torch.load(os.path.join(chunk_folder, f"chunk_{i}.pt")) for i in range(1, num_chunks+1)]

	results = []
	N = chunks[0]["indices"].T.shape[0]

	for i in tqdm(range(N)):
		scores = [{"val": chunks[j]["values"].T[i], "index": chunks[j]["indices"].T[i].item() + j*step_size} for j in range(num_chunks)]
		scores.sort(key=lambda x: x["val"])
		scores = scores[:args.topk]
		results.append([item["index"] for item in scores])
	
	torch.save(results, "/workspace/datasets/cc3m/final_conclip/negative_image_mapping.pt")

def review(args):
	dataset = FineTuningDatasetHelper(args)
	dataset2 = dsets.CocoCaptions(
		root="/workspace/datasets/coco/coco_train2017/train2017",
		annFile="/workspace/datasets/coco/coco_ann2017/annotations/captions_train2017.json")

	for j in range(5, 8):
		concept_present = dataset.annotations[j]["sop_data"]["exclusion-prompt"]
		concept_absent = dataset.annotations[j]["sop_data"]["sop_chosen_object"]


		mapping = torch.load("/workspace/datasets/cc3m/final_conclip/negative_image_mapping.pt")
		negative_image_indices = mapping[j]

		negative_images = [dataset2[i][0] for i in negative_image_indices]

		for k in range(3):	
			chosen = negative_images[k]
			title = f"Present: {concept_present} \n Absent: {concept_absent}"

			plt.imshow(chosen)
			plt.title(title)
			plt.axis("off")
			plt.savefig(f"negative_image_reviews/neg_review_{j}_{k}.png")

def main(args):
	# encode_coco_images(args)
	# encode_texts(args)
	# get_scores(args)
	# merge_topk(args)
	review(args)


if __name__ == "__main__":
	args = setup_args()
	main(args)
