import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets as dsets
import clip
import os
import sys
from tqdm import tqdm
from training_utils.loss_functions import *
from training_utils.schedulers import *
from data import *
from training_utils.helpers import *


def zero_shot_eval(args, model, epoch, preprocess):
	model.eval()

	if "cifar10" not in args.eval_dataset and args.eval_dataset != "imagenet":
		dataset, loader = setup_zero_shot(args, preprocess)
	elif args.eval_dataset == "cifar10":
		dataset = dsets.CIFAR10(root="/workspace/datasets/cifar10", download=True, train=False, transform=preprocess)
		loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
	
	elif args.eval_dataset == "cifar100":
		dataset = dsets.CIFAR100(root="/workspace/datasets/cifar100", download=True, train=False, transform=preprocess)
		loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

	elif args.eval_dataset == "imagenet":
		dataset = dsets.ImageFolder(root="/workspace/datasets/imagenet/images/val", transform=preprocess)
		loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
	
	if args.eval_dataset == "cifar10" or args.eval_dataset == "cifar100":
		categories_tokenized = clip.tokenize([f"{c}" for c in dataset.classes]).to(args.device)
		neg_categories_tokenized = clip.tokenize([f"this is not a photo of a {c}" for c in dataset.classes]).to(args.device)
	
	if args.eval_dataset != "imagenet" and args.eval_dataset not in ["cifar10", "cifar100"]:
		categories_tokenized = clip.tokenize([f"{c}" for c in dataset.class_names]).to(args.device)
		neg_categories_tokenized = clip.tokenize([f"this is not a photo of a {c}" for c in dataset.class_names]).to(args.device)
	
	autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
	bar = tqdm(total=len(loader))


	# imagenet_save_name = args.clip_model_name.lower().replace("-", "_").replace("/", "") + "_conclip_nci_plus_b16.pt"
	imagenet_save_name = args.clip_model_name.lower().replace("-", "_").replace("/", "") + "laclip_b32.pt"
	imagenet_features_path = os.path.join("../checkpoints", "imagenet_zeroshot_weights", imagenet_save_name)
	with torch.no_grad():
		if args.eval_dataset == "imagenet" and not os.path.exists(imagenet_features_path):
			categories_tokenized = clip.tokenize([f"{c}" for c in IMAGENET_MAPPING]).to(args.device)
			neg_categories_tokenized = clip.tokenize([f"this is not a photo of a {c}" for c in IMAGENET_MAPPING]).to(args.device)


			category_features = model.encode_text(categories_tokenized)
			category_features = zeroshot_classifier(clip.tokenize, model, IMAGENET_MAPPING, TEMPLATES)
			category_features /= category_features.norm(dim=-1, keepdim=True)
			neg_category_features = model.encode_text(neg_categories_tokenized)
			neg_category_features /= neg_category_features.norm(dim=-1, keepdim=True)
			# category_features = category_features.view(1000, 768)
			# print("Imagenet zeroshot weights made!")
			torch.save(category_features, imagenet_features_path)

		# elif args.eval_dataset == "imagenet" and os.path.exists(imagenet_features_path):
		# 	# category_features = torch.load(imagenet_features_path)
		# 	categories_tokenized = clip.tokenize([f"this is a photo of a {c}" for c in IMAGENET_MAPPING]).to(args.device)
		# 	category_features = model.encode_text(categories_tokenized)
		# 	category_features /= category_features.norm(dim=-1, keepdim=True)
		else:
			# category_features = torch.load(imagenet_features_path)
			# neg_categories_tokenized = clip.tokenize([f"this is not a photo of a {c}" for c in IMAGENET_MAPPING]).to(args.device)
			category_features = model.encode_text(categories_tokenized)
			category_features /= category_features.norm(dim=-1, keepdim=True)

			neg_category_features = model.encode_text(neg_categories_tokenized)
			neg_category_features /= neg_category_features.norm(dim=-1, keepdim=True)
		
		correct, total = 0, 0
		negcorrect = 0        
		for (images, labels) in loader:
			images = images.to(args.device)
			labels = labels.long().to(args.device)

			image_features = model.encode_image(images)
			image_features /= image_features.norm(dim=-1, keepdim=True)

			logit_scale = model.logit_scale.item()

			logits = (logit_scale * image_features @ category_features.T).softmax(dim=-1)
			preds = logits.argmax(dim=-1)

			logits2 = (logit_scale * image_features @ neg_category_features.T).softmax(dim=-1)
			preds2 = logits2.argmax(dim=-1)

			correct += (preds == labels).sum().item()
			negcorrect += (preds2 == labels).sum().item()
			total += labels.shape[0]
			accuracy = round(correct/total * 100, 2)
			negaccuracy = round(negcorrect/total * 100, 2)

			bar.update(1)
			bar.set_postfix({"eval_at_epoch": epoch+1, "accuracy": accuracy, "delta": round(accuracy-negaccuracy, 2)})
		
		tqdm.write(f"Zero-shot image classification")
		tqdm.write(f"{args.eval_dataset.upper()} Accuracy = {accuracy} --- Epoch = {epoch+1}")
		tqdm.write("  ")

	return {"epoch": epoch+1, "zero_shot_accuracy": accuracy}

def main(args):
	model, preprocess = clip.load(args.clip_model_name, device=args.device)
	# ckpt = torch.load("../checkpoints/conclip_nci_plus3/ckpt_5_conclip_nci_plus3.pt")["model"]
	# ckpt = torch.load("../checkpoints/crepe_clip_b32/ckpt_13_crepe_clip_b32.pt")["model"]
	# ckpt = torch.load("../checkpoints/conclip_loss2_b32/ckpt_5_conclip_loss2_b32.pt")["model"]
	# ckpt = torch.load("../checkpoints/conclip_nci_plus_l14/ckpt_5_conclip_nci_plus_l14.pt")["model"]
	# ckpt = torch.load("../checkpoints/pretrained_vlms/negclip_b32.pth")["state_dict"]

	if args.use_laclip == 'true':
		laclip_mode_mapping = {
					'ViT-B/32': 'laion400m_b32',
					'ViT-B/16': 'laion400m_b16',
					'ViT-L/14': 'laion400m_l14'}
		assert args.clip_model_name in laclip_mode_mapping, 'this laclip checkpoint has not been downloaded.'
		laclip_mode = laclip_mode_mapping[args.clip_model_name]
		checkpoint_path = f"/workspace/jaisidh/btp/checkpoints/laclip_{laclip_mode}/laclip_{laclip_mode}.pt"
		checkpoint = torch.load(checkpoint_path)["state_dict"]
		model.load_state_dict(checkpoint)

	print(args.use_laclip, 'bruhhhh', args.clip_model_name)
	print('-'*25)
	model = model.float()
	model.load_state_dict(checkpoint)
	model = model.float().to(args.device)
	print("loaded")
	logs = zero_shot_eval(args, model, -1, preprocess)
	print(" ")

def get_args():
	parser = argparse.ArgumentParser()

	# model args
	parser.add_argument("--clip-model-name", type=str, default="ViT-B/16")
	parser.add_argument("--use-laclip", type=str, default="true")
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--lock-image-encoder", type=str, default="off")

	# training args
	parser.add_argument("--learning-rate", type=float, default=1e-6) 
	parser.add_argument("--precision", type=str, default="amp") 
	parser.add_argument("--norm-gradient-clip", type=float, default=None)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--warmup-steps", type=int, default=50)
	parser.add_argument("--weight-decay", type=float, default=0.2)

	# data args
	parser.add_argument("--eval-dataset", type=str, default="caltech_101")
	parser.add_argument("--num-workers", type=int, default=1)
	parser.add_argument("--batch-size", type=int, default=200) 
	parser.add_argument("--early-exit", type=int, default=-1)
	parser.add_argument("--val-point", type=int, default=150)
	parser.add_argument("--negative-images", type=str, default="off")

	# save args
	parser.add_argument("--save-every", type=int, default=1)
	parser.add_argument("--evaluate-every", type=int, default=1)
	parser.add_argument("--ckpt-save-folder", type=str, default="../checkpoints")
	parser.add_argument("--logs-save-folder", type=str, default="../logs")
	parser.add_argument("--experiment-name", type=str, default="conclip_nci_plus")

	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = get_args()
	# print(f"CoN-CLIP nci+3 {args.clip_model_name}")
	print(f"LaCLIP {args.clip_model_name}")
	# print(f"Vanilla CLIP {args.clip_model_name}")
	# print(f"Neg-CLIP {args.clip_model_name}")
	# print(f"CREPE-CLIP {args.clip_model_name}")
	main(args)
