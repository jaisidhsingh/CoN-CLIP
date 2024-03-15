import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import os
import argparse
import sys
from tqdm import tqdm

src_dir = os.path.abspath("../")
sys.path.append(src_dir)

from data.evaluation_datasets import CCNegEvalDataset, CCNeg2EvalDataset
from models import get_vlm, get_vlm_name, collate_for_vlm


@torch.no_grad()
def main(args):
	if args.average_over_clip_models == "off":
	
		if args.model_name == "clip" and args.clip_mode != "default":
			model = get_vlm(args.model_name)(model_name=args.clip_mode)

		elif args.model_name == "conclip" and args.conclip_mode != "nci":
			model_name = f"conclip_{args.conclip_mode}"
			print(model_name)
			model = get_vlm("conclip")(model_name=model_name)
		
		elif args.model_name == "laclip":
			args.model_name = "clip"
			model = get_vlm('clip')(model_name=args.clip_mode)
			laclip_mode_mapping = {
				'ViT-B/32': 'laion400m_b32',
				'ViT-B/16': 'laion400m_b16',
				'ViT-L/14': 'laion400m_l14'}
			assert args.clip_mode in laclip_mode_mapping, f'download the appropriate laclip checkpoint for the given clip mode: {args.clip_mode} and setup the eval file appropriately!!'
			laclip_mode = laclip_mode_mapping[args.clip_mode]
			checkpoint = torch.load(f'/workspace/jaisidh/btp/checkpoints/laclip_{laclip_mode}/laclip_{laclip_mode}.pt')
			model.model.load_state_dict(checkpoint['state_dict'])

		else:
			model_name = get_vlm_name(args.model_name)
			model = get_vlm(args.model_name)(model_name=model_name)

		collate_function = collate_for_vlm(args.model_name)

		if args.ccneg_mode == 1:
			dataset = CCNegEvalDataset(args)
		
		elif args.ccneg_mode == 2:
			dataset = CCNeg2EvalDataset(args)
			collate_function = dataset.collate

		loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_function)
		tqdm.write("CC-Neg loaded for evaluation")

		tqdm.write("Here we go!")
		tqdm.write(" ")

		if args.ccneg_mode == 1:
			accuracy = model.score_over_loader(loader)
		elif args.ccneg_mode == 2:
			accuracy = model.score_over_loader2(loader)
		
		tqdm.write(f"{args.model_name} Accuracy: {accuracy}")
	
	elif args.averge_over_clip_models == "on":
		model_names = ["RN50", "RN101", "ViT-B/16", "ViT-B/32", "ViT-L/14"]
		accuracies = []
		for model_name in model_names:
			clip_variant = model_name
			model = get_vlm(args.model_name)(clip_variant)
			collate_function = collate_for_vlm(args.model_name)

			dataset = CCNegEvalDataset(args)
			loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_function)
			tqdm.write("CC-Neg loaded for evaluation")

			tqdm.write("Here we go!")
			tqdm.write(" ")

			accuracy = model.score_over_loader(loader)		
			accuracies.append(accuracies)
		
		avg_accuracy = round(sum(accuracies)/len(accuracies), 2)
		tqdm.write(f"CLIP average accuracy: {avg_accuracy}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--model-name", type=str, default="clip", choices=["clip", "blip", "flava", "conclip", "negclip", 'laclip'])
	parser.add_argument("--conclip-mode", type=str, default="nci_plus3")
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-eval-samples", type=int, default=40000)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--clip-mode", type=str, default="ViT-B/32")
	parser.add_argument("--num-ops", type=int, default=-1)
	parser.add_argument("--average-over-clip-models", type=str, default="off", choices=["on", "off"])
	parser.add_argument("--ccneg-mode", type=int, default=1)

	args = parser.parse_args()
	main(args)

