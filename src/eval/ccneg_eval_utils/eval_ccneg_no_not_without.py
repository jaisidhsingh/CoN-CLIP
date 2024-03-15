import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import clip
import os
import argparse
import sys
from tqdm import tqdm

src_dir = os.path.abspath("../")
sys.path.append(src_dir)

from data.evaluation_datasets import CCNegEvalDataset
from models import get_vlm, get_vlm_name, collate_for_vlm

def get_subset(args, dataset):
	index_list = []
	for idx in range(len(dataset)):
		neg_caption = dataset.annotations[idx]['sop_data']['negative-prompt']
		neg_caption = ' ' + neg_caption + ' '
		for nword in args.negate_words:
			nword = ' ' + nword + ' '
			if nword in neg_caption:
				index_list.append(idx)
				break
	index_list = torch.Tensor(index_list).long()
	subset = Subset(dataset, index_list)
	return subset

@torch.no_grad()
def main(args):
	if args.average_over_clip_models == "off":
	
		if args.model_name == "clip" and args.clip_mode != "default":
			model = get_vlm(args.model_name)(model_name=args.clip_mode)

		elif args.model_name == "conclip" and args.conclip_mode != "nci":
			model_name = f"conclip_{args.conclip_mode}"
			model = get_vlm(args.model_name)(model_name=model_name)

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

		dataset = CCNegEvalDataset(args)
		dataset = get_subset(args, dataset)#!

		loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_function)
		tqdm.write(f"CC-Neg loaded for evaluation \n- subset of size {len(dataset)} \n- negate words {args.negate_words}")

		tqdm.write("Here we go!")
		tqdm.write(" ")

		accuracy = model.score_over_loader(loader)
	
	elif args.averge_over_clip_models == "on":
		model_names = ["RN50", "RN101", "ViT-B/16", "ViT-B/32", "ViT-L/14"]
		accuracies = []
		for model_name in model_names:
			clip_variant = model_name
			model = get_vlm(args.model_name)(clip_variant)
			collate_function = collate_for_vlm(args.model_name)

			dataset = CCNegEvalDataset(args)
			dataset = get_subset(args, dataset)#!

			loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_function)
			tqdm.write(f"CC-Neg loaded for evaluation \n- subset of size {len(dataset)} \n- negate words {args.negate_words}")

			tqdm.write("Here we go!")
			tqdm.write(" ")

			accuracy = model.score_over_loader(loader)		
			accuracies.append(accuracies)
		
		avg_accuracy = round(sum(accuracies)/len(accuracies), 2)
		tqdm.write(f"CLIP average accuracy: {avg_accuracy}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--model-name", type=str, default="clip", choices=["clip", "blip", "flava", "conclip", "negclip", "laclip"])
	parser.add_argument("--conclip-mode", type=str, default="nci", choices=["nci_plus3", "nci", "nc", "ft", "loss1_b32", "loss2_b32", "loss12_b32"])
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--num-eval-samples", type=int, default=40000)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--clip-mode", type=str, default="ViT-B/32")
	parser.add_argument("--num-ops", type=int, default=-1)
	parser.add_argument("--average-over-clip-models", type=str, default="off", choices=["on", "off"])
	parser.add_argument('--negate-words', type=str, nargs='+')

	args = parser.parse_args()
	main(args)

