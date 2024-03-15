from .finetuning_datasets import *
from .evaluation_datasets import *
from .imagenet_classes import *


def get_finetuning_dataset(args, preprocess):
	if args.negative_images == "off":
		return FineTuningDataset(args, preprocess)
	elif args.negative_images == "on":
		return FineTuningDatasetWithNegatives(args, preprocess)
	elif args.negative_images == "on+":
		return FineTuningDatasetWithNegatives(args, preprocess)

	