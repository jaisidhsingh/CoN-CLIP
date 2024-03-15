import torch

from configs.model_configs import configs
from .clip_model import ClipRunner
from .conclip_model import CoNClipRunner
from .flava_model import FlavaRunner
from .blip_model import BlipRunner
from .negclip_model import NegClipRunner


"""
dataset returns a batch of (images, captions, negative_captions) where
images: [pil_image_1, pil_image_2, ...]
captions:  [caption_1, caption_2, ...]
negative_captions: [negative_caption_1, negative_caption_2, ...]
"""

def clip_collate(batch):
	images = [item[0] for item in batch]
	captions = [item[1][0] for item in batch]
	negative_captions = [item[2][0] for item in batch]
	return (images, captions, negative_captions)

def conclip_collate(batch):
	images = [item[0] for item in batch]
	captions = [item[1][0] for item in batch]
	negative_captions = [item[2][0] for item in batch]
	return (images, captions, negative_captions)

def flava_collate(batch):
	images = [item[0] for item in batch]
	captions = [item[1][0] for item in batch]
	negative_captions = [item[2][0] for item in batch]
	return (images, captions, negative_captions)

def blip_collate(batch):
	images = [item[0] for item in batch]
	captions = [item[1][0] for item in batch]
	negative_captions = [item[2][0] for item in batch]
	return (images, captions, negative_captions)

def collate_for_vlm(model_name):
	model_mapping = {
		"clip": clip_collate,
		"conclip": conclip_collate,
		"blip": blip_collate,
		"flava": flava_collate,
		"negclip": clip_collate
	}
	return model_mapping[model_name]

def get_vlm_name(model_name):
	return configs.vlm_name_mapping[model_name]

def get_vlm(model_name):
	model_mapping = {
		"clip": ClipRunner,
		"conclip": CoNClipRunner,
		"blip": BlipRunner,
		"flava": FlavaRunner,
		"negclip": NegClipRunner
	}
	return model_mapping[model_name]
