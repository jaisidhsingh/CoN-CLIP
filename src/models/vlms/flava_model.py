import torch
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor
from transformers import logging
import numpy as np
logging.set_verbosity_warning()

from sklearn.metrics import roc_auc_score

def calculate_auc(labels, preds):
    auc_score = roc_auc_score(labels, preds)
    return auc_score


class FlavaRunner():
	def __init__(self, model_name):
		self.model_name = model_name
		self.device = "cuda"
		self.seq_len = 77
		self.model = FlavaForPreTraining.from_pretrained(self.model_name).eval().to(self.device)
		self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
		self.processor = FlavaProcessor.from_pretrained(self.model_name)
		self.feature_extractor = FlavaFeatureExtractor.from_pretrained(self.model_name)

	@torch.no_grad()
	def embed_image(self, x):
		batch_size = len(x)
		inputs = self.feature_extractor(images=x, return_tensors="pt").to(self.device)
		image_features = self.model.flava.get_image_features(**inputs)[:, 0, :]
		# embed_dim = image_features.shape[-1]
		# image_features = image_features.view(batch_size, embed_dim)
		image_features /= image_features.norm(dim=-1, keepdim=True)
		return image_features

	@torch.no_grad()
	def embed_text(self, x):
		inputs = self.tokenizer(text=x, return_tensors="pt", padding="max_length", max_length=self.seq_len).to(self.device)
		text_features = self.model.flava.get_text_features(**inputs)[:, 0, :]
		text_features /= text_features.norm(dim=-1, keepdim=True)
		return text_features

	@torch.no_grad()
	def score_over_loader(self, loader):
		bar = tqdm(total=len(loader))
		correct, total = 0, 0

		for idx, (images, captions, negative_captions) in enumerate(loader):
			batch_size = len(images)

			# send inputs to GPU	
			# images = images.to(self.device)

			# forward pass
			image_features = self.embed_image(images)
			caption_features = self.embed_text(captions)
			negative_caption_features = self.embed_text(negative_captions)

			# get similarities across batch
			sim1 = image_features @ caption_features.T # shape: BxB
			sim2 = image_features @ negative_caption_features.T # shape: BxB

			# get similarities of only (I_i, c_i) and (I_i, c'_i)
			preds1, preds2 = torch.diag(sim1).view(batch_size, 1), torch.diag(sim2).view(batch_size, 1) # each has shape: Bx1
			preds_combined = torch.cat([preds1, preds2], dim=1).to(self.device)
			preds = preds_combined.argmax(dim=1) # shape: Bx1

			# true caption is at index 0 and negated caption is at index 1
			labels = torch.zeros(batch_size).long().to(self.device)
			correct += (preds == labels).sum().item()
			total += batch_size
			
			# get accuracy of matching to the true caption c_i over c'_i for a particular I_i
			accuracy = round(correct/total, 4) * 100

			bar.update(1)
			bar.set_postfix({"batch_index": idx, "accuracy": accuracy})

		# log results	
		tqdm.write(f"Accuracy of Flava {self.model_name}: {accuracy}")	

	@torch.no_grad()
	def score_over_loader2(self, loader):
		bar = tqdm(total=len(loader))
		correct, total = [0, 0], [0, 0]

		all_probs = []
		all_labels = []
		for idx, (positive_images, negative_images, positive_captions, negative_captions) in enumerate(loader):
			batch_size = len(positive_images)

			# send inputs to GPU	
			# images = images.to(self.device)

			# forward pass
			for polarity, images in enumerate([positive_images, negative_images]):
				image_features = self.embed_image(images)
				caption_features = self.embed_text(positive_captions)
				negative_caption_features = self.embed_text(negative_captions)

				# get similarities across batch
				sim1 = image_features @ caption_features.T # shape: BxB
				sim2 = image_features @ negative_caption_features.T # shape: BxB

				# get similarities of only (I_i, c_i) and (I_i, c'_i)
				preds1, preds2 = torch.diag(sim1).view(batch_size, 1), torch.diag(sim2).view(batch_size, 1) # each has shape: Bx1
				preds_combined = torch.cat([preds1, preds2], dim=1).to(self.device)
				probs = preds_combined.softmax(dim=1)[:, 1].to(self.device)
				preds = preds_combined.argmax(dim=1) # shape: Bx1

				# true caption is at index 0 and negated caption is at index 1
				labels = torch.zeros(batch_size).long().to(self.device) + polarity
				correct[polarity] += (preds == labels).sum().item()
				total[polarity] += batch_size
				all_probs.append(probs.detach().cpu().numpy())
				all_labels.append(labels.detach().cpu().numpy())
			
			# get accuracy of matching to the true caption c_i over c'_i for a particular I_i
			pos_accuracy = round(correct[0]/total[0], 4) * 100
			neg_accuracy = round(correct[1]/total[1], 4) * 100
			avg_accuracy = round( (correct[0]/total[0] + correct[1]/total[1]) / 2, 4) * 100

			bar.update(1)
			bar.set_postfix({"batch_index": idx,  "avg_accuracy": avg_accuracy, "pos_retrieval_accuracy": pos_accuracy, "neg_retrieval_accuracy": neg_accuracy,})
		
		bar.close()
		
		#calculate auc score
		all_labels, all_probs = [np.concatenate(arr) for arr in [all_labels, all_probs]]
		auc_score = calculate_auc(all_labels, all_probs)

		# log results
		tqdm.write(f"Retrieval Accuracy of Flava [**for Positives**]{self.model_name}: {pos_accuracy}")
		tqdm.write(f"Retrieval Accuracy of Flava [**for Negatives**]{self.model_name}: {neg_accuracy}")
		tqdm.write(f"Net accuracy: {avg_accuracy}")
		tqdm.write(f"AUC score: {auc_score}")

