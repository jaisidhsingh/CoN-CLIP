import sys
import numpy as np
import torch
from transformers import AutoProcessor, BlipForImageTextRetrieval
from transformers import logging
from tqdm import tqdm
logging.set_verbosity_warning()
import numpy as np

from sklearn.metrics import roc_auc_score

def calculate_auc(labels, preds):
    auc_score = roc_auc_score(labels, preds)
    return auc_score

class BlipRunner():
	def __init__(self, model_name):
		self.model_name = model_name
		self.device = "cuda"
		self.seq_len = None
		self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name).eval().to(self.device)
		self.processor = AutoProcessor.from_pretrained(self.model_name)

		self.return_dict = self.model.config.use_return_dict
		self.output_attentions = self.model.config.output_attentions
		self.output_hidden_states = self.model.config.output_hidden_states

	@torch.no_grad()
	def embed_image(self, pixel_values):
		# assume that ``processor`` is applied already
		vision_outputs = self.model.vision_model(
			pixel_values=pixel_values,
			output_attentions=self.output_attentions,
			output_hidden_states=self.output_hidden_states,
			return_dict=self.return_dict,
		)

		image_embeds = vision_outputs[0]
		image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
		image_features = self.model.vision_proj(image_embeds[:, 0, :])
		image_features /= image_features.norm(dim=-1, keepdim=True)
		return image_features

	@torch.no_grad()
	def embed_text(self, inputs):
		# assume that ``processor`` is applied already
		input_ids = inputs["input_ids"]
		input_ids = torch.from_numpy(np.array(input_ids)).to(self.device)
		attention_mask = inputs["attention_mask"]
		attention_mask = torch.from_numpy(np.array(attention_mask)).to(self.device)

		# blip's image-text matching head is not used; cosine similarity of features is used directly.
		question_embeds = self.model.text_encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			return_dict=self.return_dict,
		)
		question_embeds = question_embeds[0] if not self.return_dict else question_embeds.last_hidden_state
		text_features = self.text_proj(question_embeds[:, 0, :])
		text_features /= text_features.norm(dim=-1, keepdim=True)
		return text_features

	@torch.no_grad()
	def score_over_loader(self, loader):
		bar = tqdm(total=len(loader))
		correct, total = 0, 0

		for idx, (images, captions, negative_captions) in enumerate(loader):
			batch_size = len(images)

			# captions = captions[:100]
			# TODO: check GPU casting
			# apply the processor that huggingface needs
			true_inputs = self.processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
			negative_inputs = self.processor(images=images, text=negative_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

			true_outputs = self.model(**true_inputs)
			negative_outputs = self.model(**negative_inputs)
			preds1, preds2 = true_outputs["itm_score"], negative_outputs["itm_score"]
			preds1, preds2 = preds1.softmax(dim=1)[:, 1].view(batch_size, 1), preds2.softmax(dim=1)[:, 1].view(batch_size, 1) # this step seems iffy
			preds_combined = torch.cat([preds1, preds2], dim=1).to(self.device)
			preds = preds_combined.argmax(dim=1)

			# true caption is at index 0 and negated caption is at index 1
			labels = torch.zeros(batch_size).long().to(self.device)
			correct += (preds == labels).sum().item()
			total += batch_size

			# get accuracy of matching to the true caption c_i over c'_i for a particular I_i
			accuracy = round(correct/total, 4) * 100

			bar.update(1)
			bar.set_postfix({"batch_index": idx, "accuracy": accuracy})
		
		# log results
		tqdm.write(f"Accuracy of BLIP {self.model_name}: {accuracy}")


	@torch.no_grad()
	def score_over_loader2(self, loader):
		bar = tqdm(total=len(loader))
		correct = [0, 0]
		total = [0, 0]

		all_probs = []
		all_labels = []
		for idx, (positive_images, negative_images, positive_captions, negative_captions) in enumerate(loader):
			batch_size = len(positive_images)

			# captions = captions[:100]
			# TODO: check GPU casting
			# apply the processor that huggingface needs
			for polarity, images in enumerate([positive_images, negative_images]):
				true_inputs = self.processor(images=images, text=positive_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)
				negative_inputs = self.processor(images=images, text=negative_captions, return_tensors="pt", padding=True, truncation=True).to(self.device)

				true_outputs = self.model(**true_inputs)
				negative_outputs = self.model(**negative_inputs)
				preds1, preds2 = true_outputs["itm_score"], negative_outputs["itm_score"]
				preds1, preds2 = preds1.softmax(dim=1)[:, 1].view(batch_size, 1), preds2.softmax(dim=1)[:, 1].view(batch_size, 1) # this step seems iffy
				preds_combined = torch.cat([preds1, preds2], dim=1).to(self.device)
				probs = preds_combined.softmax(dim=1)[:, 1].to(self.device)
				preds = preds_combined.argmax(dim=1)

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
		tqdm.write(f"Retrieval Accuracy of BLIP [**for Positives**]{self.model_name}: {pos_accuracy}")
		tqdm.write(f"Retrieval Accuracy of BLIP [**for Negatives**]{self.model_name}: {neg_accuracy}")
		tqdm.write(f"Net accuracy: {avg_accuracy}")
		tqdm.write(f"AUC score: {auc_score}")