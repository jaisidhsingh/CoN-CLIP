import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels_dtype = torch.long
    
    def set_loss_type(self, new_loss_type):
        self.loss_type == new_loss_type 

    def get_labels(self, batch_size):
        labels = torch.arange(batch_size, dtype=self.labels_dtype)
        return labels
    
    def forward(self, image_features, text_features, logit_scale):
        # obtain labels for cross-entropy loss
        batch_size = image_features.shape[0]
        labels = self.get_labels(batch_size)
        labels = labels.to(image_features.device)
        
        # obtain logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features[:batch_size] @ image_features.T 

        # calculate symmetric cross-entropy loss over logits
        total_loss = (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        ) / 2
        
        return total_loss


class CustomLossWithNegatives(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels_dtype = torch.long
        self.loss_type = "negclip"
    
    def set_loss_type(self, new_loss_type):
        self.loss_type = new_loss_type
        
    def get_labels(self, batch_size):
        labels = torch.arange(batch_size, dtype=self.labels_dtype)
        return labels

    def forward(self,
                image_features,
                text_features,
                logit_scale
                ):

        batch_size = image_features.shape[0] // 2
        # 0 -> batch_size -1 = positive
        # batch_size -> 2 * batch_size -1 = negative

        # get labels for loss calculation
        labels = self.get_labels(batch_size)
        labels = labels.to(image_features.device)

        text_features_rev = torch.cat([text_features[batch_size:], text_features[:batch_size]], dim=0)

        logits1 = logit_scale * image_features[:batch_size] @ text_features.T
        logits2 = logit_scale * text_features[:batch_size] @ image_features.T
        logits3 = logit_scale * image_features[batch_size:] @ text_features_rev.T

        total_loss = (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels) + F.cross_entropy(logits3, labels))/3

        return total_loss


def get_criterion(args):
    if args.negative_images == "off":
        return CustomLoss()
    elif args.negative_images == "on":
        return CustomLossWithNegatives()
