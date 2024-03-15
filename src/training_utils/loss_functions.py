import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels_dtype = torch.long
        self.loss_type = "negclip"
    
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
        # logits_per_image = logit_scale * image_features @ text_features[:batch_size].T
        
        if self.loss_type == "ours":
            logits_per_text = logit_scale * text_features @ image_features.T # ours
        elif self.loss_type == "negclip":
            logits_per_text = logit_scale * text_features[:batch_size] @ image_features.T # like Neg-CLIP

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

        logits_per_image = logit_scale * image_features[:batch_size] @ text_features.T
        logits_per_text1 = logit_scale * text_features[:batch_size] @ image_features[:batch_size].T

        logits_per_text2 = logit_scale * text_features[:batch_size] @ image_features.T

        loss1, loss2 = 0, 0
        # loss1 = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text1, labels))/2
        loss2 = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text2, labels) + F.cross_entropy(logits_per_text1, labels))/3

        # total_loss = (loss1 + loss2)/2
        total_loss = loss2
        return total_loss


class CustomLossWithNegativesDis(nn.Module):
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

        text_features_flipped = torch.cat([text_features[batch_size:], text_features[:batch_size]], dim=0).to(text_features.device)

        # get labels for loss calculation
        labels = self.get_labels(batch_size)
        labels = labels.to(image_features.device)

        # loss 1
        # logits_per_image1 = logit_scale * image_features[:batch_size] @ text_features.T

        # # loss 2
        logits_per_text2 = logit_scale * text_features[:batch_size] @ image_features.T

        # # loss 3
        # logits_per_image2 = logit_scale * image_features[batch_size:] @ text_features_flipped.T

        total_loss = (
            # F.cross_entropy(logits_per_image1, labels) + \
            F.cross_entropy(logits_per_text2, labels)  + \
            F.cross_entropy(logits_per_image2, labels)
        )/2
        return total_loss


def get_criterion(args):
    if args.negative_images == "off":
        return CustomLoss()
    elif args.negative_images == "on":
        return CustomLossWithNegatives()
    elif args.negative_images == "on+":
        return CustomLossWithNegativesDis() 


"""
DUMP:
======
        # # make negative features
        # negative_image_features = torch.cat([image_features[batch_size:], image_features[:batch_size]], dim=0)
        # negative_text_features = torch.cat([text_features[batch_size:], text_features[:batch_size]], dim=0)

        # # positive logits
        # logits_per_image_positive = logit_scale * image_features[:batch_size] @ text_features.T

        # if self.loss_type == "ours":
        #     logits_per_text_positive = logit_scale * text_features[:batch_size] @ image_features.T # ours
        # elif self.loss_type == "negclip":
        #     logits_per_text_positive = logit_scale * text_features[:batch_size] @ image_features[:batch_size].T # like Neg-CLIP

        # # negative logits
        # logits_per_image_negative = logit_scale * image_features[batch_size:] @ negative_text_features.T

        # if self.loss_type == "ours":
        #     logits_per_text_negative = logit_scale * text_features[batch_size:] @ negative_image_features # ours
        # elif self.loss_type == "negclip":
        #     logits_per_text_negative = logit_scale * text_features[batch_size:] @ image_features[batch_size:].T # like Neg-CLIP

        # # get loss
        # loss_positive = (F.cross_entropy(logits_per_image_positive, labels) + F.cross_entropy(logits_per_text_positive, labels)) / 2
        # loss_negative = (F.cross_entropy(logits_per_image_negative, labels) + F.cross_entropy(logits_per_text_negative, labels)) / 2

        # total_loss = (loss_positive + loss_negative) * 0.5
        # return total_loss

"""
