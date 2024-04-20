from contextlib import suppress
import torch
import torch.nn.functional as F
import os
import sys
import argparse

src_dir = os.path.abspath("../")
sys.path.append(src_dir)

import numpy as np
from tqdm import tqdm


def setup_args():
    parser = argparse.ArgumentParser()
    
    # model args
    parser.add_argument("--clip-model-name", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lock-image-encoder", type=str, default="on")

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
    parser.add_argument("--negative-images", type=str, default="on")

    # save args
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--ckpt-save-folder", type=str, default="../checkpoints")
    parser.add_argument("--logs-save-folder", type=str, default="../logs")
    parser.add_argument("--experiment-name", type=str, default="conclip_b32")
    
    args = parser.parse_args()
    return args

def print2lines():
    print("  ")
    print("  ")

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def clip_forward_pass(model, images, texts):
    image_features = model.encode_image(images)
    image_features = F.normalize(image_features, dim=-1)

    text_features = model.encode_text(texts)
    text_features = F.normalize(text_features, dim=-1)

    return image_features, text_features, model.logit_scale.exp()

def check_and_save(args, model, optimizer, logs, epoch):
    if (epoch + 1) % args.save_every == 0:
        dump = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": args}
        
        save_name = f"ckpt_{epoch+1}_{args.experiment_name}.pt"
        save_folder = os.path.join(args.ckpt_save_folder, args.experiment_name)
        os.makedirs(save_folder, exist_ok=True)

        save_path = os.path.join(save_folder, save_name)
        torch.save(dump, save_path)

        print(f"Checkpoint saved at epoch: {epoch+1}!")
        print2lines()

def save_at_end(args, model, optimizer, logs, epoch):
    last_ckpt_path = os.path.join(
        args.ckpt_save_folder, 
        args.experiment_name, 
        f"ckpt_{epoch+1}_{args.experiment_name}.pt")

    dump = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": args}
    torch.save(dump, last_ckpt_path)

    logs_save_path = os.path.join(
        args.logs_save_folder,
        args.experiment_name,
        f"results_{args.experiment_name}.pt"
    )
    os.makedirs(os.path.join(args.logs_save_folder, args.experiment_name), exist_ok=True)
    torch.save(logs, logs_save_path)

def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text[:len(logits_per_image)]}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

@torch.no_grad()
def image_text_matching_eval(args, model, loader):
    model.eval()
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress

    all_image_features, all_text_features = [], []

    for i, batch in tqdm(enumerate(loader)):
        if i >= args.val_point:
            break
        else:
            negative_images = None
            
            if len(batch) == 4:
                images, negative_images, captions, negative_captions = batch
            else:
                images, captions, negative_captions = batch
            
            all_texts = torch.cat([captions, negative_captions], dim=0).squeeze(1).to(args.device)
            if negative_images is not None:
                all_images = torch.cat([images, negative_images], dim=0).to(args.device)
            else:
                all_images = images.to(args.device)

            with autocast():
                image_features, text_features, logit_scale = clip_forward_pass(model, all_images, all_texts)
                all_image_features.append(image_features)
                all_text_features.append(text_features)
    
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    metrics = get_metrics(image_features, text_features, logit_scale)
    return metrics
