import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import os
import sys
from tqdm import tqdm
from training_utils.loss_functions import *
from training_utils.schedulers import *
from data import *
from training_utils.helpers import *


def train_one_epoch(args, epoch, model, loader, optimizer, criterion, scheduler, scaler, early_exit=-1):
    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress

    bar = tqdm(total=len(loader))
    running_loss = 0
    
    for i, batch in enumerate(loader):
        step = len(loader) * epoch + i
        scheduler(step) 

        negative_images, subjects, negated_objects = None, None, None
        # if len(batch) == 6:
        #     images, negative_images, captions, negative_captions, subjects, negated_objects = batch
        if len(batch) == 4:
            images, negative_images, captions, negative_captions = batch
        elif len(batch) == 3:
            images, captions, negative_captions = batch

        # concate texts conditionally
        if subjects is not None and negated_objects is not None:
            all_texts = torch.cat([captions, negative_captions, subjects, negated_objects], dim=0).squeeze(1)

        else:
            all_texts = torch.cat([captions, negative_captions], dim=0).squeeze(1)

        # put text inputs on GPU        
        all_texts = all_texts.to(args.device, non_blocking=True)

        if negative_images is not None:
            all_images = torch.cat([images, negative_images], dim=0).to(args.device, non_blocking=True)
        else:
            all_images = images.to(args.device, non_blocking=True)

        # zero grads before forward pass
        optimizer.zero_grad()

        # forward pass 
        with autocast():
            image_features, text_features, logit_scale = clip_forward_pass(model, all_images, all_texts)
            loss = criterion(image_features, text_features, logit_scale)
            running_loss += loss.item()

        # backward pass if scaler is on
        if scaler is not None:
            scaler.scale(loss).backward()

            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)            
            scaler.step(optimizer)
            scaler.update()
        
        # backward pass if scaler is off        
        else:
            loss.backward()

            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)            
            optimizer.step()

        # clamp to 4.6052 = ln(100) as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100)) 

        # make logs after this
        avg_loss = running_loss / (i+1)
        logs = {"epoch": epoch+1, "avg_loss": avg_loss}
        bar.update(1)
        bar.set_postfix(logs)
        
        if early_exit == i:
            return logs 
    
    return logs


def zero_shot_eval(args, model, epoch, preprocess):
    model.eval()
    dataset, loader = setup_zero_shot(args, preprocess) 
    categories_tokenized = clip.tokenize(dataset.class_names).to(args.device)

    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    bar = tqdm(total=len(loader))

    with torch.no_grad():
        category_features = model.encode_text(categories_tokenized)
        category_features /= category_features.norm(dim=-1, keepdim=True)

        correct, total = 0, 0        
        for (images, labels) in loader:
            images = images.to(args.device)
            labels = labels.long().to(args.device)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.item()

            logits = (logit_scale * image_features @ category_features.T).softmax(dim=-1)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            accuracy = round(correct/total * 100, 2)

            bar.update(1)
            bar.set_postfix({"eval_at_epoch": epoch+1, "accuracy": accuracy})
        
        tqdm.write(f"Zero-shot image classification")
        tqdm.write(f"Accuracy = {accuracy} --- Epoch = {epoch+1}")
        tqdm.write("  ")

        return {"epoch": epoch+1, "zero_shot_accuracy": accuracy}


def train(args):
    tqdm.write("Training...")
    model, preprocess = clip.load(args.clip_model_name, device=args.device)
    
    if args.precision == "amp":
        model = model.float()
        model = model.to(args.device)

    # lock the visual encoder
    if args.lock_image_encoder == "on":
        for param in model.visual.parameters():
            param.requires_grad = False
        tqdm.write("Locked the visual encoder. Only the text encoder will be fine-tuned.")

    model.train()
    tqdm.write(f"CLIP: {args.clip_model_name} loaded and set to train")
    
    dataset = get_finetuning_dataset(args, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    tqdm.write("Fine-tuning dataset loaded")

    num_steps_per_epoch = len(loader)
    total_steps = args.epochs * num_steps_per_epoch
    
    criterion = get_criterion(args)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    logs = {"fine_tuning": {}, "zero_shot_eval": {}}

    tqdm.write("Here we go!")
    tqdm.write(" ")

    # start training loop
    for epoch in range(args.epochs):
        # fine-tune every epoch and get logs
        training_logs = train_one_epoch(args, epoch, model, loader, optimizer, criterion, scheduler, scaler, early_exit=args.early_exit)

        # zero shot image classification is done for evaluation every "args.evaluate_every" epochs
        eval_logs = None
        if (epoch + 1) % args.evaluate_every == 0:
            eval_logs = zero_shot_eval(args, model, epoch, preprocess)

        # store logs
        logs["fine_tuning"][f"epoch_{epoch+1}"] = training_logs
        if eval_logs is not None:
            logs["zero_shot_eval"][f"epoch_{epoch+1}"] = eval_logs
        else:
            logs["zero_shot_eval"][f"epoch_{epoch+1}"] = {"epoch": epoch+1, "accuracy": -1}

        # checkpoint of (model, optimizer, logs) is stored every "args.save_every" epochs
        check_and_save(args, model, optimizer, logs, epoch)

    # sanity check so that last checkpoint is saved no matter what
    save_at_end(args, model, optimizer, logs, epoch)

    # phew
    tqdm.write("Done!")


if __name__ == "__main__":
    args = setup_args()
    train(args)
