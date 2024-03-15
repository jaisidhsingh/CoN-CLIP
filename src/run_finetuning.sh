python3 crepe_finetuning.py \
	--clip-model-name=ViT-B/32 \
	--experiment-name=crepe_clip_b32 \
	--negative-images=off \
	--lock-image-encoder=on \
	--batch-size=200 \
	--num-workers=4
