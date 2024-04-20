python3 conclip_fine_tuning.py \
	--clip-model-name=ViT-B/32 \
	--experiment-name=conclip_b32 \
	--negative-images=on \
	--lock-image-encoder=on \
	--batch-size=200 \
	--num-workers=4
