# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=clip \
# 	--ccneg-mode=2;

# # python3 ccneg_eval_utils/eval_ccneg.py \
# # 	--model-name=flava \
# # 	--ccneg-mode=2;

# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=negclip \
# 	--ccneg-mode=2;

python3 ccneg_eval_utils/eval_ccneg.py \
	--model-name=laclip \
	--clip-mode="ViT-B/16" \
	--ccneg-mode=1;
python3 ccneg_eval_utils/eval_ccneg.py \
	--model-name=laclip \
	--clip-mode="ViT-B/32" \
	--ccneg-mode=1;
python3 ccneg_eval_utils/eval_ccneg.py \
	--model-name=laclip \
	--clip-mode="ViT-L/14" \
	--ccneg-mode=1;

# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=blip \
# 	--ccneg-mode=2;

# echo "----------------------------------------------------------------------------------";
# echo "Flava accuracies are given above for # object-predicate pairs: 1-5 (top to bottom)";
# echo "----------------------------------------------------------------------------------";

# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=blip \
# 	--num-ops=1 \
# 	--batch-size=256;

# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=blip \
# 	--num-ops=2 \
# 	--batch-size=256;

# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=blip \
# 	--num-ops=3 \
# 	--batch-size=256;

# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=blip \
# 	--num-ops=4 \
# 	--batch-size=256;

# python3 ccneg_eval_utils/eval_ccneg.py \
# 	--model-name=blip \
# 	--num-ops=5 \
# 	--batch-size=256;

# echo "----------------------------------------------------------------------------------";
# echo "Blip accuracies are given above for # object-predicate pairs: 1-5 (top to bottom)";
# echo "----------------------------------------------------------------------------------";

