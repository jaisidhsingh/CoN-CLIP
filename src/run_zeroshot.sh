
# echo "-----------------------------------------------------------------------------------"
# echo "LaCLIP ViT-B/16"
# echo "-----------------------------------------------------------------------------------"

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=caltech_101 \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=stanford_cars \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=flowers_102 \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=food_101 \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=oxford_pets \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=cifar10 \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=1;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=cifar100 \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=1;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=imagenet \
# 	--clip-model-name="ViT-B/16" \
# 	--batch-size=256 \
# 	--num-workers=1;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=imagenet \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=1;

echo "-----------------------------------------------------------------------------------"
echo "LaCLIP ViT-B/32"
echo "-----------------------------------------------------------------------------------"

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=caltech_101 \
# 	--clip-model-name="ViT-B/32" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=stanford_cars \
# 	--clip-model-name="ViT-B/32" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=flowers_102 \
# 	--clip-model-name="ViT-B/32" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=food_101 \
# 	--clip-model-name="ViT-B/32" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=oxford_pets \
# 	--clip-model-name="ViT-B/32" \
# 	--batch-size=256 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=cifar10 \
# 	--clip-model-name="ViT-B/32" \
# 	--batch-size=256 \
# 	--num-workers=1;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=cifar100 \
# 	--clip-model-name="ViT-B/32" \
# 	--batch-size=256 \
# 	--num-workers=1;

python3 zeroshot_classification.py \
	--use-laclip=true \
	--eval-dataset=imagenet \
	--clip-model-name="ViT-B/32" \
	--batch-size=256 \
	--num-workers=1;

# echo "-----------------------------------------------------------------------------------"
# echo "LaCLIP ViT-L/14"
# echo "-----------------------------------------------------------------------------------"

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=caltech_101 \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=stanford_cars \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=flowers_102 \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=food_101 \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=oxford_pets \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=4;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=cifar10 \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=1;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=cifar100 \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=1;

# python3 zeroshot_classification.py \
# 	--use-laclip=true \
# 	--eval-dataset=imagenet \
# 	--clip-model-name="ViT-L/14" \
# 	--batch-size=128 \
# 	--num-workers=1;
