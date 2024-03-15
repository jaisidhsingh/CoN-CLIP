from types import SimpleNamespace


configs = SimpleNamespace(**{})

# fine-tuning dataset configs
configs.finetuning_dataset_path = "/workspace/datasets/ccneg/preprocessed_data.pt"
configs.negative_image_dataset_root = "/workspace/datasets/coco/coco_train2017/train2017"
configs.negative_image_dataset_annotations_path = "/workspace/datasets/coco/coco_ann2017/annotations/captions_train2017.json"
configs.negative_image_ft_mapping_path = "/workspace/datasets/cc3m/final_conclip/negative_image_mapping.pt"

# zero-shot image classification dataset configs
configs.zeroshot_data_loader = lambda x: f"/workspace/datasets/{x}/preprocessed_data.pt"
configs.num_ccneg_eval_samples = 40000  # -- the last 40,000 indices consist of the decided evaluation split for CC-Neg
