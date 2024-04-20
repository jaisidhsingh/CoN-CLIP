from types import SimpleNamespace


configs = SimpleNamespace(**{})


## CC-Neg (for fine-tuning and evaluation)
configs.ccneg_root_folder = "../../ccneg_dataset"         ## change this accordingly
configs.finetuning_dataset_path = f"{configs.ccneg_root_folder}/ccneg_preprocessed.pt"
configs.negative_image_ft_mapping_path = f"{configs.ccneg_root_folder}/distractor_image_mapping.pt"
configs.num_ccneg_eval_samples = 40000  ## the last 40,000 indices consist of the decided evaluation split for CC-Neg

## MS-COCO (for finetuning)
configs.coco_root_folder = "/workspace/datasets/coco"           ## change this accordingly
configs.negative_image_dataset_root = f"{configs.coco_root_folder}/coco_train2017/train2017"    ## path to coco train2017 images
configs.negative_image_dataset_annotations_path = f"{configs.coco_root_folder}/coco_ann2017/annotations/captions_train2017.json"    ## path to coco train2017 annotations (not used but required by torchvision dataset)
