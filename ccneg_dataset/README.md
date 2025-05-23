# Downloading the CC-Neg Dataset

## CC-Neg: Images

The images for CC-Neg come from the ImageLabels split of the CC-3M which we prepare and provide <a href="https://drive.proton.me/urls/CPD0RPRVA4#lDBGak33rXEn">here</a>. Please find a compressed file called `ccneg_images.zip` in this directory, download, and extract the images. Verify that the structure of the `ccneg_images` folder becomes

```plaintext
ccneg_images
|___ cc3m_subset_images_extracted_final
	 |___ image1.jpg
         |___ image2.jpg
         ...

```

## CC-Neg: Annotations

The annotations containing the true caption and the negated (false) caption for each image in CC-Neg can be downloaded from <a href="https://drive.proton.me/urls/GC34W9VACG#xitsiMVh4HUU">here</a>. This file, named `ccneg_preprocessed.pt` must be downloaded into this directory. The helper for using distractor images during fine-tuning is provided <a href="https://drive.proton.me/urls/JAC3FBXP58#Yi2bmdl0xodv">here</a>, named `distractor_image_mapping.pt`.

## Paths for Data Configs

This directory is specified in configs given in <a href="../src/configs/__init__.py">`src/configs/__init__.py`</a> which is accessed by the <a href="../src/data">`src/data`</a> folder. Here, the <a href="../src/data/evaluation_datasets.py">`src/data/evaluation_datasets.py`</a> and <a href="../src/data/finetuning_datasets.py">`src/data/finetuning_datasets.py`</a> use the configs to load in the dataset. For finetuning CLIP to get CoN-CLIP, we use MS-COCO along with CC-Neg. The root folders for both these datasets must be specificed in <a href="../src/configs/__init__.py">`src/configs/__init__.py`</a>. Be sure to check this before running code which utilizes CC-Neg.
