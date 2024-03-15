# python3 sugar_crepe_utils/main_eval.py \
# 	--model ViT-B/32 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
# 	--data_root ./sugar_crepe_utils/data \
# 	--ckpt /workspace/jaisidh/btp/checkpoints/pretrained_vlms/negclip_b32.pth;

# python3 sugar_crepe_utils/main_eval.py \
# 	--model ViT-B/32 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
# 	--data_root ./sugar_crepe_utils/data \
# 	--ckpt /workspace/jaisidh/btp/checkpoints/conclip_nci_plus3/ckpt_5_conclip_nci_plus3.pt;

echo "------------------------------------------------------------------------------------------";

python3 sugar_crepe_utils/main_eval.py \
	--model ViT-B/16 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
	--data_root ./sugar_crepe_utils/data \
	--ckpt /workspace/jaisidh/btp/checkpoints/laclip_laion400m_b16/laclip_laion400m_b16.pt;

echo "------------------------------------------------------------------------------------------";

# python3 sugar_crepe_utils/main_eval.py \
# 	--model ViT-B/32 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
# 	--data_root ./sugar_crepe_utils/data \
# 	--ckpt /workspace/jaisidh/btp/checkpoints/laclip_laion400m_b32/laclip_laion400m_b32.pt;

# echo "------------------------------------------------------------------------------------------";

python3 sugar_crepe_utils/main_eval.py \
	--model ViT-L/14 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
	--data_root ./sugar_crepe_utils/data \
	--ckpt /workspace/jaisidh/btp/checkpoints/laclip_laion400m_l14/laclip_laion400m_l14.pt;

# python3 sugar_crepe_utils/main_eval.py \
# 	--model ViT-B/32 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
# 	--data_root ./sugar_crepe_utils/data \
# 	--ckpt /workspace/jaisidh/btp/checkpoints/crepe_clip_b32/ckpt_13_crepe_clip_b32.pt;
	
# /workspace/jaisidh/btp/checkpoints/conclip_nci_plus2/ckpt_5_conclip_nci_plus2.pt;

# python3 sugar_crepe_utils/main_eval.py \
# 	--model ViT-B/32 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
# 	--data_root ./sugar_crepe_utils/data \
# 	--ckpt /workspace/jaisidh/btp/checkpoints/conclip_nci_plus3/ckpt_5_conclip_nci_plus3.pt;

# python3 sugar_crepe_utils/main_eval.py \
# 	--model ViT-L/14 --output /workspace/jaisidh/btp/src/eval/sugar_crepe_utils/outputs \
# 	--data_root ./sugar_crepe_utils/data \
# 	--ckpt /workspace/jaisidh/btp/checkpoints/conclip_nci_plus_l14/ckpt_5_conclip_nci_plus_l14.pt;

