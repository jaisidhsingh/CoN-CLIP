from types import SimpleNamespace


configs = SimpleNamespace(**{})
configs.vlm_name_mapping = {
	"clip": "ViT-B/16",
	"conclip": "conclip_nci",
	"blip": "Salesforce/blip-itm-base-coco",
	"flava": "facebook/flava-full",
	"negclip": "ViT-B/32"
}