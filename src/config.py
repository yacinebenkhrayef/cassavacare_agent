import os

CFG = {
    # Model
    "model_name"          : "tf_efficientnetv2_s",
    "num_classes"         : 5,
    "dropout"             : 0.6,
    "img_size"            : 224,

    # Training
    "batch_size"          : 32,
    "epochs"              : 25,
    "warmup_epochs"       : 5,
    "patience"            : 10,
    "lr"                  : 5e-4,
    "weight_decay"        : 5e-4,
    "clip_grad"           : 1.0,
    "mixup_alpha"         : 0.2,
    "seed"                : 42,

    # DataLoader
    "num_workers"         : 4,
    "pin_memory"          : True,
    "persistent_workers"  : True,
    "tta_augs"            : 5,

    # Paths (update base_img_dir if dataset is local)
    "base_img_dir"        : "/kaggle/input/datasets/nirmalsankalana/cassava-leaf-disease-classification/data",
    "splits_dir"          : "data/processed",
    "label_map_path"      : "data/processed/label_num_to_disease_map.json",
    "class_weights_path"  : "data/processed/class_weights.json",
    "checkpoint_path"     : "models/best_model_efficientnetv2_s.pth",
    "scripted_path"       : "models/best_model_scripted_efficientnetv2_s.pt",
    "metrics_path"        : "logs/training_metrics_efficientnetv2_s.csv",

    # W&B
    "wandb_project"       : "cassava-leaf-disease",
    "wandb_run_name"      : "efficientnetv2_s_224_strong_reg_tta",
    "wandb_save_model"    : True,
    "wandb_log_images"    : True,
}

# Label mapping (mirrors label_num_to_disease_map.json)
LABEL_MAP = {
    0: "bacterial_blight",
    1: "brown_streak_disease",
    2: "green_mottle",
    3: "healthy",
    4: "mosaic_disease",
}

SHORT_NAMES = ["CBB", "CBSD", "CGM", "Healthy", "CMD"]
CLASS_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#D4537E", "#888780"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
