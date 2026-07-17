# src/model.py
import torch
import torch.nn as nn
import timm


class EfficientNetCassava(nn.Module):
    """EfficientNetV2-S fine-tuned for 5-class cassava leaf disease classification.

    Architecture:
      - Backbone: tf_efficientnetv2_s (pretrained on ImageNet)
      - Freezing: all layers frozen except last 3 blocks + classifier
      - Head: Dropout(0.6) + Linear(in_features, num_classes)
      - Grad-CAM hooks on target_layer (last block)
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.6,
                 pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model("tf_efficientnetv2_s",
                                       pretrained=pretrained)

        # Freeze entire backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last 3 blocks (blocks[-3:])
        n_blocks = len(self.model.blocks)
        for i in range(n_blocks - 3, n_blocks):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True

        # Replace classifier head
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # Grad-CAM target
        self.target_layer = self.model.blocks[-1]
        self.gradients = None
        self.activations = None
        self._hooks: list = []

    # ── Grad-CAM hooks ─────────────────────────────────────
    def _activations_hook(self, module, input, output):
        self.activations = output

    def _gradients_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def register_gradcam_hooks(self):
        h1 = self.target_layer.register_forward_hook(self._activations_hook)
        h2 = self.target_layer.register_full_backward_hook(self._gradients_hook)
        self._hooks = [h1, h2]

    def remove_gradcam_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(cfg: dict, register_hooks: bool = True) -> EfficientNetCassava:
    """Instantiate model from CFG dict and print parameter counts."""
    model = EfficientNetCassava(
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
        pretrained=True,
    )
    if register_hooks:
        model.register_gradcam_hooks()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
    return model


def load_model(checkpoint_path: str, cfg: dict,
               device: torch.device = None) -> EfficientNetCassava:
    """Load a trained checkpoint. Safe for inference (no Grad-CAM hooks)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Always instantiate a standard Python-based EfficientNetCassava model
    model = EfficientNetCassava(
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
        pretrained=False,
    )

    # 2. Attempt to load the file
    try:
        loaded_obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        loaded_obj = torch.jit.load(checkpoint_path, map_location=device)

    # 3. Extract the raw state_dict (parameters)
    if isinstance(loaded_obj, torch.jit.ScriptModule) or hasattr(loaded_obj, "forward"):
        # Extract weights from the TorchScript module container
        raw_state_dict = loaded_obj.state_dict()
    else:
        raw_state_dict = loaded_obj

    # 4. Align prefix names in case of discrepancies
    model_keys = set(model.state_dict().keys())
    state_keys = set(raw_state_dict.keys())
    
    cleaned_state_dict = {}
    
    # Check if we need to prepend 'model.' (e.g., if JIT was saved directly from the inner timm backbone)
    if not any(k.startswith("model.") for k in state_keys) and any(k.startswith("model.") for k in model_keys):
        cleaned_state_dict = {f"model.{k}": v for k, v in raw_state_dict.items()}
    else:
        # Strip common parallel wrapper prefixes like 'module.' if they exist
        for k, v in raw_state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[new_key] = v

    # 5. Load the extracted weights with strict=False to bypass structural JIT buffers if any
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device).eval()
    
    return model