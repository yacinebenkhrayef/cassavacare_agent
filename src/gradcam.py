from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class GradCAMWrapper:
    """Thin wrapper around pytorch-grad-cam for EfficientNetCassava."""

    def __init__(self, model, device: torch.device, img_size: int = 224):
        assert GRAD_CAM_AVAILABLE, "Install grad-cam: pip install grad-cam"
        self.model = model
        self.device = device
        self.img_size = img_size
        # Hook onto last conv block (compatible with pytorch-grad-cam API)
        self.cam = GradCAM(model=model.model,
                           target_layers=[model.target_layer])

    def _preprocess(self, img_path: str):
        tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        raw = Image.open(img_path).convert("RGB").resize(
            (self.img_size, self.img_size))
        raw_np = np.array(raw) / 255.0
        tensor = tf(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        return tensor, raw_np

    def generate_heatmap(self, img_path: str, target_class: int = None
                         ) -> tuple[int, float, np.ndarray]:
        """Generate Grad-CAM overlay for img_path.

        Args:
            img_path: path to image
            target_class: if None, uses the predicted class

        Returns:
            (pred_class, confidence, overlay_rgb_uint8)
        """
        tensor, raw_np = self._preprocess(img_path)
        self.model.eval()

        # Forward pass to get prediction
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()

        target = [ClassifierOutputTarget(target_class or pred)]
        grayscale = self.cam(input_tensor=tensor, targets=target)
        overlay = show_cam_on_image(raw_np.astype(np.float32),
                                    grayscale[0], use_rgb=True)
        return pred, conf, overlay
