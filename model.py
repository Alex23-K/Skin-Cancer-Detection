import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import requests
import os
from PIL import Image
from collections import OrderedDict


# ========================================
# EXACT MobileNetV4 from your training code
# ========================================

def make_divisible(value: float, divisor: int, min_value: float = None, round_down_protect: bool = True) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    padding = (kernel_size - 1) // 2
    layers = OrderedDict()
    layers['conv'] = nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups)
    if norm:
        layers['BatchNorm2d'] = nn.BatchNorm2d(oup)
    if act:
        layers['relu'] = nn.ReLU6(inplace=True)
    return nn.Sequential(layers)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = OrderedDict()
        if expand_ratio != 1:
            layers['exp_1x1'] = conv_2d(inp, hidden_dim, 3, stride)
        layers['red_1x1'] = conv_2d(
            hidden_dim if expand_ratio != 1 else inp,
            oup,
            kernel_size=1,
            stride=1,
            act=act
        )
        self.block = nn.Sequential(layers)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res_connect else out


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, inp, oup, start_k, mid_k, mid_down, stride, expand_ratio):
        super().__init__()
        if start_k:
            s0 = stride if not mid_down else 1
            self._start_dw_ = conv_2d(inp, inp, start_k, s0, groups=inp, act=False)
        expand_ch = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_ch, 1)
        if mid_k:
            s1 = stride if mid_down else 1
            self._middle_dw = conv_2d(expand_ch, expand_ch, mid_k, s1, groups=expand_ch)
        self._proj_conv = conv_2d(expand_ch, oup, 1, act=False)

    def forward(self, x):
        if hasattr(self, '_start_dw_'):
            x = self._start_dw_(x)
        x = self._expand_conv(x)
        if hasattr(self, '_middle_dw'):
            x = self._middle_dw(x)
        return self._proj_conv(x)


MNV4ConvMedium_BLOCK_SPECS = {
    "conv0": {"block_name": "convbn", "block_specs": [[3, 32, 3, 2]]},
    "layer1": {"block_name": "fused_ib", "block_specs": [[32, 48, 2, 4.0, True]]},
    "layer2": {"block_name": "uib", "block_specs": [[48, 80, 3, 5, True, 2, 4],
                                                    [80, 80, 3, 3, True, 1, 2]]},
    "layer3": {"block_name": "uib", "block_specs": [[80, 160, 3, 5, True, 2, 6],
                                                    [160, 160, 3, 3, True, 1, 4],
                                                    [160, 160, 3, 3, True, 1, 4],
                                                    [160, 160, 3, 5, True, 1, 4],
                                                    [160, 160, 3, 3, True, 1, 4],
                                                    [160, 160, 3, 0, True, 1, 4],
                                                    [160, 160, 0, 0, True, 1, 2],
                                                    [160, 160, 3, 0, True, 1, 4]]},
    "layer4": {"block_name": "uib", "block_specs": [[160, 256, 5, 5, True, 2, 6],
                                                    [256, 256, 5, 5, True, 1, 4],
                                                    [256, 256, 3, 5, True, 1, 4],
                                                    [256, 256, 3, 5, True, 1, 4],
                                                    [256, 256, 0, 0, True, 1, 4],
                                                    [256, 256, 3, 0, True, 1, 4],
                                                    [256, 256, 3, 5, True, 1, 2],
                                                    [256, 256, 5, 5, True, 1, 4],
                                                    [256, 256, 0, 0, True, 1, 4],
                                                    [256, 256, 0, 0, True, 1, 4],
                                                    [256, 256, 5, 0, True, 1, 2]]},
    "layer5": {"block_name": "convbn", "block_specs": [[256, 960, 1, 1],
                                                       [960, 1280, 1, 1]]}
}


def build_blocks(spec):
    bn = spec["block_name"]
    layers = []
    num_specs = len(spec["block_specs"])
    for idx, params in enumerate(spec["block_specs"]):
        # convbn-only tweak: if there are multiple convbn blocks,
        # shift the second one's index from 1 -> 2 so you get convbn_2
        if bn == "convbn" and num_specs > 1 and idx == 1:
            name = f"{bn}_{idx + 1}"  # e.g. convbn_2
        else:
            name = f"{bn}_{idx}"  # e.g. convbn_0 (and for others)

        if bn == "convbn":
            inp, oup, k, s = params
            layers.append((name, conv_2d(inp, oup, kernel_size=k, stride=s)))
        elif bn == "fused_ib":
            inp, oup, s, exp, act = params
            layers.append((name, InvertedResidual(inp, oup, s, exp, act=act)))
        elif bn == "uib":
            inp, oup, sk, mdk, mdown, s, exp = params
            layers.append((name,
                           UniversalInvertedBottleneckBlock(inp, oup, sk, mdk, mdown, s, exp)))
    return nn.Sequential(OrderedDict(layers))


class MobileNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        specs = MNV4ConvMedium_BLOCK_SPECS
        self.conv0 = build_blocks(specs["conv0"])
        self.layer1 = build_blocks(specs["layer1"])
        self.layer2 = build_blocks(specs["layer2"])
        self.layer3 = build_blocks(specs["layer3"])
        self.layer4 = build_blocks(specs["layer4"])
        self.layer5 = build_blocks(specs["layer5"])

    def forward(self, x):
        x = self.conv0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return [F.adaptive_avg_pool2d(x, 1)]


# ========================================
# EXACT FusionModel from your training code
# ========================================

class FusionModel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = False, backbone_ckpt_path: str = None):
        """
        EXACT match to your training model
        """
        super().__init__()

        # 1) Backbone (must match your training exactly)
        self.backbone = MobileNetV4()
        if pretrained and backbone_ckpt_path:
            sd = torch.load(backbone_ckpt_path, map_location='cpu')
            self.backbone.load_state_dict(sd, strict=False)

        # 2) Figure out feature dim by a dummy forward
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)[-1]
            self.backbone_dim = feat.shape[1]
            print(f"🔍 Backbone feature dimension: {self.backbone_dim}")

        # 3) Metadata MLP (2→64→32→16) - EXACT match to your training
        self.meta_net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU()
        )

        # 4) Classifier head ((backbone_dim+16)→512→256→num_classes)
        combined = self.backbone_dim + 16
        print(f"🔍 Combined feature dimension: {combined}")
        self.classifier = nn.Sequential(
            nn.Linear(combined, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # 5) Initialize any new Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor = None):
        """
        EXACT forward pass matching your training
        Returns dict with 'logits' key for compatibility
        """
        # 1) image → backbone features
        feats = self.backbone(image)[-1].view(image.size(0), -1)

        # 2) metadata → MLP or zeros if None
        if metadata is None:
            metadata = torch.zeros(image.size(0), 2, device=image.device)
        meta_feats = self.meta_net(metadata)

        # 3) fuse and classify
        x = torch.cat([feats, meta_feats], dim=1)
        logits = self.classifier(x)

        # Return format expected by app.py
        return {"logits": logits}


# ========================================
# Checkpoint inspection function
# ========================================

def inspect_model_checkpoint(model_path):
    """Detailed inspection of the saved model"""
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")

            # Check for model_state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✅ Found model_state_dict with {len(state_dict)} parameters")

                # Print training info if available
                if 'epoch' in checkpoint:
                    print(f"Saved at epoch: {checkpoint['epoch']}")
                if 'val_acc' in checkpoint:
                    print(f"Validation accuracy: {checkpoint['val_acc']}")
            else:
                state_dict = checkpoint
                print(f"✅ Direct state dict with {len(state_dict)} parameters")
        else:
            print(f"❌ Unexpected checkpoint format: {type(checkpoint)}")
            return None

        return state_dict

    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")
        return None


# ========================================
# Model loading function
# ========================================

def load_model(model_path, device='cpu'):
    """
    Load the trained fusion model
    """
    print("Inspecting model checkpoint...")
    state_dict = inspect_model_checkpoint(model_path)

    if state_dict is None:
        raise ValueError("Could not load checkpoint")

    # Create model exactly as trained
    model = FusionModel(num_classes=4)

    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            print(f"Loading from checkpoint with keys: {list(checkpoint.keys())}")
            if 'epoch' in checkpoint:
                print(f"Model was saved at epoch: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model_state_dict = checkpoint

        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in model: {len(missing_keys)} keys")

        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")

        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("🎉 PERFECT MATCH! All keys loaded successfully!")
        elif len(missing_keys) == 0:
            print("✅ All required keys loaded (some unexpected keys ignored)")
        else:
            print("⚠️ Some keys missing - model may not work correctly")

        model.eval()
        print(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


# ========================================
# Image preprocessing and prediction
# ========================================

def get_transforms():
    """Get the same transforms used during training"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict_image(model, image, metadata=None, device='cpu'):
    """
    Make prediction on a single image
    Args:
        model: Trained FusionModel
        image: PIL Image or tensor
        metadata: Optional tuple (age_norm, gender_enc)
        device: Device to run inference on
    Returns:
        dict with prediction results
    """
    transform = get_transforms()

    # Prepare image
    if isinstance(image, Image.Image):
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    else:
        image_tensor = image

    image_tensor = image_tensor.to(device)

    # Prepare metadata
    if metadata is not None:
        metadata_tensor = torch.tensor([metadata], dtype=torch.float32).to(device)
    else:
        metadata_tensor = None

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor, metadata_tensor)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1)

    # Debug information
    print(f"🔍 DEBUG - Raw logits: {logits}")
    print(f"🔍 DEBUG - Softmax probabilities: {probabilities}")
    print(f"🔍 DEBUG - Predicted class: {predicted_class.item()}")

    return {
        'predicted_class': predicted_class.cpu().item(),
        'probabilities': probabilities.cpu().numpy()[0],
        'confidence': probabilities.max().cpu().item()
    }


def debug_model_predictions(model, device='cpu'):
    """
    Debug function to test model on different types of images
    """
    print("🔍 DEBUGGING MODEL BEHAVIOR")
    print("=" * 50)

    # Create test images of different types
    test_images = {
        "Red (possible lesion)": Image.new('RGB', (224, 224), color=(200, 50, 50)),
        "Skin color": Image.new('RGB', (224, 224), color=(220, 177, 145)),
        "Green (nature)": Image.new('RGB', (224, 224), color=(50, 200, 50)),
        "Blue (non-skin)": Image.new('RGB', (224, 224), color=(50, 50, 200)),
        "White (background)": Image.new('RGB', (224, 224), color=(255, 255, 255)),
        "Black": Image.new('RGB', (224, 224), color=(0, 0, 0))
    }

    for name, test_img in test_images.items():
        print(f"\n🎨 Testing {name}:")
        try:
            result = predict_image(model, test_img, metadata=None, device=device)
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            print(f"  → Prediction: {LABELS[predicted_class]} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"  → Error: {e}")

    print("\n" + "=" * 50)


# Class labels - CHECK IF THIS MATCHES YOUR TRAINING!
LABELS = {
    0: "Melanoma Detected",
    1: "Other Skin Cancer Detected",
    2: "Benign Lesion Detected",
    3: "No Lesion Detected"
}