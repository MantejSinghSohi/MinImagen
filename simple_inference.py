"""
MINIMAL MINIMAGEN INFERENCE SCRIPT
Run: python simple_inference.py
"""

import torch
import os

from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, BaseTest, SuperTest
from minimagen.training import get_default_args
from torchvision.transforms import ToPILImage

# ============= CONFIGURATION =============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "./minimagen_checkpoint.pth"
IMAGE_SIZE_BASE = 32
IMAGE_SIZE_FINAL = 64
TIMESTEPS = 200

print(f"Device: {DEVICE}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint not found. Run simple_train.py first!")
    exit(1)

# ============= LOAD MODEL =============
print(f"Loading checkpoint...")

unet_base = Unet(**get_default_args(BaseTest)).to(DEVICE)
unet_super = Unet(**get_default_args(SuperTest)).to(DEVICE)
unets = [unet_base, unet_super]

imagen = Imagen(
    unets=unets,
    image_sizes=(IMAGE_SIZE_BASE, IMAGE_SIZE_FINAL),
    timesteps=TIMESTEPS,
    text_encoder_name='t5_small'
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
imagen.load_state_dict(checkpoint['imagen_state'])
imagen.eval()

print("✓ Model loaded!")

# ============= GENERATE IMAGES =============
print("\n" + "="*80)
print("GENERATING IMAGES")
print("="*80 + "\n")

captions = [
    'a happy dog',
    'a red house',
    'a blue car',
]

print(f"Generating {len(captions)} images...\n")

try:
    with torch.no_grad():
        images = imagen.sample(
            texts=captions,
            cond_scale=3.0,
        )
    
    # Save images
    output_dir = "./generated_images"
    os.makedirs(output_dir, exist_ok=True)

    for i, (img, caption) in enumerate(zip(images, captions)):
        filename = f"{output_dir}/generated_{i:02d}_{caption.replace(' ', '_')}.png"
    
        # Convert tensor to PIL Image
        if isinstance(img, torch.Tensor):
            img = ToPILImage()(img)
    
        img.save(filename)
    
    print(f"\n✓ Done! Images in {output_dir}/")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
