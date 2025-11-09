"""
MINIMAL MINIMAGEN TRAINING SCRIPT
Run: python simple_train.py
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

from minimagen.Imagen import Imagen
from minimagen.Unet import Unet, BaseTest, SuperTest
from minimagen.training import get_default_args

# ============= CONFIGURATION =============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE_BASE = 64
IMAGE_SIZE_FINAL = 128
TIMESTEPS = 200
DATA_DIR = "./tiny_dataset"
CHECKPOINT_PATH = "./minimagen_checkpoint.pth"

print(f"Device: {DEVICE}")
print(f"Using image sizes: {IMAGE_SIZE_BASE} → {IMAGE_SIZE_FINAL}")

# ============= DATASET CLASS =============
class SimpleImageCaptionDataset(Dataset):
    def __init__(self, image_dir, image_size=64):
        self.image_size = image_size
        self.image_files = [
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        print(f"Found {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        img_tensor = (img_tensor * 2) - 1  # Normalize to [-1, 1]
        
        caption = "a photo"
        return img_tensor, caption

# ============= CREATE DATASET & DATALOADER =============
print("Creating dataset...")
dataset = SimpleImageCaptionDataset(DATA_DIR, image_size=IMAGE_SIZE_FINAL)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"Created dataloader with {len(dataloader)} batches")

# ============= CREATE MODEL =============
print("Creating models...")
unet_base = Unet(**get_default_args(BaseTest)).to(DEVICE)
unet_super = Unet(**get_default_args(SuperTest)).to(DEVICE)
unets = [unet_base, unet_super]

imagen = Imagen(
    unets=unets,
    image_sizes=(IMAGE_SIZE_BASE, IMAGE_SIZE_FINAL),
    timesteps=TIMESTEPS,
    cond_drop_prob=0.15,
    text_encoder_name='t5_small'
).to(DEVICE)

print("Model created!")
print(f"Total parameters: {sum(p.numel() for p in imagen.parameters()):,}")

# ============= OPTIMIZER =============
optimizer = optim.Adam(imagen.parameters(), lr=LEARNING_RATE)

# ============= TRAINING LOOP =============
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

try:
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(DEVICE)
            
            # Forward pass - train base model
            loss = imagen(images, captions, unet_number=1)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(imagen.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"✓ Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}\n")
    
    # Save model
    torch.save({
        'imagen_state': imagen.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': {
            'image_sizes': (IMAGE_SIZE_BASE, IMAGE_SIZE_FINAL),
            'timesteps': TIMESTEPS,
        }
    }, CHECKPOINT_PATH)
    
    print(f"✓ Model saved to {CHECKPOINT_PATH}")
    print("Training complete! Now run: python simple_inference.py")
    
except KeyboardInterrupt:
    print("\n⚠ Training interrupted")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()