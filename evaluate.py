# evaluate.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from sinet import SINet
from dataset import SalientObjectDataset
import os
from PIL import Image

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Datasets and DataLoaders
test_dataset = SalientObjectDataset(
    image_root='path_to_test_images',
    mask_root='path_to_test_masks',  # Optional, if you have ground truth
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Model
model = SINet()
model.load_state_dict(torch.load('checkpoints/sinet_epoch_30.pth', map_location='cpu'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Create output directory
output_dir = 'output_masks'
os.makedirs(output_dir, exist_ok=True)

# Evaluation Loop
with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        images = batch['image'].to(device)

        # Forward pass
        outputs = model(images)
        outputs = outputs.squeeze(1)  # [B, H, W]

        # Resize to original size if necessary
        output = outputs.cpu()

        # Save predicted masks
        save_path = os.path.join(output_dir, f'mask_{idx + 1}.png')
        save_image(output, save_path)

        print(f'Saved mask {idx + 1}')

print('Evaluation completed. Masks saved in output_masks directory.')
