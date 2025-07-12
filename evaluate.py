import torch
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim
from model.student_model import LightSharpenNet
from utils.cifar_dataset import get_cifar_loaders
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Student Model
model = LightSharpenNet().to(device)
model.load_state_dict(torch.load("student_kd_model.pth", map_location=device))
model.eval()

# Load Data
_, test_loader = get_cifar_loaders(batch_size=1)

def tensor_to_image(tensor):
    img = tensor.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # C x H x W → H x W x C
    img = np.clip(img, 0, 1)
    return img

ssim_scores = []

for images, _ in tqdm(test_loader, desc="Evaluating SSIM"):
    images = images.to(device)
    degraded = TF.resize(images, [16, 16])
    degraded = TF.resize(degraded, [32, 32])
    degraded = degraded.to(device)

    with torch.no_grad():
        output = model(degraded)

    gt_img = tensor_to_image(images)
    pred_img = tensor_to_image(output)

    score = ssim(gt_img, pred_img, data_range=1.0, win_size=7, channel_axis=2)
    ssim_scores.append(score)

avg_ssim = np.mean(ssim_scores)
print(f"\n✅ Average SSIM on test set: {avg_ssim * 100:.2f}%")
