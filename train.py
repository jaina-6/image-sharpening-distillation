import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms.functional as TF

from model.student_model import LightSharpenNet
from model.teacher_model import TeacherSharpenNet
from utils.cifar_dataset import get_cifar_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_loader, _ = get_cifar_loaders(batch_size=64)

# Models
teacher = TeacherSharpenNet().to(device)
student = LightSharpenNet().to(device)

# Freeze teacher


# Optimizer
optimizer = optim.Adam(student.parameters(), lr=1e-3)
epochs = 10

# Optional: train teacher first (or load pre-trained teacher)
print("📢 Training Teacher model on CIFAR (optional)...")

teacher_optimizer = optim.Adam(teacher.parameters(), lr=1e-3)

for epoch in range(2):
    teacher.train()
    total_loss = 0
    for images, _ in tqdm(train_loader, desc=f"[Teacher] Epoch {epoch+1}/2"):
        images = images.to(device)
        degraded = TF.resize(images, [16, 16])
        degraded = TF.resize(degraded, [32, 32])
        degraded = degraded.to(device)

        output = teacher(degraded)
        loss = F.mse_loss(output, images)

        teacher_optimizer.zero_grad()
        loss.backward()
        teacher_optimizer.step()

        total_loss += loss.item()

    print(f"📘 Teacher Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
print("✅ Teacher trained briefly.")

for param in teacher.parameters():
    param.requires_grad = False
# Main Training (Student mimics teacher)
alpha = 0.9  # weight for mimicking teacher
beta = 0.1   # weight for true GT loss

for epoch in range(epochs):
    student.train()
    total_loss = 0

    for images, _ in tqdm(train_loader, desc=f"[Student] Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        degraded = TF.resize(images, [16, 16])
        degraded = TF.resize(degraded, [32, 32])
        degraded = degraded.to(device)

        with torch.no_grad():
            teacher_out = teacher(degraded)

        student_out = student(degraded)

        loss_kd = F.mse_loss(student_out, teacher_out)
        loss_gt = F.mse_loss(student_out, images)
        loss = alpha * loss_kd + beta * loss_gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Distillation Loss: {total_loss / len(train_loader):.4f}")

torch.save(student.state_dict(), "student_kd_model.pth")
print("✅ Student model trained with distillation and saved as student_kd_model.pth")

torch.save(teacher.state_dict(), "teacher_model.pth")
print("✅ Teacher model saved.")
