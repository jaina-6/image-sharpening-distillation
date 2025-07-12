# 📸 Real-Time Image Sharpening Using Knowledge Distillation

A lightweight image sharpening deep learning model for real-time webcam/video enhancement. Trained using Knowledge Distillation with a high-performing teacher and a fast student network.

---

## 🚀 Project Features

- Real-time image sharpening (30–60 FPS)
- Knowledge distillation (Teacher ➝ Student)
- Webcam demo using OpenCV
- SSIM evaluation on CIFAR-10 test set

---

## 🧠 Model Architecture

### 🔷 Teacher Model
- 4-layer CNN
- Trained on degraded vs. high-res CIFAR-10 images
- High accuracy but slower inference

### 🔶 Student Model
- 2-layer lightweight CNN
- Trained to mimic teacher + predict ground truth
- Real-time performance (30–60 FPS)

### 🔁 Distillation Loss

```python
Loss = α * MSE(Student, Teacher) + β * MSE(Student, GroundTruth)
