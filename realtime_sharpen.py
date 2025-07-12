import cv2
import torch
import torchvision.transforms as T
import numpy as np
from model.student_model import LightSharpenNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load student model
model = LightSharpenNet().to(device)
model.load_state_dict(torch.load("student_kd_model.pth", map_location=device))
model.eval()

# Transformations
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("📸 Press 'q' to quit the live sharpening window.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB and normalize
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = to_tensor(img).unsqueeze(0).to(device)

    # Simulate poor quality (downscale & upscale)
    down = T.Resize([60, 80])(pil_img)
    degraded = T.Resize([240, 320])(down)

    # Model prediction
    with torch.no_grad():
        sharpened = model(degraded)

    # Convert to displayable images
    input_np = degraded.squeeze().permute(1, 2, 0).cpu().numpy()
    output_np = sharpened.squeeze().permute(1, 2, 0).cpu().numpy()

    # Convert to BGR format for OpenCV
    input_bgr = cv2.cvtColor((input_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    output_bgr = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    combined = np.hstack((input_bgr, output_bgr))
    cv2.imshow("Left: Blurred | Right: Sharpened by Student Model", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
