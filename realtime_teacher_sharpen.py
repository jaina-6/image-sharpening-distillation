import cv2
import torch
import torchvision.transforms as T
import numpy as np
from model.teacher_model import TeacherSharpenNet  # 👈 Import Teacher model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load teacher model
model = TeacherSharpenNet().to(device)
model.load_state_dict(torch.load("teacher_model.pth", map_location=device))  # 👈 load trained teacher
model.eval()

# Transforms
to_tensor = T.ToTensor()
resize_down = T.Resize([60, 80])      # simulate low-res
resize_up = T.Resize([240, 320])      # upscale back

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("🎥 Press 'q' to quit (Teacher model sharpening).")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB tensor
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_img = to_tensor(rgb).unsqueeze(0).to(device)

    # Simulate network degradation
    degraded = resize_up(resize_down(tensor_img))

    # Get sharpened output
    with torch.no_grad():
        sharpened = model(degraded)

    # Convert tensors to displayable images
    inp_np = degraded.squeeze().permute(1, 2, 0).cpu().numpy()
    out_np = sharpened.squeeze().permute(1, 2, 0).cpu().numpy()

    input_bgr = cv2.cvtColor((inp_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    output_bgr = cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    combined = np.hstack((input_bgr, output_bgr))
    cv2.imshow("Left: Blurred | Right: Teacher Model Sharpened", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
