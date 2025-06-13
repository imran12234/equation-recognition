import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn

# Class mapping
IDX_TO_CHAR = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '='
}

# Define model with 5 classifiers
class EquationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier1 = nn.Linear(64 * 7 * 7, 15)
        self.classifier2 = nn.Linear(64 * 7 * 7, 15)
        self.classifier3 = nn.Linear(64 * 7 * 7, 15)
        self.classifier4 = nn.Linear(64 * 7 * 7, 15)
        self.classifier5 = nn.Linear(64 * 7 * 7, 15)

    def forward(self, x):
        x = self.features(x)
        return [
            self.classifier1(x),
            self.classifier2(x),
            self.classifier3(x),
            self.classifier4(x),
            self.classifier5(x)
        ]

# Load model
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EquationModel().to(DEVICE)
model.load_state_dict(torch.load("model/equation_model.pt", map_location=DEVICE))
model.eval()

# Transform for input
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def predict_from_frame(frame):
    # Split ROI into 5 character segments
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    char_w = frame.shape[1] // 5
    predictions = []

    for i in range(5):
        char = frame[:, i * char_w:(i + 1) * char_w]
        pil_img = Image.fromarray(char)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outs = model(input_tensor)
            idx = outs[i].argmax(dim=1).item()
            predictions.append(IDX_TO_CHAR.get(idx, '?'))

    return ''.join(predictions)

# Live cam loop
cap = cv2.VideoCapture(0)
print("📝 Hold handwritten equation in the blue box. Press [q] to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    box_w, box_h = int(w * 0.9), int(h * 0.5)
    x = (w - box_w) // 2
    y = (h - box_h) // 2
    roi = frame[y:y + box_h, x:x + box_w]

    try:
        pred = predict_from_frame(roi)
    except Exception:
        pred = "..."

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)
    cv2.putText(frame, f"Predicted: {pred}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Live Equation Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
