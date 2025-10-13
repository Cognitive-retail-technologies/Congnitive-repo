# predict.py
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image
import argparse

# -----------------------
# Config
# -----------------------
MODEL_PATH = "model.pth"
CONF_THRESHOLD = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Argument parser for image path
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to image to predict")
args = parser.parse_args()
IMAGE_PATH = args.image

# ---------------------------------
# Data transform (same as training)
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# Load dataset to get class names
# -----------------------
train_ds = datasets.ImageFolder("datasets/train", transform=transform)
classes = train_ds.classes

# -----------------------
# Load model
# -----------------------
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------
# Prediction function
# -----------------------
def predict(image_path, threshold=CONF_THRESHOLD):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        max_prob, pred = torch.max(probs, 1)
        if max_prob.item() < threshold:
            return "Unknown Document"
        else:
            return classes[pred.item()]

# -----------------------
# Make prediction 
# -----------------------
result = predict(IMAGE_PATH)
print(f"Prediction: {result}")

