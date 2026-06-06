import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN

#CLASSES 
classes = [
    "altocumulus",
    "cirrus",
    "cumulonimbus",
    "cumulus",
    "nimbostratus",
    "stratocumulus",
    "stratus"
]

#  LOAD MODEL 
model = SimpleCNN()

model.load_state_dict(
    torch.load("sky_model.pth", map_location="cpu")
)

model.eval()

# IMAGE TRANSFORM 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    confidence_value = confidence.item() * 100

    if confidence.item() < 0.60:
        return (
            "⚠ Not a sky/cloud image",
            confidence_value
        )

    return (
        classes[predicted.item()],
        confidence_value
    )