import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np

# Load ONNX model
session = ort.InferenceSession("AgeRaceGenderNet_v1.onnx")

# Get input and output names
input_name = session.get_inputs()[0].name
output_names = [out.name for out in session.get_outputs()]

# Preprocessing: Load and transform image
# NOTE: The input image is assumed to be cropped and aligned to contain only the face

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open("images/face1.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).numpy()  # Shape: (1, 3, 256, 256)

# Run inference
age_logits, gender_logits, race_logits = session.run(output_names, {input_name: img_tensor})

# Postprocessing: Get predictions
age_pred = int(np.argmax(age_logits, axis=1)[0])
gender_pred = int(np.argmax(gender_logits, axis=1)[0])
race_pred = int(np.argmax(race_logits, axis=1)[0])

# Convert predictions to labels
def get_gender_text(gender_idx):
    return 'Male' if gender_idx == 0 else 'Female'

def get_race_text(race_idx):
    race_map = {
        0: 'White',
        1: 'Black',
        2: 'Asian',
        3: 'Indian',
        4: 'Other'
    }
    return race_map.get(race_idx, 'Unknown')

# Display results
print(f"Predicted Age: {age_pred}")
print(f"Predicted Gender: {get_gender_text(gender_pred)}")
print(f"Predicted Race: {get_race_text(race_pred)}")

# https://huggingface.co/atheless/AgeRaceGenderNet