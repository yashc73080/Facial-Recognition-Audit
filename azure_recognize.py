from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import os
from dotenv import load_dotenv

load_dotenv(".env")

ENDPOINT = os.getenv("VISION_ENDPOINT")
KEY = os.getenv("VISION_KEY")

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

def analyze_face(image_path):
    with open(image_path, "rb") as f:
        detected_faces = face_client.face.detect_with_stream(
            image=f,
            return_face_attributes=["age", "gender"],
            detection_model="detection_03",
            recognition_model="recognition_04"
        )
    return detected_faces

faces = analyze_face("images/face1.jpg")

for face in faces:
    print("Face ID:", face.face_id)
    print("Age:", face.face_attributes.age)
    print("Gender:", face.face_attributes.gender)
