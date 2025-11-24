import requests
from dotenv import load_dotenv
import os
import json

load_dotenv('.env')

client_id = os.getenv('CLIENT_KEY')
client_secret = os.getenv('SECRET_KEY')

image_path = 'images/face10.jpg'

with open(image_path, 'rb') as f:
    files = {'data': f}
    keywords = requests.post(
        'https://api.everypixel.com/v1/keywords',
        files=files,
        auth=(client_id, client_secret)
    ).json()

# Re-open file for second endpoint (stream was consumed)
with open(image_path, 'rb') as f:
    files = {'data': f}
    faces = requests.post(
        'https://api.everypixel.com/v1/faces',
        files=files,
        auth=(client_id, client_secret)
    ).json()

print(json.dumps(keywords, indent=4))
print(json.dumps(faces, indent=4))

# https://labs.everypixel.com/account/image_keywording
# https://labs.everypixel.com/account/age_recognition