from deepface import DeepFace
import json
import click

@click.command()
@click.option('--image', prompt='Image path', help='Path to the image file.')
def main(image):
  img_path = f"images/{image}.jpg"
  attributes = ['age', 'gender', 'race']
  objs = DeepFace.analyze(img_path=img_path, actions=attributes)

  print(json.dumps(objs, indent=2))

if __name__ == '__main__':
  main()  

# Run with `python deepface_recognize.py --image=face1`
# https://github.com/serengil/deepface