import cv2
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path

def json_loader(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        labels, points = [], []
        for shape in data['shapes']:
            labels += [shape['label']]
            points += [np.array(shape['points'])]
        return labels[0], points

def image_loader(image_path):
    return cv2.imread(image_path)

class Train(object):
    
    def __iter__(self):
        filepaths = Path('dataset/external/train').rglob('*.png')
        for filepath in filepaths:
            image_path = filepath.as_posix()
            image = image_loader(image_path)
            json_path = image_path.replace('png', 'json')
            label, points = json_loader(json_path)
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, pts=points, color=(255, 255, 255))
            yield filepath.name, image, mask, label

def base64_to_image(encoded):
    byte = base64.b64decode(encoded)
    arr = np.frombuffer(byte, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image

class Test(object):

    def __iter__(self):
        filepaths = Path('dataset/external/test').rglob('*.png')
        df = pd.read_csv('dataset/external/sample_submission.csv', index_col='id')
        for filepath in filepaths:
            image_path = filepath.as_posix()
            image = image_loader(image_path)
            index = int(filepath.stem)
            data = df.iloc[index]
            label = data['class']
            encoded = data['base64 encoded PNG (mask)']
            mask = base64_to_image(encoded)
            yield filepath.name, image, mask, label
