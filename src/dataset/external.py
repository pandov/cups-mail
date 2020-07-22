import cv2
import json
import base64
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.datasets.folder import default_loader

def json_loader(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        labels, points = [], []
        for shape in data['shapes']:
            labels += [shape['label']]
            points += [np.array(shape['points'])]
        return labels[0], points

class Train(object):
    
    def __iter__(self):
        filepaths = Path('dataset/external/train').rglob('*.png')
        for filepath in filepaths:
            image_path = filepath.as_posix()
            image = cv2.imread(image_path)
            json_path = image_path.replace('png', 'json')

            # test_mask = np.zeros_like(image)

            # with open(json_path, 'r') as f:
            #     layout = json.load(f)

            # for shape in layout['shapes']:
            #     polygon = np.array([point[::-1] for point in shape['points']])
            #     cv2.fillPoly(test_mask, [polygon[:, [1, 0]]], color=(255, 255, 255))

            label, points = json_loader(json_path)
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, pts=points, color=(255, 255, 255))

            # print(np.array_equal(mask, test_mask))
            yield filepath.name, image, mask, label#, test_mask

def base64_to_image(encoded):
    byte = base64.b64decode(encoded)
    arr = np.frombuffer(byte, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image

class SampleSubmission(object):

    def __iter__(self):
        df = pd.read_csv('dataset/external/sample_submission.csv')
        for i, row in df.iterrows():
            idx = row['id']
            label = row['class']
            encoded = row['base64 encoded PNG (mask)']
            mask = base64_to_image(encoded)
            yield idx, label, mask

class OutputSubmission(object):

    def __init__(self, outpath):
        sources = Path('dataset/external/test').rglob('*.png')
        outputs = Path(outpath).rglob('*.png')

        sorts = lambda gen: sorted(list(gen), key=lambda f: f.stem)
        
        self.sources = sorts(sources)
        self.outputs = sorts(outputs)

    def __getitem__(self, index):
        source = self.sources[index]
        output = self.outputs[index]

        imsource = cv2.imread(source.as_posix())
        imoutput = cv2.imread(output.as_posix())
        return source.stem, imsource, imoutput

class Test(object):

    def __iter__(self):
        filepaths = Path('dataset/external/test').rglob('*.png')
        for filepath in filepaths:
            image_path = filepath.as_posix()
            image = default_loader(image_path)
            yield filepath, image
