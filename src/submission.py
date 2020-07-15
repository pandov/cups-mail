import torch
import cv2
import base64
import numpy as np
import pandas as pd
from src.nn import get_segmentation_components, get_classification_components, get_class_names
from src.dataset import Test
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sample_submission = './dataset/external/sample_submission.csv'
date = '15-07-20'
num_experiment = 0
predicted_masks = f'.tmp/{date}/tests/'
output_submission = f'.tmp/{date}/submission.csv'
segmentation_best = f'.tmp/{date}/segmentation/{num_experiment}/checkpoints/best.pth'
classification_best = f'.tmp/{date}/classification/{num_experiment}/checkpoints/best.pth'

def addata(columns, *args):
    return dict(zip(columns, args))

def negative_normalize(img):
    img = 1 - img
    img -= img.min()
    img /= img.max()
    return img

def get_test_transform():
    return transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Lambda(negative_normalize),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

if __name__ == '__main__':
    sample = pd.read_csv(sample_submission)
    columns = sample.columns
    df = pd.DataFrame(columns=columns)

    dataset = Test()
    transform = get_test_transform()

    segmentation = get_segmentation_components(1)[:2]
    segmentation_weights = torch.load(segmentation_best)
    segmentation.load_state_dict(segmentation_weights['model_state_dict'])
    segmentation.eval()
    classification = get_classification_components(1)[:2]
    classification_weights = torch.load(classification_best)
    classification.load_state_dict(classification_weights['model_state_dict'])
    classification.eval()

    for filepath, image in tqdm(list(dataset)):
        savepath = predicted_masks + filepath.name

        image = transform(image).unsqueeze(0)

        predict_segmentation = segmentation(image).detach().squeeze(0).permute(1, 2, 0).numpy()
        predict_segmentation -= predict_segmentation.min()
        predict_segmentation /= predict_segmentation.max()
        predict_segmentation[predict_segmentation < 0.4] = 0
        predict_segmentation[predict_segmentation >= 0.4] = 1
        predict_segmentation *= 255
        predict_mask = predict_segmentation.astype(np.uint8)
        cv2.imwrite(savepath, predict_mask)
        with open(savepath, 'rb') as f:
            predict_base64 = str(base64.b64encode(f.read()), 'utf-8')

        predict_classification = classification(image).detach().squeeze(0)
        predict_probs = torch.softmax(predict_classification, dim=0)
        class_names = get_class_names()
        predict_label = class_names[predict_probs.argmax()]

        df = df.append(addata(columns, filepath.stem, predict_label, predict_base64), ignore_index=True)
        df.to_csv(output_submission, index=False)
