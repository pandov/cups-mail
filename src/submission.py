import torch
import cv2
import base64
import numpy as np
import pandas as pd
from src.nn import get_segmentation_components, get_classification_components, get_multimodel_components, get_class_names
from src.dataset import Test
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from catalyst.utils import get_device

sample_submission = './dataset/external/sample_submission.csv'
date = '22-07-20'
num_experiment = 0
predicted_masks = f'.tmp/{date}/tests/'
output_submission = f'.tmp/{date}/submission.csv'
# segmentation_best = f'.tmp/{date}/segmentation/{num_experiment}/checkpoints/best.pth'
# classification_best = f'.tmp/{date}/classification/{num_experiment}/checkpoints/best.pth'
segmentation_best = f'.tmp/{date}/segmentation/unet_timm-efficientnet-b4/checkpoints/best.pth'
classification_best = f'.tmp/{date}/classification/timm-efficientnet-b4/checkpoints/best.pth'
# multimodel_best = f'.tmp/{date}/multimodel/{num_experiment}/checkpoints/best.pth'

def addata(columns, *args):
    return dict(zip(columns, args))

def negative_normalize(img):
    img = 1 - img
    img -= img.min()
    img /= img.max()
    return img

def get_test_transform():
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Lambda(negative_normalize),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_test_transform_clf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Lambda(negative_normalize),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

if __name__ == '__main__':
    sample = pd.read_csv(sample_submission)
    columns = sample.columns
    df = pd.DataFrame(columns=columns)

    dataset = Test()
    transform = get_test_transform()
    transform_clf = get_test_transform_clf()
    device = get_device()

    # multimodel = get_multimodel_components('resnet50')['model'].to(device)
    # multimodel_weights = torch.load(multimodel_best)
    # multimodel.load_state_dict(multimodel_weights['model_state_dict'])
    # multimodel.eval()

    segmentation = get_segmentation_components('unet', 'timm-efficientnet-b4')['model'].to(device)
    segmentation_weights = torch.load(segmentation_best)
    segmentation.load_state_dict(segmentation_weights['model_state_dict'])
    segmentation.eval()

    classification = get_classification_components('efficientnet-b4')['model'].to(device)
    classification_weights = torch.load(classification_best)
    classification.load_state_dict(classification_weights['model_state_dict'])
    classification.eval()

    # predict_mask = np.zeros((512, 640, 3))

    unpack = lambda t: t.detach().cpu().squeeze(0)

    for filepath, image in tqdm(list(dataset)):
        savepath = predicted_masks + filepath.name

        image_clf = transform_clf(image).unsqueeze(0).to(device)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # predict = multimodel(image)
            predict_classification, predict_segmentation = map(unpack, (classification(image_clf), segmentation(image)))
            predict_segmentation = predict_segmentation.permute(1, 2, 0).numpy()

        # predict_segmentation, predict_classification = map(unpack, predict)
        # predict_segmentation = predict_segmentation.permute(1, 2, 0).numpy()

        predict_segmentation -= predict_segmentation.min()
        predict_segmentation /= predict_segmentation.max()
        predict_mask = (predict_segmentation > 0.5).astype(np.uint8)
        predict_mask *= 255

        cv2.imwrite(savepath, predict_mask)
        with open(savepath, 'rb') as f:
            predict_base64 = str(base64.b64encode(f.read()), 'utf-8')

        predict_probs = torch.softmax(predict_classification, dim=0)
        class_names = get_class_names()
        predict_label = class_names[predict_probs.argmax()]

        df = df.append(addata(columns, filepath.stem, predict_label, predict_base64), ignore_index=True)
        df.to_csv(output_submission, index=False)

        torch.cuda.empty_cache()
