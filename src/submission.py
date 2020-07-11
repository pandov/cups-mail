import torch
import cv2
import base64
import pandas as pd
from src.nn import get_segmentation_model, get_classification_model, get_class_names
from src.dataset import Test
from PIL import Image
from torchvision import transforms

sample_submission = './dataset/external/sample_submission.csv'
output_submission = './dataset/processed/submission.csv'
predicted_masks = './dataset/processed/tests/'
segmentation_best = './logs/segmentation/checkpoints/best.pth'
classification_best = './logs/classification/checkpoints/best.pth'

def adata(columns, *args):
    return dict(zip(columns, args))

if __name__ == '__main__':
    sample = pd.read_csv(sample_submission)
    columns = sample.columns
    df = pd.DataFrame(columns=columns)

    dataset = Test()
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    class_names = get_class_names()

    segmentation = get_segmentation_model()
    segmentation_weights = torch.load(segmentation_best)
    segmentation.load_state_dict(segmentation_weights['model_state_dict'])
    segmentation.eval()
    classification = get_classification_model(len(class_names))
    classification_weights = torch.load(classification_best)
    classification.load_state_dict(classification_weights['model_state_dict'])
    classification.eval()

    for filepath, image in dataset:
        savepath = predicted_masks + filepath.name

        image = transform(image).unsqueeze(0)

        predict_segmentation = segmentation(image).detach().squeeze(0).permute(1, 2, 0).numpy()
        cv2.imwrite(savepath, predict_segmentation)
        with open(savepath, 'rb') as f:
            predict_base64 = str(base64.b64encode(f.read()), 'utf-8')

        predict_classification = classification(image).detach().squeeze(0)
        predict_probs = torch.softmax(predict_classification, dim=0)
        predict_label = class_names[predict_probs.argmax()]
        df = df.append(adata(columns, filepath.stem, predict_label, predict_base64), ignore_index=True)
        break

    df.to_csv(output_submission, index=False)
