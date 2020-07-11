# %%

def _test_dataset_external(dataset):
    import cv2
    size = lambda x: 'x'.join(map(str, x.shape))
    for i, (filename, image, mask, label) in enumerate(iter(dataset)):
        if i == 2: break
        cv2.imshow(f'samples/{label}/{filename}-{size(image)}', image)
        cv2.imshow(f'masks/{label}/{filename}-{size(image)}', mask)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_dataset_external_test():
    from src.dataset import Test
    _test_dataset_external(Test())

def test_dataset_external_train():
    from src.dataset import Train
    _test_dataset_external(Train())

def _torch2cv(tensor, dtype):
    tensor = tensor.permute(1, 2, 0)
    tensor -= tensor.min()
    tensor *= 255 / tensor.max()
    image = tensor.numpy().astype(dtype)
    return image

def test_dataset():
    import cv2
    import numpy as np
    from src.nn import BACTERIA
    dataset = BACTERIA()
    datasets = next(dataset.crossval(2))
    size = lambda x: 'x'.join(map(str, x.shape))
    for i, (filename, image, mask, label) in enumerate(datasets['train']):
        if i == 5: break
        image = _torch2cv(image, np.uint8)
        mask = _torch2cv(mask, np.uint8)
        cv2.imshow(f'sample-{filename}-{size(image)}', image)
        cv2.imshow(f'mask-{filename}-{size(mask)}', mask)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

test_dataset()

# %%