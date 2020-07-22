# %%

def _test_dataset_external(dataset):
    import cv2
    size = lambda x: 'x'.join(map(str, x.shape))
    for i, (filename, image, mask, label, test_mask) in enumerate(iter(dataset)):
        # if i == 2: break
        cv2.imshow(f'samples/{label}/{filename}-{size(image)}', image)
        cv2.imshow(f'masks/{label}/{filename}-{size(mask)}', mask)
        # cv2.imshow(f'test_mask/{label}/{filename}-{size(test_mask)}', test_mask)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_dataset_external_test():
    from src.dataset import Test
    _test_dataset_external(Test())

def test_dataset_external_train():
    from src.dataset import Train
    _test_dataset_external(Train())

def test_dataset():
    import cv2
    import numpy as np
    from src.nn import BACTERIA

    def _torch2cv(tensor, dtype=np.uint8):
        tensor = tensor.permute(1, 2, 0)
        tensor -= tensor.min()
        tensor *= 255 / tensor.max()
        image = tensor.numpy().astype(dtype)
        return image

    dataset = BACTERIA('train', keys=['name', 'image', 'mask', 'label'], apply_mask=True, is_resized=True)
    size = lambda x: 'x'.join(map(str, x.shape))
    for i, (name, image, mask, label) in enumerate(dataset):
        # if i == 10: break
        image = _torch2cv(image)
        mask = _torch2cv(mask)
        cv2.imshow(f'sample-{name}-{label}-{size(image)}', image)
        cv2.imshow(f'mask-{size(mask)}', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_output_submission():
    import cv2
    from src.dataset.external import OutputSubmission
    out = OutputSubmission('.tmp/22-07-20/tests')
    for name, source, output in out:
        source[output == 0] /= 2
        cv2.imshow(f'{name}-source', source)
        cv2.imshow(f'{name}-output', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

test_dataset()

   # %%
