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
