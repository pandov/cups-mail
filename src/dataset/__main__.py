
if __name__ == '__main__':

    import cv2
    from pathlib import Path
    from .external import Train

    savepath = Path('dataset/processed')
    samples = savepath.joinpath('samples')
    masks = savepath.joinpath('masks')

    dataset = Train()

    for filename, image, mask, label in iter(dataset):

        samples_label = samples.joinpath(label)
        samples_label.mkdir(exist_ok=True, parents=True)
        sample_filename = samples_label.joinpath(filename).as_posix()
        cv2.imwrite(sample_filename, image)

        masks_label = masks.joinpath(label)
        masks_label.mkdir(exist_ok=True, parents=True)
        mask_filename = masks_label.joinpath(filename).as_posix()
        cv2.imwrite(mask_filename, mask)

        print('Saved:', filename)
