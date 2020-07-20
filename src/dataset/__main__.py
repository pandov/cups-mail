
if __name__ == '__main__':

    import cv2
    from pathlib import Path
    from .external import Train

    savepath = Path('dataset/processed')

    dataset = Train()

    items = dict()
    counts = dict(general=0)

    for name, image, mask, label in iter(dataset):

        if items.get(label) is None:
            items[label] = []

        items[label].append([name, image, mask])

        if counts.get(label) is None:
            counts[label] = 0

        counts[label] += 1
        counts['general'] += 1

    for label, data in items.items():
        count = counts[label] // 4
        for i, (name, image, mask) in enumerate(data):
            stage = 'train' if count < i else 'valid'

            path_stage = savepath.joinpath(stage)
            path_stage_sample = path_stage.joinpath('samples')
            path_stage_sample_label = path_stage_sample.joinpath(label)
            path_stage_sample_label.mkdir(exist_ok=True, parents=True)
            sample_path = path_stage_sample_label.joinpath(name).as_posix()
            cv2.imwrite(sample_path, image)

            path_stage_mask = path_stage.joinpath('masks')
            path_stage_mask_label = path_stage_mask.joinpath(label)
            path_stage_mask_label.mkdir(exist_ok=True, parents=True)
            mask_path = path_stage_mask_label.joinpath(name).as_posix()
            cv2.imwrite(mask_path, mask)

            print('Save:', name)

    general = 0
    train = 0
    valid = 0

    for key,   in counts.items():
        k = value // 4
        general += value
        train += (value - k)
        valid += k
    
    print('General:', general)
    print('Train:', train)
    print('Valid:', valid)
    print('Share:', valid / general)
