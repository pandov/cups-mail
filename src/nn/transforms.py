from torchvision import transforms
import torchvision.transforms.functional as TF

class Compose(transforms.Compose):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Grayscale(transforms.Grayscale):

    def __call__(self, img, mask):
        img = TF.to_grayscale(img, 3)
        mask = TF.to_grayscale(mask, 1)
        return img, mask

class ToTensor(transforms.ToTensor):

    def __call__(self, img, mask):
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        return img, mask

class Normalize(transforms.Normalize):

    def __call__(self, img, mask):
        img = TF.normalize(img, self.mean, self.std, self.inplace)
        # mask = TF.normalize(mask, self.mean, self.std, self.inplace)
        return img, mask
