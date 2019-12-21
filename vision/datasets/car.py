import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat

IMAGE_EXTENSTOINS = [".png", ".jpg", ".jpeg", ".bmp"]
ATTR_ANNO = "cars_meta.mat"

def _is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext.lower() in IMAGE_EXTENSTOINS


def _find_images_and_annotation(root_dir):
    images = {}
    attr = None
    assert os.path.exists(root_dir), "{} not exists".format(root_dir)
    train_annot = loadmat(
        root_dir +'/cars_train_annos.mat'
        )['annotations'][0]
    test_annot = loadmat(
        root_dir +'/cars_test_annos_withlabels.mat'
        )['annotations'][0]

    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in sorted(fnames):
            if _is_image(fname):
                path = os.path.join(root, fname)
                index = int(os.path.splitext(fname)[0])-1
                if 'cars_test' in path:
                    label = test_annot[index][4][0][0]
                    index += 8144
                else:
                    label = train_annot[index][4][0][0]
                
                onehot = [0 if i != label-1 else 1 for i in range(196)]
                # Onehot is not used while training, it is only used
                # with infer_car.py after the training to generate z_params
                # for each class
                images[index] = {
                    "path": path,
                    "attr": onehot,
                }
            elif fname.lower() == ATTR_ANNO:
                attr = os.path.join(root, fname)

    assert attr is not None, "Failed to find `CARS CLASS NAMES FILE`"
    attrs = loadmat(attr)['class_names'][0]
    # begin to parse all image
    print("Begin to parse all image attrs")
    print("Found {} images, with {} attrs".format(len(images), len(attrs)))
    return images, attrs


class CarDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([
                                           transforms.CenterCrop(160),
                                           transforms.Resize(32),
                                           transforms.ToTensor()])):
        super().__init__()
        dicts, attrs = _find_images_and_annotation(root_dir)
        self.data = dicts
        self.attrs = attrs
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        path = data["path"]
        attr = data["attr"]
        image= Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "x": image,
            "y_onehot": np.asarray(attr, dtype=np.float32)
        }

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import cv2
    celeba = CelebADataset("/home/chaiyujin/Downloads/Dataset/CelebA")
    d = celeba[0]
    print(d["x"].size())
    img = d["x"].permute(1, 2, 0).contiguous().numpy()
    print(np.min(img), np.max(img))
    cv2.imshow("img", img)
    cv2.waitKey()
