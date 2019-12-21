"""Train script.

Usage:
    infer_car.py <hparams> <dataset_root> <z_dir>
"""
import os
import cv2
import random
import torch
import vision
import numpy as np
from docopt import docopt
from torchvision import transforms
from glow.builder import build
from glow.config import JsonConfig
import imageio
from tqdm import tqdm
from datetime import datetime

def select_index(name, l, r, description=None):
    index = None
    while index is None:
        print("Select {} with index [{}, {}),"
              "or {} for random selection".format(name, l, r, l - 1))
        if description is not None:
            for i, d in enumerate(description):
                print("{}: {}".format(i, d))
        try:
            line = int(input().strip())
            if l - 1 <= line < r:
                index = line
                if index == l - 1:
                    index = random.randint(l, r - 1)
        except Exception:
            pass
    return index


def run_z(graph, z):
    graph.eval()
    x = graph(z=torch.tensor([z]).cuda(), eps_std=0.3, reverse=True)
    img = x[0].permute(1, 2, 0).detach().cpu().numpy()
    img = img[:, :, ::-1]
    img = cv2.resize(img, (256, 256))
    return img

def save_image(img, name):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite("pictures/infer/{}.png".format(name), img)
    cv2.imshow("img", img)
    while 1:
        if cv2.waitKey(33):
            break

def save_images(images, names):
    if not os.path.exists("pictures/infer/"):
        os.makedirs("pictures/infer/")
    for img, name in zip(images, names):
        save_image(img, name)

def interact():
    while 1:
        print('What do you want?')
        print()
        print('1: To interpolate between two images press')
        print('2: To convert a car image to another car model press')
        choice = int(input())
        if choice == 1:
            base_index = select_index("base image", 0, len(dataset))
            sec_index = select_index("base image", 0, len(dataset))
            # Encode and decode the original image to see that
            # The model mimics the original image that is
            # The generator fools the discriminator    
            z_base = graph.generate_z(dataset[base_index]["x"])
            save_image(run_z(graph, z_base), 'encoded_' + str(base_index) + str(datetime.now()))

            z_base = graph.generate_z(dataset[base_index]["x"])
            z_delta = graph.generate_z(dataset[sec_index]["x"])
            return base_index, z_base, sec_index, z_delta
        elif choice == 2:
            base_index = select_index("base image", 0, len(dataset))
            attr_index = select_index("attritube", 0, len(delta_Z), dataset.attrs)
            z_base = graph.generate_z(dataset[base_index]["x"])
            z_delta = delta_Z[attr_index]
            return base_index, z_base, attr_index, z_delta


if __name__ == "__main__":
    args = docopt(__doc__)
    hparams = args["<hparams>"]
    dataset_root = args["<dataset_root>"]
    z_dir = args["<z_dir>"]
    assert os.path.exists(dataset_root), (
        "Failed to find root dir `{}` of dataset.".format(dataset_root))
    assert os.path.exists(hparams), (
        "Failed to find hparams josn `{}`".format(hparams))
    if not os.path.exists(z_dir):
        print("Generate Z to {}".format(z_dir))
        os.makedirs(z_dir)
        generate_z = True
    else:
        print("Load Z from {}".format(z_dir))
        generate_z = False

    hparams = JsonConfig("hparams/car.json")
    dataset = vision.Datasets["car"]
    # set transform of dataset
    transform = transforms.Compose([
        #transforms.CenterCrop(hparams.Data.center_crop),
        transforms.Resize((hparams.Data.resize, hparams.Data.resize)),
        transforms.ToTensor()])
    # build
    graph = build(hparams, False)["graph"]
    dataset = dataset(dataset_root, transform=transform)

    # get Z
    if not generate_z:
        # try to load
        try:
            delta_Z = []
            for i in range(hparams.Glow.y_classes):
                z = np.load(os.path.join(z_dir, "detla_z_{}.npy".format(i)))
                delta_Z.append(z)
        except FileNotFoundError:
            # need to generate
            generate_z = True
            print("Failed to load {} Z".format(hparams.Glow.y_classes))
            quit()
    if generate_z:
        delta_Z = graph.generate_attr_deltaz(dataset)
        for i, z in enumerate(delta_Z):
            np.save(os.path.join(z_dir, "detla_z_{}.npy".format(i)), z)
        print("Finish generating")

    # interact with user
    base_index, z_base, attr_index, z_delta = interact()

    # begin to generate new image
    graph.eval()
    images = []
    names = []
    images.append(run_z(graph, z_base))
    names.append("reconstruct_origin")
    interplate_n = 100

    for i in tqdm(range(0, interplate_n+1)):
        d = z_delta * float(i) / float(interplate_n)
        img = (np.clip(run_z(graph, z_base + d), 0, 1) * 255).astype(np.uint8)
        images.append(img)
        names.append("attr_{}_{}".format('RAMBO', interplate_n + i))

    imageio.mimsave(
        'pictures/infer/first_' 
        + str(base_index) 
        + '_second_' 
        + str(attr_index) 
        + '_date: '
        + str(datetime.now())
        + '.gif',
        images
    )
    #save_images(images, names)
