import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_mask(rgb_mask, colormap):
    output_mask = []

    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask

if __name__ == "__main__":
    """ Create a directory """
    create_dir("masks")

    """ Dataset paths """
    dataset_path = "data"

    train_images = sorted(glob(os.path.join(dataset_path, "train","images", "*.jpg")))
    train_masks = sorted(glob(os.path.join(dataset_path, "train", "masks", "*.png")))
    val_images = sorted(glob(os.path.join(dataset_path, "val", "images", "*.jpg")))
    val_masks = sorted(glob(os.path.join(dataset_path, "val", "masks", "*.png")))

    print(f"Train Images: {len(train_images)}")
    print(f"Train RGB Masks: {len(train_masks)}")
    print(f"Val Images: {len(val_images)}")
    print(f"Val RGB Masks: {len(val_masks)}")


    COLORMAP = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    CLASSES = ['stirrer', 'pan', 'food']

    """ Displaying the class name and its pixel value """
    for name, color in zip(CLASSES, COLORMAP):
        print(f"{name} - {color}")

    """ Loop over the images and masks """
    for x, y in tqdm(zip(train_images, train_masks), total=len(train_images)):
        """ Extract the name """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading the image and mask """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        mask = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Resizing the image and mask """
        image = cv2.resize(image, (320, 320))
        mask = cv2.resize(mask, (320, 320))

        """ Processing the mask to one-hot mask """
        processed_mask = process_mask(mask, COLORMAP)

        """ Converting one-hot mask to single channel mask """
        grayscale_mask = np.argmax(processed_mask, axis=-1)
        grayscale_mask = (grayscale_mask / len(CLASSES)) * 255
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)

        """ Saving the image """
        #line = np.ones((320, 5, 3)) * 255
        cat_images = np.concatenate([grayscale_mask, grayscale_mask, grayscale_mask], axis=-1)


        cv2.imwrite(f"new_data/train/masks/{name}.jpg", cat_images)

    for x, y in tqdm(zip(val_images, val_masks), total=len(val_images)):
        """ Extract the name """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading the image and mask """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        mask = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Resizing the image and mask """
        image = cv2.resize(image, (320, 320))
        mask = cv2.resize(mask, (320, 320))

        """ Processing the mask to one-hot mask """
        processed_mask = process_mask(mask, COLORMAP)

        """ Converting one-hot mask to single channel mask """
        grayscale_mask = np.argmax(processed_mask, axis=-1)
        grayscale_mask = (grayscale_mask / len(CLASSES)) * 255
        grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)

        """ Saving the image """
        #line = np.ones((320, 5, 3)) * 255
        cat_images = np.concatenate([grayscale_mask, grayscale_mask, grayscale_mask], axis=-1)

        cv2.imwrite(f"new_data/val/mask/{name}.jpg", cat_images)

