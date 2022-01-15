import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x


def load_data(path):
    x = sorted(glob(os.path.join(path, "images", "*.png")))
    return x

def save_results(ori_x, y_pred, save_image_path):
    line = np.ones((H, 2, 3)) * 255
    concat_img = np.concatenate([ori_x, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, concat_img)

if __name__ == "__main__":
    """ Save the results in this folder """
    create_dir("test_results")

    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    dataset_path = os.path.join("new_data", "test")
    test_x = load_data(dataset_path)

    for x in tqdm(test_x, total=len(test_x)):

        """ Extracting name """
        name = x.split("\\")[-1].split(".")[0]
        ori_x, x = read_image(x)

        """ Prediction """
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = y_pred * 255

        """ Saving the images """
        save_image_path = f"test_results/{name}.png"
        save_results(ori_x, y_pred, save_image_path)

