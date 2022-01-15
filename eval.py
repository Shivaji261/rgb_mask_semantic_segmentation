import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import r2_score, mean_squared_error, max_error
from metrics import dice_loss, dice_coef, iou

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (512, 512)
    # x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 2, 3)) * 255

    # ori_y = np.expand_dims(ori_y, axis=-1)
    # ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    # y_pred = np.expand_dims(y_pred, axis=-1)
    # y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Save the results in this folder """
    create_dir("val_results")

    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    dataset_path = os.path.join("new_data", "val")
    test_x, test_y = load_data(dataset_path)

    """ Make the prediction and calculate the metrics values """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        """ Extracting name """
        name = x.split("\\")[-1].split(".")[0]

        """ Read the image and mask """
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        """ Prediction """
        y_pred = model.predict(np.expand_dims(x, axis=0))[0]
        y_pred = y_pred * 255

        """ Saving the images """
        save_image_path = f"val_results/{name}.png"
        save_results(ori_x, ori_y, y_pred, save_image_path)

        """ Flatten the array """
        y = ori_y.ravel()
        y_pred = y_pred.ravel()

        """ Calculate the metrics """
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        max_err = max_error(y, y_pred)
        SCORE.append([name, r2, mse, max_err])

    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"r2_score: {score[0]:0.5f}")
    print(f"mse: {score[1]:0.5f}")
    print(f"max_error: {score[2]:0.5f}")

    """ Saving """
    df = pd.DataFrame(SCORE, columns=["Image", "R2", "MSE", "Max Error"])
    df.to_csv("files/score.csv")