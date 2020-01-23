import cv2
from PIL import Image
from keras.layers import *
from keras.models import load_model
import numpy as np
from CAGAN_stackGANv2_256_covar_refiner_mixup import minibatchAB_demo, train_A, cycle_variables, showX, imageSize
from instance_normalization import InstanceNormalization
import os
import sys
import glob

fn = sys.argv[1]  # test_img_path
out_dir = sys.argv[2]  # save_dir
# data = sys.argv[3]  # data_dir
model_path = sys.argv[3]


def G(fn_generate, X):
    r = np.array([fn_generate([X[i:i + 1]]) for i in range(X.shape[0])])
    return r.swapaxes(0, 1)[:, :, 0]


# Load Data
demo_batch = minibatchAB_demo(train_A, 1, fn)
epoch, A = next(demo_batch)
_, A = demo_batch.send(1)

# Load model
model = load_model(model_path, custom_objects={
                   'InstanceNormalization': InstanceNormalization}, compile=False)
real, fake_output_list, rec_input_list, fn_generate_list, alpha_list, idt = cycle_variables(
    model)
fake_0, fake_1, fake_2 = fake_output_list
rec_0, rec_1, rec_2 = rec_input_list
cycle0_generate, cycle1_generate, cycle_generate = fn_generate_list
alpha_0, alpha_1, alpha_2 = alpha_list

# Start Prediction
rA = G(cycle_generate, A)
arr = np.concatenate([A[:, :, :, :3], A[:, :, :, 3:6],
                      A[:, :, :, 6:9], rA[0], rA[1]])
res_img = showX(arr, (imageSize, int(imageSize * 0.75)), 5)
out_path = os.path.join(out_dir, os.path.basename(
    model_path).split('.')[0] + '__' + os.path.basename(fn).split('/')[-1].split('.')[0] + '__res.png')
res_img.save(out_path)
