import cv2
import numpy as np
from skimage import io, color, filters, exposure
from skimage.morphology import skeletonize
import fingerprint_enhancer

def convert_to_grayscale_and_enchance_image(image_path):
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 3:
        gray_img = color.rgb2gray(img)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray_img = color.rgb2gray(color.rgba2rgb(img))
    elif img.ndim == 2:
        gray_img = img
    else:
        raise ValueError("Unsupported image format")

    resized_img = cv2.resize(gray_img, None, fx=5, fy=5)
    local_binary = filters.threshold_local(resized_img, block_size=55, method='gaussian')
    binarized_img = resized_img > local_binary
    out = fingerprint_enhancer.enhance_Fingerprint(binarized_img)
    out[out == 255] = 1
    skeleton = skeletonize(out)
    inverted_skeleton = np.invert(skeleton)
    return inverted_skeleton

def image_preprocesing_for_LBP(image_path):
    img = io.imread(image_path)
    if img.ndim == 3 and img.shape[2] == 3:
        gray_img = color.rgb2gray(img)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray_img = color.rgb2gray(color.rgba2rgb(img))
    elif img.ndim == 2:
        gray_img = img
    else:
        raise ValueError("Unsupported image format")

    resized_img = cv2.resize(gray_img, None, fx=5, fy=5)
    blurred_img = filters.gaussian(resized_img, sigma=1)
    equalized_img = exposure.equalize_hist(blurred_img)
    return equalized_img