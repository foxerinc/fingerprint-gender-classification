import numpy as np
from localbinarypatterns import LocalBinaryPatterns
from ridgedensity import RidgeDensityCalculator
import os

def store_ridge_density_features(arr_file):
    array_ridge_density = []
    for i in range(len(arr_file)):
        convert_to_int = (1 - arr_file[i]).astype(np.uint8) * 255
        area = np.prod(arr_file[i].shape)
        ridge_count = sum(255 for row in convert_to_int for pixel in row if pixel == 255)
        ridge_density = RidgeDensityCalculator(ridge_count, area)
        count_ridge_density = ridge_density.calculate_ridge_density()
        array_ridge_density.append(count_ridge_density)
    return array_ridge_density

def store_LBP_features(arr_file):
    array_lbp = []
    desc = LocalBinaryPatterns(8, 1)
    for i in range(len(arr_file)):
        histogram = desc.describe(arr_file[i])
        array_lbp.append(histogram)
    return array_lbp

def store_hand(arr_file):
    temp_array = []
    for i in arr_file:
        path = os.path.basename(i)
        file = path.split("_")
        array =[]
        location = file[4]
        
        #array.append(file[0])

        if location == "thumb":
            array.append(0)
            if file[3] == "Left":
                array.append(0)
            elif file[3] == "Right":
                array.append(1)
        elif location == "middle":
            array.append(1)
            if file[3] == "Left":
                array.append(0)
            elif file[3] == "Right":
                array.append(1)
        elif location == "ring":
            array.append(2)
            if file[3] == "Left":
                array.append(0)
            elif file[3] == "Right":
                array.append(1)
        elif location == "index":
            array.append(3)
            if file[3] == "Left":
                array.append(0)
            elif file[3] == "Right":
                array.append(1)
        else:
            array.append(4)
            if file[3] == "Left":
                array.append(0)
            elif file[3] == "Right":
                array.append(1)
        temp_array.append(array)
    return temp_array
