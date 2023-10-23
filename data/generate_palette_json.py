import numpy as np
import os
import cv2
from ict.KMeans import KMeans
from ict.MMCQ import MMCQ
import json


def testKeans(pixDatas, maxColor):
    # start = time.process_time()
    themes = list(map(lambda d: KMeans(d, maxColor).quantize(), pixDatas))
    # print("MMCQ Time cost: {0}".format(time.process_time() - start))
    return themes
    # imgPalette(pixDatas, themes)


def extract_color_theme_Kmeans_1(imgs_lists, maxColor = 3):
    non_white_pixels = np.array([imgs_lists[np.any(imgs_lists != [255, 255, 255], axis=2), :]])
    # print(non_white_pixels.shape)
    pixDatas = [non_white_pixels]
    # pixDatas = [imgs_lists]
    # print(pixDatas[0].shape)
    themes = [testKeans(pixDatas, maxColor)]
    return np.array(themes[0])


def testMMCQ(pixDatas, maxColor):
    # start = time.process_time()
    themes = list(map(lambda d: MMCQ(d, maxColor).quantize(), pixDatas))
    # print("MMCQ Time cost: {0}".format(time.process_time() - start))
    return themes
    # imgPalette(pixDatas, themes)


def extract_color_theme_MMCQ_1(imgs_lists, maxColor = 3):
    non_white_pixels = np.array([imgs_lists[np.any(imgs_lists != [255, 255, 255], axis=2), :]])
    # print(non_white_pixels.shape)
    pixDatas = [non_white_pixels]
    # pixDatas = [imgs_lists]
    # print(pixDatas[0].shape)
    themes = [testMMCQ(pixDatas, maxColor)]
    return np.array(themes[0])


img_dir = "../datasets/icon/train/train_B"
# img_dir = "../datasets/icon/test/test_B"

my_dict = {}

count = 0
for file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, file)
    img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # color_theme = extract_color_theme_Kmeans_1(img_array).tolist()
    color_theme = extract_color_theme_MMCQ_1(img_array).tolist()
    count += 1
    print(count, color_theme)
    my_dict[file] = color_theme

color_palette_json = "../datasets/icon/palette_icon_train.json"
# color_palette_json = "../datasets/icon/palette_icon_test.json"

with open(color_palette_json, 'w') as file_obj:
    json.dump(my_dict, file_obj)
