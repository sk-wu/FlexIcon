import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import time
from data.ict.MMCQ import MMCQ
from data.ict.OQ import OQ
from data.ict.KMeans import KMeans


def imgPalette(imgs, themes, titles):
    N = len(imgs)
    fig = plt.figure()
    gs = gridspec.GridSpec(len(imgs), len(themes)+1)
    print(N)
    for i in range(N):
        im = fig.add_subplot(gs[i, 0])
        im.imshow(imgs[i])
        im.set_title("Image %s" % str(i+1))
        im.xaxis.set_ticks([])
        im.yaxis.set_ticks([])
        t = 1
        for themeLst in themes:
            theme = themeLst[i]
            pale = np.zeros(imgs[i].shape, dtype=np.uint8)
            h, w, _ = pale.shape
            ph  = h / len(theme)
            for y in range(h):
                pale[y,:,:] = np.array(theme[int(y / ph)], dtype=np.uint8)
            pl = fig.add_subplot(gs[i, t])
            pl.imshow(pale)
            pl.set_title(titles[t-1])
            pl.xaxis.set_ticks([])
            pl.yaxis.set_ticks([])
            t += 1
    plt.show()


def getPixData(imgfile='/home/Desktop/3.jpg'):
    return cv2.resize(cv.cvtColor(cv.imread(imgfile), cv.COLOR_BGR2RGB), (256, 256))


def testMMCQ(pixDatas, maxColor):
    # start = time.process_time()
    themes = list(map(lambda d: MMCQ(d, maxColor).quantize(), pixDatas))
    # print("MMCQ Time cost: {0}".format(time.process_time() - start))
    return themes
    # imgPalette(pixDatas, themes)


def testOQ(pixDatas, maxColor):
    start  = time.process_time()
    themes = list(map(lambda d: OQ(d, maxColor).quantize(), pixDatas))
    print("OQ Time cost: {0}".format(time.process_time() - start))
    return themes
    # imgPalette(pixDatas, themes)


def testKmeans(pixDatas, maxColor, skl=True):
    start = time.process_time()
    themes = list(map(lambda d: KMeans(d, maxColor, skl).quantize(), pixDatas))
    print("KMeans Time cost: {0}".format(time.process_time() - start))
    return themes


def vs():
    imgs = map(lambda i: 'my_imgs/photo%s.jpg' % i, range(1,4))
    pixDatas = list(map(getPixData, imgs))
    maxColor = 8
    themes = [testMMCQ(pixDatas, maxColor), testOQ(pixDatas, maxColor), testKmeans(pixDatas, maxColor)]
    imgPalette(pixDatas, themes, ["MMCQ Palette", "OQ Palette", "KMeans Palette"])


def kmvs():
    imgs = map(lambda i: 'my_imgs/photo%s.jpg' % i, range(1,4))
    pixDatas = list(map(getPixData, imgs))
    maxColor = 5
    # themes = [testKmeans(pixDatas, maxColor), testKmeans(pixDatas, maxColor, False)]
    themes = [testKmeans(pixDatas, maxColor)]
    imgPalette(pixDatas, themes, ["KMeans Palette", "KMeans DIY"])
    print(themes)
    # imgPalette(pixDatas, themes, ["KMeans Palette"])


def vs_1():
    imgs = map(lambda i: 'my_imgs/photo%s.jpg' % i, range(1,4))
    pixDatas = list(map(getPixData, imgs))
    maxColor = 5
    themes = [testMMCQ(pixDatas, maxColor)]
    imgPalette(pixDatas, themes, ["MMCQ Palette"])


def vs_2():
    imgs = map(lambda i: 'my_imgs/photo%s.jpg' % i, range(1,4))
    pixDatas = list(map(getPixData, imgs))
    maxColor = 5
    themes = [testOQ(pixDatas, maxColor)]
    imgPalette(pixDatas, themes, ["QQ Palette"])


def extract_color_theme_MMCQ(imgs_lists, maxColor = 8):
    pixDatas = list(map(getPixData, imgs_lists))
    themes = [testMMCQ(pixDatas, maxColor)]
    return themes


def getPixData_1(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def extract_color_theme_MMCQ_1(imgs_lists, maxColor = 3):
    pixDatas = [imgs_lists]
    themes = [testMMCQ(pixDatas, maxColor)]
    return np.array(themes[0])






# if __name__ == '__main__':
    # print(testMMCQ([getPixData()], 5))
    # kmvs()
    # print(testKmeans([getPixData()], 5, False))
    # kmvs()
    # vs_1()
    # vs_2()
    # print(testKmeans([getPixData()], 5))
    # vs()
