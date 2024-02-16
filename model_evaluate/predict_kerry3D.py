# predict kerry3d timeslices
from utils.common_tools import *
from utils.model_utils import *
import numpy as np
from model_evaluate.predictTimeSlice import predict_slice


THRESH_HOLD=0.5
UP=50
DOWN=350#450
LEFT=30
RIGHT=1030

import options
import argparse
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()
description=opt.description
model_name=opt.arch


def getData(item):
    seismPath = "/data/" #the base path of the kerry3d data
    gx = np.load(seismPath + 'kerry3d.npz')['arr_0']#(287, 735, 1252)
    gx=np.single(gx)
    gx=gx[item,:,:]
    gx=gx.transpose()
    return gx


def drawline(seismic,predict,item):
    origin=seismic[UP:DOWN,LEFT:RIGHT]
    predict=predict[UP:DOWN,LEFT:RIGHT]

    plt.matshow(predict, cmap=plt.get_cmap("Spectral"), alpha=0.3)

    mask=predict
    # 单通道转三通道
    ymax = 255
    ymin = 0
    xmax = max(map(max, origin))
    xmin = min(map(min, origin))
    origin_copy = np.copy(origin)
    for i in range(origin.shape[0]):
        for j in range(origin.shape[1]):
            origin_copy[i][j] = round(((ymax - ymin) * (origin_copy[i][j] - xmin) / (xmax - xmin)) + ymin)
    image = np.expand_dims(origin_copy, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    # 保存三通道地质图像
    # 将fault区域标注到图像上
    for i in range(origin.shape[0]):
        for j in range(origin.shape[1]):
            # 大于阈值区域上色
            if mask[i][j] >= THRESH_HOLD:
                image[i][j] = [0, 0, 255]
    # 保存标注好的图像
    path=os.path.join(opt.save_dir,'{}_{}_drawline_{}.png'.format(item, model_name,description))
    cv2.imwrite(path)


if __name__ == '__main__':

    array=[50,78]# timeslice number
    for item in array:
        seismic = getData(item)
        predict=predict_slice(seismic,model_name)
        drawline(seismic,predict,item)
