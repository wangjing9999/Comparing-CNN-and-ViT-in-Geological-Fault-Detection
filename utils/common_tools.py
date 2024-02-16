
import logging
from datetime import datetime
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import os
from natsort import natsorted
from glob import glob
from torch import nn
from torch.nn.modules.utils import _pair

def resize(img,size,fill=0,method='padding'):
    _, ow, oh = img.shape
    diff_x = size - ow
    diff_y = size - oh
    if method=='constant_padding':
        img = F.pad(img, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2], 'constant',fill)
    elif method=='reflect_padding':
        pad=nn.ReflectionPad2d([diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        img=pad(img)
    elif method=='replication_padding':
        pad=nn.ReplicationPad2d([diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        img=pad(img)
    elif method=="interpolate":
        CROP_SIZE=_pair(size)
        img=img.unsqueeze(0)
        # linear | bilinear | bicubic | trilinear
        img = F.interpolate(img, size=CROP_SIZE, mode='bicubic', align_corners=False)
        img=img.squeeze(0)
    elif method=='extend':
        img=img.squeeze(0)
        left_img=img[:diff_x // 2]
        right=diff_x - diff_x // 2
        right_img=img[-right:]
        print(left_img.shape,right_img.shape)
        img=torch.concat([left_img,img,right_img],dim=0)
        up_img = img[:, :diff_y // 2]
        down = diff_y - diff_y // 2
        down_img = img[:, -down:]
        print("111  ",up_img.shape,img.shape,down_img.shape)
        img=torch.concat([up_img,img,down_img],dim=1)
        img=img.unsqueeze(0)
        print(img.shape)
    elif method=='hybrid':
        # step1:插值到180
        _,w,h=img.shape
        interplot_extendsize=(size-w)//3+w
        CROP_SIZE = _pair(interplot_extendsize)
        img = img.unsqueeze(0)
        # linear | bilinear | bicubic | trilinear
        img = F.interpolate(img, size=CROP_SIZE, mode='bicubic', align_corners=False)
        img = img.squeeze(0)
        # step2:replication
        pad = nn.ReplicationPad2d([diff_x // 2, diff_x - diff_x // 2,
                                   0, 0])
        img = pad(img)
    return img



def getPartDatasets(list,rate,seed=1234):

    count=len(list)
    train_num=int(count*rate)
    train_list=[]
    setup_seed(seed)
    train_idx = random.sample(range(0, count),train_num)
    for item in train_idx:
        train_list.append(list[item])
    return train_list


def acc_metrics(outputs, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(outputs)):
    # TP    predict 和 label 同时为1
        TP += ((outputs[i] == 1) & (labels[i] == 1)).sum()
        # TN    predict 和 label 同时为0
        TN += ((outputs[i] == 0) & (labels[i] == 0)).sum()
        # FN    predict 0 label 1
        FN += ((outputs[i] == 0) & (labels[i] == 1)).sum()
        # FP    predict 1 label 0
        FP += ((outputs[i] == 1) & (labels[i] == 0)).sum()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return p, r, F1, acc


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor,smooth):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    return iou


def setup_seed(seed=12345):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


def show_confMat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, perc=False):
    """
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes: list or tuple, 类别名称
    :param set_name: str, 数据集名称 train or valid or test_from_anyu?
    :param out_dir:  str, 图片要保存的文件夹
    :param epoch:  int, 第几个epoch
    :param verbose: bool, 是否打印精度信息
    :param perc: bool, 是否采用百分比，图像分割时用，因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10]
    plt.figure(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_tmp, cmap=cmap)
    plt.colorbar(fraction=0.03)

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:

        cls_per_nums = confusion_mat.sum(axis=1).reshape((cls_num, 1))
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                         fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir



def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_last_path(path, session):
	x = natsorted(glob(os.path.join(path,'*%s'%session)))[-1]
	return x


# def check_data_dir(path_tmp):
#     assert os.path.exists(path_tmp), \
#         "\n\n路径不存在，当前变量中指定的路径是：\n{}\n请检查相对路径的设置，或者文件是否存在".format(os.path.abspath(path_tmp))
def _upsample_like(src,tar):

    # src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    _,_,w,h=tar.size()
    size=(w,h)
    src=F.interpolate(src,size,mode='bilinear',align_corners=False)

    return src

def create_logger(BASE_DIR):
    from datetime import datetime

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    path_log = os.path.join(BASE_DIR, "{}_log.log".format(time_str))
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger


def adjust_learning_rate(optimizer, epoch, args, multiple):
    """Sets the learning rate to the initial LR decayed by 0.95 every 20 epochs"""
    # lr = args.lr * (0.95 ** (epoch // 4))
    lr = args.lr * (0.95 ** (epoch // 20))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]


if __name__ == "__main__":

    setup_seed(2)
    print(np.random.randint(0, 10, 1))
