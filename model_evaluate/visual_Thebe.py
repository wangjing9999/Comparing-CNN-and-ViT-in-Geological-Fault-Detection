
import torchvision.transforms.functional as TF
from model_evaluate.predictTimeSlice import predict_slice

from utils.common_tools import *
from utils.image_tools import *

import matplotlib.pyplot as plt
import torch.utils.data

import options
import argparse
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()


model_name = opt.arch
save_path=opt.save_dir
best_iou_threshold=0.5

seed = opt.seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_seed(seed)

UPPER_BOUND = 800
LOWER_BOUND = 1300



def getData(item):
    data_path=opt.data_dir
    sesimic = np.load(os.path.join(data_path, "test/seismic/{}.npy".format(item)))
    sesimic = sesimic[UPPER_BOUND: LOWER_BOUND, :]
    fault = np.load(os.path.join(data_path, "test/annotation/{}.npy".format(item)))
    fault = fault[UPPER_BOUND: LOWER_BOUND, :]
    return sesimic,fault


def show_seismicwithmask(sesimic,fault):
    plt.imshow(np.squeeze(sesimic), cmap="gray", aspect='auto')
    plt.imshow(np.squeeze(fault), cmap='faults', aspect='auto')
    plt.rcParams['figure.figsize']=(45,16)
    plt.axis('off')
    path = os.path.join(save_path, "seismicwithmask_{}.png".format(item))
    plt.savefig(path)
    plt.clf()


def transparent_cmap(cmap, N=255):
    from matplotlib.colors import LinearSegmentedColormap
    # get colormap
    ncolors = 256
    color_array = plt.get_cmap('gray')(range(ncolors))
    # change alpha values
    color_array[:, -1] = np.linspace(0.0, 1.0, ncolors)
    color_array[255] = [1., 0., 0., 0.5]
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='faults', colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap
def show_seismic(sesimic,item):
    sesimic = sesimic[UPPER_BOUND: LOWER_BOUND, :]
    print(sesimic.shape)
    plt.imshow(np.squeeze(sesimic), cmap="gray", aspect='auto')
    plt.rcParams['figure.figsize'] = (45, 16)
    plt.axis('off')
    path=os.path.join(save_path,"seismic_{}.png".format(item))
    plt.savefig(path)



def drawsesiwithcolor(seis,predict,item):

    predicted_mask = predict < best_iou_threshold
    predict[predicted_mask == 1] = 0
    size_w, size_h = predict.shape

   # 宽度扩大2倍，以便看的更清晰
    output = TF.to_tensor(predict)
    CROP_SIZE = (size_w*2, size_h)
    output=output.unsqueeze(1)
    output = F.interpolate(output, size=CROP_SIZE, mode='bilinear', align_corners=False)
    output=output.squeeze().detach().cpu().numpy()

    seis = TF.to_tensor(seis)
    CROP_SIZE = (size_w * 2, size_h)
    seis = seis.unsqueeze(1)
    seis = F.interpolate(seis, size=CROP_SIZE, mode='bilinear', align_corners=False)
    seis = seis.squeeze().detach().cpu().numpy()

    plt.imshow(seis, cmap="gray",aspect=1.0,interpolation="bilinear")
    h, w = seis.shape
    y, x = np.mgrid[0:h, 0:w]
    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.jet)
    plt.contourf(x, y, output,  cmap=mycmap,aspect=1.0,interpolation="bilinear")
    plt.rcParams['figure.figsize'] = (35, 16)
    plt.axis('off')
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    path=os.path.join(save_path,"seismic_withcolormap.png".format(item))
    plt.savefig(path)
    plt.clf()

if __name__ == '__main__':
    number=141
    for item in range(number):

        seis,fault= getData(item)
        predict = predict_slice(seis, model_name)
        predict = predict[UPPER_BOUND: LOWER_BOUND, :]
        drawsesiwithcolor(seis,predict,item)

