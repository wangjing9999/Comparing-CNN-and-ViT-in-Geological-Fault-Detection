# the process of using 2d models to predict seismic cube
# thanks to https://github.com/Jun-Tam/3D-Seismic-Image-Fault-Segmentation/blob/master/prediction.py
######### parser ###########

from utils.image_tools import *
import os
from utils.model_utils import create_model_faultseg
import matplotlib.pyplot as plt
from model_evaluate.predictTimeSlice import predict_slice
import options
import argparse
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()


model_name=opt.arch

save_path=opt.save_dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#path of the 3d cube
img_path = opt.data_dir


def getData():

    seis = np.load(os.path.join(img_path,"/3DCube/seistest.npy"))
    fault = np.load(os.path.join(img_path,"/3DCube/faulttest.npy"))

    seis = np.moveaxis(seis, -2, -1)  # make it IL, Z, XL order
    fault = np.moveaxis(fault, -2, -1)
    #use min_max norm
    seis = (seis - seis.min(axis=(1, 2), keepdims=True)) / (
                seis.max(axis=(1, 2), keepdims=True) - seis.min(axis=(1, 2), keepdims=True))

    return seis,fault


def predict(seis,model):
    c,h,w=seis.shape
    predict_block = np.zeros((c, h, w), dtype=np.double)
    for i in range(c):
        print(i)
        seis_slice=seis[i]
        pred_slice=predict_slice(seis_slice,model,i)
        predict_block[i,:,:]=pred_slice.squeeze()

    return predict_block
def create_img_alpha(img_input,threshold=0.5):
    ''' Overlay a translucent fault image on a seismic image '''
    img_alpha = np.zeros([np.shape(img_input)[0], np.shape(img_input)[1],4])
    img_input[img_input < threshold] = 0
    img_alpha[:,:,0] = 1 # Yellow: (1,1,0), Red: (1,0,0)
    img_alpha[:,:,1] = 0
    img_alpha[:,:,2] = 0
    img_alpha[...,-1] = img_input
    return img_alpha
def show_image_synth(seis_slice, fault_slice, pred_slice, title, threshold,save_path):
    ''' Show fault prediction result on synthetic data for validation '''
    fig, axes = plt.subplots(2,3,figsize=(28,5))
    for i,ax in enumerate(axes.flat):
        plt.axes(ax)
        plt.imshow(seis_slice.T,cmap=plt.cm.gray_r)
        if  i == 0:
            plt.title('Seismic')
        elif i == 1:
            plt.imshow(create_img_alpha(pred_slice.T,threshold), alpha=1)
            plt.title('Fault Probability')
        elif i == 2:
            plt.imshow(create_img_alpha(fault_slice.T), alpha=1)
            plt.title('True Mask')
        plt.tick_params(axis='both',which='both',
                        bottom=False,left=False,labelleft=False,labelbottom=False)
    plt.text(-145,140,title,fontsize=14)

    plt.savefig(os.path.join(save_path,"{}_predict.png".format(title)))
    plt.show()


def vlm_slicer(seis_vlms,pred_vlms,fault_vlms,idx_slice=0,flag_slice=0):
    ''' Slice a seismic sub-volume for display '''
    seis_vlm = seis_vlms.copy()
    pred_vlm = pred_vlms.copy()
    fault_vlm = fault_vlms.copy()
    if   flag_slice == 0:
        seis_slice = seis_vlm[:,:,idx_slice]
        fault_slice = fault_vlm[:,:,idx_slice]
        pred_slice = pred_vlm[:,:,idx_slice]
        prefix = 'z-slice'
    elif flag_slice == 1:
        seis_slice = seis_vlm[:,idx_slice,:]
        fault_slice = fault_vlm[:,idx_slice,:]
        pred_slice = pred_vlm[:,idx_slice,:]
        prefix = 'y-slice'
    elif flag_slice == 2:
        seis_slice = seis_vlm[idx_slice,:,:]
        fault_slice = fault_vlm[idx_slice,:,:]
        pred_slice = pred_vlm[idx_slice,:,:]
        seis_slice = seis_slice.T
        fault_slice = fault_slice.T
        pred_slice = pred_slice.T
        prefix = 'x-slice'
    title = 'Test Volume ID: '  + prefix + ': ' + str(idx_slice)
    print("1111",fault_slice.max(),fault_slice.min())
    return seis_slice, fault_slice, pred_slice, title


if __name__ == '__main__':
    seis,fault=getData()
    print("data load ok")
    model = create_model_faultseg(model_name=model_name).to(device)
    print("model load ok")
    predvlm=predict(seis,model)
    np.save(os.path.join(save_path, "predict_{}.npy".format(model_name)), predvlm)
    print("predict ok")
    for i in range(3):
        idx_slice = np.random.randint(500)
        flag_slice=2
        # for flag_slice in range(3):
        seis_slice, fault_slice, pred_slice, title = vlm_slicer(seis, predvlm, fault,  idx_slice=idx_slice, flag_slice=flag_slice)
        title = "{}_{}_{}_{}".format(title, model_name,flag_slice, idx_slice)
        show_image_synth(seis_slice, fault_slice, pred_slice, title, threshold=0.5, save_path=save_path)