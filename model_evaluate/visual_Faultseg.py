import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from utils.common_tools import setup_seed
from utils.model_utils import create_model_faultseg
import cv2
import cmapy
from utils.image_tools import norm
best_iou_threshold=0.5
import options
import argparse
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()

def visual_result(rows,cols):
    sampleNo = np.random.random_integers(0, 4, cols)
    print(sampleNo)
    f, axarr = plt.subplots(rows, cols, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    [axi.set_axis_off() for axi in axarr.ravel()]
    base_path=opt.data_dir
    for ith in range(cols):
        i = sampleNo[ith]
        mask = np.fromfile("{}/test/fault/{}.dat".format(base_path,i), dtype=np.single).reshape(128,128,128)
        seis = np.fromfile("{}/test/seis/{}.dat".format(base_path,i), dtype=np.single).reshape(128,128,128)
        axis0or1 = np.random.randint(2, size=1)
        ithslice = np.random.randint(128, size=1)

        if axis0or1 == 0:
            mask = mask[:, :, ithslice].squeeze().transpose()  # [:96,:96]
            seis = seis[:, :,ithslice].squeeze().transpose()  # [:96,:96]
        else:
            mask = mask[:,ithslice, :].squeeze().transpose()  # [:96,:96]
            seis = seis[:,ithslice, :].squeeze().transpose()  # [:96,:96]

        img = TF.to_tensor(seis)
        mask = TF.to_tensor(mask)

        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(112, 112))
        img = img[:, i:i + h, j:j + w]
        mask = mask[:, i:i + h, j:j + w]
        img = norm(img)
        img = TF.normalize(img, [0.5, ], [0.5, ])

        CROP_SIZE = (224, 224)
        from torch.nn import functional as F
        img_224=F.interpolate(img.unsqueeze(0), size=CROP_SIZE, mode='bilinear', align_corners=False)
        img = img_224

        pred_unet = model_unet(img.float().to(device=device))
        pred_unet = pred_unet.detach().cpu().numpy()
        predicted_mask = pred_unet < best_iou_threshold
        pred_unet[predicted_mask == True] = 0.0

        pred_resnetunet = model_resnetunet(img.float().to(device=device))
        pred_resnetunet = pred_resnetunet.detach().cpu().numpy()
        predicted_mask = pred_resnetunet < best_iou_threshold
        pred_resnetunet[predicted_mask == True] = 0.0

        pred_transunet = model_transunet(img.float().to(device=device))
        pred_transunet = pred_transunet.detach().cpu().numpy()
        predicted_mask = pred_transunet < best_iou_threshold
        pred_transunet[predicted_mask == True] = 0.0


        pred_transattnunet = model_transattnunet(img.float().to(device=device))
        pred_transattnunet = pred_transattnunet.detach().cpu().numpy()
        predicted_mask = pred_transattnunet < best_iou_threshold
        pred_transattnunet[predicted_mask == True] = 0.0

        pred_swindeeplab = model_swindeeplab(img.float().to(device=device))
        pred_swindeeplab=pred_swindeeplab.detach().cpu().numpy()
        predicted_mask = pred_swindeeplab < best_iou_threshold
        pred_swindeeplab[predicted_mask == True] = 0.0


        pred_swinunet = model_swinunet(img.float().to(device=device))
        pred_swinunet = pred_swinunet.detach().cpu().numpy()
        predicted_mask = pred_swinunet < best_iou_threshold
        pred_swinunet[predicted_mask == True] = 0.0
        mask = mask.squeeze().detach().cpu().numpy()
        #set with color map
        mask_vision = cv2.applyColorMap((mask * 255).astype(np.uint8), cmapy.cmap('jet_r'))
        pred_unet_vision = cv2.applyColorMap((pred_unet.squeeze() * 255).astype(np.uint8), cmapy.cmap('jet_r'))
        pred_resnetunet_vision = cv2.applyColorMap((pred_resnetunet.squeeze() * 255).astype(np.uint8),
                                                 cmapy.cmap('jet_r'))
        pred_transunet_vision = cv2.applyColorMap((pred_transunet.squeeze() * 255).astype(np.uint8),cmapy.cmap('jet_r'))
        pred_transattnunet_vision = cv2.applyColorMap((pred_transattnunet.squeeze() * 255).astype(np.uint8),
                                                  cmapy.cmap('jet_r'))
        pred_swinunet_vision = cv2.applyColorMap((pred_swinunet.squeeze() * 255).astype(np.uint8),
                                                 cmapy.cmap('jet_r'))
        pred_swindeeplab_vision = cv2.applyColorMap((pred_swindeeplab.squeeze() * 255).astype(np.uint8), cmapy.cmap('jet_r'))



        axarr[ith,0].imshow(img[0].squeeze(), cmap='gray')

        axarr[ith,1].imshow(mask_vision)
        axarr[ith,2].imshow(pred_unet_vision)
        axarr[ith,3].imshow(pred_resnetunet_vision)
        axarr[ith,4].imshow(pred_transunet_vision)
        axarr[ith,5].imshow(pred_transattnunet_vision)
        axarr[ith,6].imshow(pred_swinunet_vision)
        axarr[ith,7].imshow(pred_swindeeplab_vision)

    plt.savefig("faultSeg_{}.png".format(sampleNo), bbox_inches='tight')

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
    return mycmap


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------unet model------------------
    model_unet = create_model_faultseg('unet')
    model_unet = model_unet.to(device)
    model_unet.eval()

    # --------------resnetunet model------------------
    model_resnetunet = create_model_faultseg('resnetunet')
    model_resnetunet = model_resnetunet.to(device)
    model_resnetunet.eval()


    # --------TransUnet----------
    model_transunet = create_model_faultseg('TransUnet')
    model_transunet = model_transunet.to(device)
    model_transunet.eval()


# ------transattnunet---------
    model_transattnunet = create_model_faultseg('transattunet')
    model_transattnunet = model_transattnunet.to(device)
    model_transattnunet.eval()

# ------swinunet--------
    model_swinunet = create_model_faultseg('swinunet')
    model_swinunet = model_swinunet.to(device)
    model_swinunet.eval()


    # # --------------swindeeplab model------------------
    model_swindeeplab = create_model_faultseg('swindeeplab')
    model_swindeeplab = model_swindeeplab.to(device)
    model_swindeeplab.eval()

    cols = 8
    rows =8

    mycmap = transparent_cmap(plt.cm.jet)
    seed=np.random.randint(1000)
    setup_seed(seed)
    visual_result(rows, cols)







