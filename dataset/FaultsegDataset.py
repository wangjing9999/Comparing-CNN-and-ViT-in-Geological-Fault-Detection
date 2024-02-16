import os

import torch
from os.path import splitext
from os import listdir
from torchvision import transforms
import torchvision.transforms.functional as TF
from utils.common_tools import resize
import numpy as np
import random
from utils.image_tools import norm,faultseg_augumentation
class faultSegDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir,isTrain=True):
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.isTrain=isTrain
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
    def __len__(self):
        return len(self.ids)

    def transform(self, img, mask):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img=norm(img)

        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
        img = img[:, i:i + h, j:j + w]
        mask = mask[:, i:i + h, j:j + w]

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # normalize based on dataset mean std
       # img = TF.normalize(img, [0.4915, ], [0.0655, ])
        img=TF.normalize(img,[0.5,],[0.5])
        return img, mask


    def __getitem__(self, i):
        idx = self.ids[i]

        mask_path="{}/{}.dat".format(self.masks_dir, idx)
        assert os.path.exists(mask_path), f'Files {mask_path} not exist'
        img_path = "{}/{}.dat".format(self.images_dir, idx)
        assert os.path.exists(img_path), f'Files {img_path} not exist'

        mask = np.fromfile(mask_path, dtype=np.single).reshape(128, 128, 128)
        img = np.fromfile(img_path, dtype=np.single).reshape(128, 128, 128)

        axis0or1 = np.random.randint(2, size=1)
        #axis0or1 = 1
        ithslice = np.random.randint(128, size=1)

        if axis0or1 == 0:
            #             print("axis 0 ")
            mask = mask[ithslice, :, :].squeeze().transpose()
            img = img[ithslice, :, :].squeeze().transpose()
        else:
            #             print("axis 1 ")
            mask = mask[:, ithslice, :].squeeze().transpose()
            img = img[:, ithslice, :].squeeze().transpose()


        #数据增强
            # data augumentation
        if self.isTrain:#训练集，数据增强
            aug = faultseg_augumentation(p=0.7)
            augmented = aug(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img, mask = self.transform(img, mask)

        return (img, mask)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    # data_path = r"E:\dataset\thebe_dataset"
    data_path = "/data/wangjing/faultseg/train"
    faults_dataset_train = faultSegDataset(imgs_dir="{}/seis".format(data_path),
                                         masks_dir="{}/fault".format(data_path),
                                           isTrain=False
                                         )
    # plt.figure(figsize=(1.4, 1.4))
    # plt.figure(figsize=(1.4,1.4))

    plt.axis('off')
    for i in range(2):
        img, mask = next(iter(faults_dataset_train))
        print(img.shape)
        plt.imshow(img.squeeze(),'gray')
        plt.savefig('/data/wangjing/pretrainmodel/faultseg_seis_{}.jpg'.format(i),dpi=300,  pad_inches=0.0,bbox_inches='tight')
    for i in range(2):
        #用opencv读取，转换成rgb图像
        img_cv=cv2.imread('/data/wangjing/pretrainmodel/faultseg_seis_{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE)
        img_cv=cv2.cvtColor(img_cv,cv2.COLOR_GRAY2RGB)
        cv2.imwrite('/data/wangjing/pretrainmodel/faultseg_seis_{}.jpg'.format(i),img_cv)

        # plt.imshow(mask.squeeze(), 'gray')
        # plt.savefig('mask_{}.png'.format(i), bbox_inches='tight')
