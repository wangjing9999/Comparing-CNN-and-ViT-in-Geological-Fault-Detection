# Dataset of Thebe
import torch.utils.data
from os.path import splitext
from os import listdir
import numpy as np
import torchvision.transforms.functional as TF
from utils.image_tools import norm,faultseg_augumentation

class FaultsDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir, isTrain=True):
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
        img = TF.normalize(img, [0.5, ], [0.5, ])
        return img, mask


    def __getitem__(self, i):
        idx = self.ids[i]
        mask = np.load("{}/{}.npy".format(self.masks_dir, idx))
        img = np.load("{}/{}.npy".format(self.images_dir, idx))

        img = np.asarray(img, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        if self.isTrain:  # 训练集，数据增强
            aug = faultseg_augumentation(p=0.7)

            augmented = aug(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'


        # step2: 图像预处理
        img,mask=self.transform(img,mask)
        return (img, mask)




if __name__ == "__main__":

    import cv2
    import matplotlib.pyplot as plt
    # data_path = r"E:\dataset\thebe_dataset"
    data_path = "/data/wangjing/processedThebe"
    faults_dataset_train = FaultsDataset(imgs_dir="{}/train/seismic".format(data_path),
                                         masks_dir="{}/train/annotation".format(data_path),
                                         isTrain=True
                                         )
    for i in range(2):
        img, mask = next(iter(faults_dataset_train))
        print(img.shape)
        plt.imshow(img.squeeze(), 'gray')
        plt.savefig('/data/wangjing/pretrainmodel/thebe_seis_{}.jpg'.format(i), dpi=300, pad_inches=0.0,
                    bbox_inches='tight')
        # 用opencv读取，转换成rgb图像
        img_cv = cv2.imread('/data/wangjing/pretrainmodel/thebe_seis_{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('/data/wangjing/pretrainmodel/faultseg_seis_{}.jpg'.format(i), img_cv)

