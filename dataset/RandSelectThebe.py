
#RandSelectThebe:randselect the data for dataefficient test
import torch.utils.data
from os.path import splitext
from os import listdir
import numpy as np
import torchvision.transforms.functional as TF
from utils.image_tools import norm,faultseg_augumentation
from utils.common_tools import getPartDatasets
class FaultsDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir,rate=1, isTrain=True,seed=1234):#rate: the percent of the dataset
        self.images_dir = imgs_dir
        self.masks_dir = masks_dir
        self.isTrain=isTrain
        self.rate=rate
        # set randseed for random select the image
        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        if isTrain:
            self.train_list=getPartDatasets(self.ids,rate,seed)  #seed
            print(self.train_list)

    def __len__(self):
        if self.isTrain:
            return len(self.train_list)
        else:
            return len(self.ids)

    def transform(self, img, mask):
        # to tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        img=norm(img)
        img = TF.normalize(img, [0.5, ], [0.5, ])
        return img, mask


    def __getitem__(self, i):
        if self.isTrain:
            idx=self.train_list[i]
        else:
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

        img,mask=self.transform(img,mask)
        return (img, mask)




if __name__ == "__main__":
    data_path = "/data/wangjing/processedThebe224"
    faults_dataset_train = FaultsDataset(imgs_dir="{}/train/seismic".format(data_path),
                                         masks_dir="{}/train/annotation".format(data_path),
                                         rate=0.5,
                                         isTrain=True
                                         )
    print(len(faults_dataset_train))

