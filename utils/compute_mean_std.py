import os
from PIL import Image
import numpy as np
from utils.image_tools import *
import torchvision.transforms.functional as TF
def main():
    img_channels = 1
    img_dir = "/home/wangjing/dataset/faultSeg/train/seis"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".dat")]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img = np.fromfile(img_path, dtype=np.single).reshape(128, 128, 128)
        # img先变成[0,1]
        img = TF.to_tensor(img)
        img=norm(img)
        # img = img[roi_img == 255]
        img=img.numpy()
        cumulative_mean += img.mean()
        cumulative_std += img.std()
        # cumulative_mean += img.mean(axis=0)
        # cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()