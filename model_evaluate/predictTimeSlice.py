from utils.image_tools import *
import os
import torchvision.transforms.functional as TF
from utils.model_utils import create_model_thebe
import matplotlib.pyplot as plt
class faultsDataset(torch.utils.data.Dataset):
    def __init__(self,preprocessed_images):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.images = preprocessed_images
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = TF.to_tensor(image)
        image=norm(image)
        image = TF.normalize(image, [0.5, ], [0.5, ])
        return image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict_slice(seis,model_name):
    Z, XL = seis.shape
    batch_size=2
    im_height = Z
    im_width = XL
    splitsize = 224  # 96
    stepsize = 112  # overlap half
    overlapsize = splitsize - stepsize

    horizontal_splits_number = int(np.ceil((im_width) / stepsize))
    width_after_pad = stepsize * horizontal_splits_number + 2 * overlapsize
    left_pad = int((width_after_pad - im_width) / 2)
    right_pad = width_after_pad - im_width - left_pad

    vertical_splits_number = int(np.ceil((im_height) / stepsize))
    height_after_pad = stepsize * vertical_splits_number + 2 * overlapsize

    top_pad = int((height_after_pad - im_height) / 2)
    bottom_pad = height_after_pad - im_height - top_pad

    horizontal_splits_number = horizontal_splits_number + 1
    vertical_splits_number = vertical_splits_number + 1

    X_list = []

    X_list.extend(
        split_Image(seis, True, top_pad, bottom_pad, left_pad, right_pad, splitsize, stepsize, vertical_splits_number,
                    horizontal_splits_number))

    X = np.asarray(X_list)

    faults_dataset_test = faultsDataset(X)

    test_loader = torch.utils.data.DataLoader(dataset=faults_dataset_test,
                                              batch_size=batch_size,
                                              shuffle=False)
    # 加载模型
    test_predictions = []
    imageNo = -1
    mergemethod = "smooth"
    model=create_model_thebe(model_name)
    for images in test_loader:
        images = images.type(torch.FloatTensor)
        images = images.to(device)
        outputs = model(images)

        y_preds = outputs.squeeze()
        test_predictions.extend(y_preds.detach().cpu())

        if len(test_predictions) >= vertical_splits_number * horizontal_splits_number:
            imageNo = imageNo + 1
            tosave = torch.stack(test_predictions).detach().cpu().numpy()[
                     0:vertical_splits_number * horizontal_splits_number]
            test_predictions = test_predictions[vertical_splits_number * horizontal_splits_number:]

            if mergemethod == "smooth":
                WINDOW_SPLINE_2D = window_2D(window_size=splitsize, power=2)
                # add one dimension
                tosave = np.expand_dims(tosave, -1)
                tosave = np.array([patch * WINDOW_SPLINE_2D for patch in tosave])  # 224,224,450
                tosave = tosave.reshape((vertical_splits_number, horizontal_splits_number, splitsize, splitsize, 1))
                recover_Y_test_pred = recover_Image(tosave, (im_height, im_width, 1), left_pad, right_pad, top_pad,
                                                    bottom_pad, overlapsize)

    return recover_Y_test_pred