
import torchvision.transforms.functional as TF
import torch.utils.data
from utils.image_tools import *
from utils.model_utils import *
import torch.utils.data
import time

batch_size = 32
best_iou_threshold=0.5

UPPER_BOUND = 800
LOWER_BOUND = 1300
import options
import argparse
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()

model_name = opt.arch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path=opt.save_dir

def predict_All(save_path):
    # thebe
    model = create_model_thebe(opt).to(device)

    SAMPLE_NAMES = [str(i) for i in np.arange(141)]
    for i in range(len(SAMPLE_NAMES)):
        item = SAMPLE_NAMES[i]
        seis = np.load("{}/test/seismic/{}.npy".format(opt.data_dir, item))
        t1 = time.time()
        recover_Y_test_pred=predictOneTimeSlice(model, seis)
        np.save(os.path.join(save_path, "{}_{}".format(model_name, item)),
                np.squeeze(recover_Y_test_pred)[UPPER_BOUND:LOWER_BOUND, :])

        t2 = time.time()
        print('save in {} sec'.format(t2 - t1))


if __name__ == '__main__':
    main()



