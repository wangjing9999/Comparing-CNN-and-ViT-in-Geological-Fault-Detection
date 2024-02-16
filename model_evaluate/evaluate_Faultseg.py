#for 2D Thebe evaluation
from torch.autograd import Variable
from  utils.common_tools import *
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_curve,auc
from dataset.FaultsegDataset import faultSegDataset
from utils.model_utils import *
best_iou_threshold=0.5

from torch.nn import functional as F




# ######### Set options ###########
import argparse
import options
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()
setup_seed(opt.seed)
batch_size=opt.batch_size
val_iterations=opt.val_iterations
model_name = opt.arch
workers=opt.eval_workers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def metrices_fun(outputs,masks):
    avg_accuracy_score = 0.0
    avg_precision_score = 0.0
    avg_recall_score = 0.0
    avg_f1_score = 0.0
    avg_auc = 0.0
    for i in range(len(outputs)):
        accuracy_score1 = accuracy_score(outputs[i], masks[i])
        avg_accuracy_score += accuracy_score1

        precision_score1 = precision_score(outputs[i], masks[i])
        avg_precision_score += precision_score1

        recall_score1 = recall_score(outputs[i], masks[i], pos_label=1)
        avg_recall_score += recall_score1

        f1_score1 = f1_score(outputs[i], masks[i])
        avg_f1_score += f1_score1

        fpr, tpr, thresholds = roc_curve(outputs[i], masks[i], pos_label=1)
        auc1 = auc(fpr, tpr)
        avg_auc += auc1
    avg_accuracy_score /= len(outputs)
    avg_accuracy_score *= 100
    avg_accuracy_score = ('%.2f' % avg_accuracy_score)

    avg_precision_score /= len(outputs)
    avg_precision_score *= 100
    avg_precision_score = ('%.2f' % avg_precision_score)

    avg_recall_score /= len(outputs)
    avg_recall_score *= 100
    avg_recall_score = ('%.2f' % avg_recall_score)

    avg_f1_score /= len(outputs)
    avg_f1_score *= 100
    avg_f1_score = ('%.2f' % avg_f1_score)

    avg_auc /= len(outputs)
    avg_auc *= 100
    avg_auc = ('%.2f' % avg_auc)

    print(f"avg_accuracy_score={avg_accuracy_score}%\navg_precision_score={avg_precision_score}%\n"
          f"avg_recall_score={avg_recall_score}%\navg_f1_score={avg_f1_score}%\navg_auc={avg_auc}%")

def main():
    print("{} is evaluate".format(model_name))

    # 1.loading test data
    data_path = opt.data_dir
    faults_dataset_val = faultSegDataset(imgs_dir="{}/test/seis".format(data_path),
                                         masks_dir="{}/test/fault".format(data_path))

    valid_loader = torch.utils.data.DataLoader(dataset=faults_dataset_val,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               shuffle=False)
    print("data load ok")
    # 2. loading model
    model = create_model_faultseg(model_name)
    model = model.to(device)
    print("model load ok")
    # 3.evaluate the metrics
    model.eval()
    masks=[]
    outputs=[]

    with torch.no_grad():
        for ith_iter in range(val_iterations):
            print(ith_iter)
            train_iter = iter(valid_loader)
            images, mask = next(train_iter)
            images = Variable(images.float().to(device=device))
            mask = Variable(mask.float().to(device=device))
            #reshape to 224*224
            CROP_SIZE = (224, 224)
            images = F.interpolate(images, size=CROP_SIZE, mode='bilinear', align_corners=False)
            output = model(images)
            CROP_SIZE=(112,112)
            output = F.interpolate(output, size=CROP_SIZE, mode='bilinear', align_corners=False)
            predicted_mask = output > best_iou_threshold
            predicted_mask=predicted_mask.byte()
            predicted_mask = predicted_mask.cpu()
            predicted_mask = torch.squeeze(predicted_mask)
            predicted_mask=predicted_mask.numpy()
            predicted_mask=predicted_mask.flatten()
            mask=mask.byte()
            mask=mask.cpu()
            mask=torch.squeeze(mask)
            mask=mask.numpy()
            mask=mask.flatten()

            masks.append(mask)
            outputs.append(predicted_mask)

        metrices_fun(outputs,masks)

if __name__ == '__main__':

    main()
