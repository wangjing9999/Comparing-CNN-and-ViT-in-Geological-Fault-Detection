

from  utils.common_tools import *
from model_evaluate.predictTimeSlice import predict_slice
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,roc_curve,auc
best_iou_threshold=0.5

UPPER_BOUND = 800
LOWER_BOUND = 1300
print("UPPER_BOUND", UPPER_BOUND)
print("LOWER_BOUND", LOWER_BOUND)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
import options
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()
model_name=opt.arch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def getData(item):
    data_path=opt.data_dir
    sesimic = np.load(os.path.join(data_path, "test/seismic/{}.npy".format(item)))
    sesimic = sesimic[UPPER_BOUND: LOWER_BOUND, :]
    fault = np.load(os.path.join(data_path, "test/annotation/{}.npy".format(item)))

    fault = fault[UPPER_BOUND: LOWER_BOUND, :]
    print(fault.shape)
    pred=predict_slice(sesimic,model_name)
    return sesimic,fault,pred

def getmetrics(pred,fault):
    pred = np.where(pred > best_iou_threshold, 1, 0)
    pred = pred.flatten()
    fault = fault.flatten()

    accuracy_score_item = accuracy_score(fault, pred)


    precision_score_item = precision_score(fault, pred)
    recall_score_item = recall_score(fault, pred, pos_label=1)

    f1_score_item = f1_score(fault, pred)

    fpr, tpr, thresholds = roc_curve(fault, pred, pos_label=1)
    auc_item = auc(fpr, tpr)
    return accuracy_score_item,precision_score_item,recall_score_item,f1_score_item,auc_item

def main():
    torch.cuda.set_device(0)
    print("{} is evaluate".format(model_name))

    num = 141
    avg_accuracy_score = 0.0
    avg_precision_score = 0.0
    avg_recall_score = 0.0
    avg_f1_score = 0.0
    avg_auc = 0.0

    for item in range(num):
        print(item)
        _,fault,pred=getData(item)
        accuracy_score_item,precision_score_item,recall_score_item,f1_score_item,auc_item=getmetrics(pred,fault)
        avg_accuracy_score+=accuracy_score_item
        avg_precision_score += precision_score_item
        avg_recall_score += recall_score_item
        avg_f1_score += f1_score_item
        avg_auc += auc_item
        avg_accuracy_score /= num
        avg_accuracy_score *= 100
        avg_accuracy_score = ('%.2f' % avg_accuracy_score)

        avg_precision_score /= num
        avg_precision_score *= 100
        avg_precision_score = ('%.2f' % avg_precision_score)

        avg_recall_score /= num
        avg_recall_score *= 100
        avg_recall_score = ('%.2f' % avg_recall_score)

        avg_f1_score /= num
        avg_f1_score *= 100
        avg_f1_score = ('%.2f' % avg_f1_score)

        avg_auc /= num
        avg_auc *= 100
        avg_auc = ('%.2f' % avg_auc)

    print(f"avg_accuracy_score={avg_accuracy_score}%\navg_precision_score={avg_precision_score}%\n"
          f"avg_recall_score={avg_recall_score}%\navg_f1_score={avg_f1_score}%\navg_auc={avg_auc}%")


if __name__ == '__main__':
    main()
