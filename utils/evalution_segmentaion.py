import numpy as np
import torch
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.f1_score=0
        self.Recall=0
        self.Precious=0
        self.eps=1e-5

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    def Mean_F1(self):
        return self.f1_score
    def Mean_Recall(self):
        return self.Recall
    def Mean_Precious(self):
        return self.Precious
    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.Precious,self.Recall,self.f1_score=self.caculate_f1score(pre_image,gt_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
    def caculate_f1score(self,pred,gt):

        output = pred.reshape(-1, )
        target = gt.reshape(-1, )
        tp = np.sum(output * target)  # TP
        fp = np.sum(output * (1 - target))  # FP
        fn = np.sum((1 - output) * target)  # FN
        tn = np.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        f1 = 2 * precision * recall / (precision + recall)
        return  precision,recall,f1

# if __name__ == "__main__":
#     miou = Evaluator(2)
#     miouVal = 0
#     accVal = 0
#     for index, (predict, label) in enumerate(MIoU_dataloader):
#         predict = predict.cpu().numpy()
#         label = label.cpu().numpy()
#         miou.add_batch(label,predict)
#         accVal += miou.Pixel_Accuracy()
#         miouVal += miou.Mean_Intersection_over_Union()
#         print('acc and miou are {},{}'.format(miou.Pixel_Accuracy(),miou.Mean_Intersection_over_Union()))
#     print('all acc and miou are {},{}'.format(accVal/len(MIoU_dataloader),miouVal/len(MIoU_dataloader)))