
import torch
import torch.nn as nn
from losses.DiceLoss import BinaryDiceLoss

class BCEDiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCEDiceLoss, self).__init__()
        self.bce_func =  nn.BCELoss()
        self.dice_func = BinaryDiceLoss()

    # loss = loss_f(outputs_1.cpu(), outputs.cpu(), labels.cpu())
    def forward(self, predict, target):
        loss_bce=self.bce_func(predict,target)
        loss_dice=self.dice_func(predict,target)
        return 0.5*loss_dice + 0.5*loss_bce


if __name__ == "__main__":

    fake_out = torch.tensor([0.67, 0.22, 0.05, 0.34], dtype=torch.float32)
    fake_label = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    loss_f = BCEDiceLoss()
    loss = loss_f(fake_out, fake_label)

    print(loss)




