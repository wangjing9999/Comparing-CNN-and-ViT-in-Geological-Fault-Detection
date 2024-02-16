
from torch.autograd import Variable
import argparse
import options
from utils.common_tools import  *
from utils.model_utils import *
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
from dataset.FaultsegDataset import *
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='fault detection')).parse_args()
print(opt)

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
torch.backends.cudnn.benchmark = True


from losses.bce_dice_loss import BCEDiceLoss
from utils.model_utils import *
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR

from utils.common_tools import resize
description=opt.description
resize=True
def main():
    ######### Logs dir ###########
    log_dir = os.path.join(opt.save_dir,  "faultseg", opt.arch,description )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("Now time is : ", datetime.datetime.now().isoformat())
    logger = create_logger(log_dir)

    logger.info(opt)


    setup_seed(opt.seed)
    ######### Model ###########
    model = get_arch(opt)

    ######### Optimizer ###########
    start_epoch = 1
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=opt.weight_decay)
    else:
        raise Exception("Error optimizer...")

    ######### DataParallel ###########
    model = torch.nn.DataParallel(model)
    model.cuda()

    ######### Scheduler ###########
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 30
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

    ######### Resume ###########
    if opt.resume:
        path_chk_rest = opt.pretrain_weights
        print("Resume from " + path_chk_rest)
        load_checkpoint(model, path_chk_rest)
        start_epoch = load_start_epoch(path_chk_rest) + 1
        lr = load_optim(optimizer, path_chk_rest)


        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')


    ######### Loss ###########
    criterion = BCEDiceLoss().cuda()

    ######### DataLoader ###########
    print('===> Loading datasets')

    train_dataset = faultSegDataset(imgs_dir="{}/train/seis".format(opt.data_dir),
                                           masks_dir="{}/train/fault".format(opt.data_dir),isTrain=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batch_size,
                                               num_workers=opt.train_workers,
                                               shuffle=True)


    val_dataset = faultSegDataset(imgs_dir="{}/validation/seis".format(opt.data_dir),
                                         masks_dir="{}/validation/fault".format(opt.data_dir),isTrain=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=opt.batch_size,
                                               num_workers=opt.eval_workers,
                                               shuffle=False)





    ######### train ###########
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
    t_start_train = time.time()
    mean_train_losses = []
    mean_val_losses = []
    mean_train_accuracies = []
    mean_val_accuracies = []
    t_start = time.time()
    best_miou = 0
    logger.info("start training")
    for epoch in range(start_epoch, opt.nepoch):
        one_epoch_start_time = time.time()
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        model.train()

        for ith_iter in range(opt.train_iterations):
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
            images = Variable(images.float()).cuda()
            masks = Variable(masks.float()).cuda()
            if resize:
                CROP_SIZE = (224, 224)
                from torch.nn import functional as F
                images = F.interpolate(images, size=CROP_SIZE, mode='bilinear', align_corners=False)
                outputs = model(images)
                CROP_SIZE = (112, 112)
                from torch.nn import functional as F
                outputs = F.interpolate(outputs, size=CROP_SIZE, mode='bilinear', align_corners=False)
            else:
                outputs = model(images)
            #         break
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_losses.append(loss.data)
            predicted_mask = outputs > opt.best_iou_threshold
            train_acc = iou_pytorch(predicted_mask.squeeze(1).byte(), masks.squeeze(1).byte(), opt.SMOOTH)
            train_accuracies.append(train_acc.mean())
            if ith_iter % opt.print_iter == 0:
                logger.info(
                    "Epoch {}:[{}/{}] ,Acc: {:.2%},Loss: {:.4f}".format(epoch, ith_iter + 1, opt.train_iterations,
                                                                        train_acc.mean().item(), loss.item()))

        end_one_epoch_time = time.time()
        logger.info("epoch{} traing time:{}".format(epoch, ((end_one_epoch_time - one_epoch_start_time) / 3600)))

        # validate the model
        model.eval()
        for ith_iter in range(opt.val_iterations):
            val_iter = iter(val_loader)
            images, masks = next(val_iter)
            images = Variable(images.float()).cuda()
            masks = Variable(masks.float()).cuda()
            if resize:
                CROP_SIZE = (224, 224)
                from torch.nn import functional as F
                images = F.interpolate(images, size=CROP_SIZE, mode='bilinear', align_corners=False)
                outputs = model(images)
                CROP_SIZE = (112, 112)
                from torch.nn import functional as F
                outputs = F.interpolate(outputs, size=CROP_SIZE, mode='bilinear', align_corners=False)
            else:
                outputs = model(images)
            loss = criterion(outputs, masks)

            val_losses.append(loss.data)
            predicted_mask = outputs > opt.best_iou_threshold
            val_acc = iou_pytorch(predicted_mask.byte(), masks.squeeze(1).byte(), opt.SMOOTH)
            val_accuracies.append(val_acc.mean())

        mean_train_losses.append(torch.mean(torch.stack(train_losses)))
        mean_val_losses.append(torch.mean(torch.stack(val_losses)))
        mean_train_accuracies.append(torch.mean(torch.stack(train_accuracies)))
        mean_val_accuracies.append(torch.mean(torch.stack(val_accuracies)))

        scheduler.step(torch.mean(torch.stack(val_losses)))

        for param_group in optimizer.param_groups:
            learningRate = param_group['lr']

        # Print Epoch result
        t_end = time.time()

        val_iou = torch.mean(torch.stack(val_accuracies))

        logger.info(
            'Epoch: {}. Train Loss: {:.4f}. Val Loss: {:.4f}. Train IoU: {:.4f}. Val IoU: {:.4f}. Time: {}. LR: {}'
            .format(epoch, torch.mean(torch.stack(train_losses)), torch.mean(torch.stack(val_losses)),
                    torch.mean(torch.stack(train_accuracies)), val_iou,
                    t_end - t_start, scheduler.get_lr()[0]))

        # optimizer.param_groups[0]["lr"]
        if best_miou < val_iou or epoch == opt.nepoch - 1:

            best_epoch = epoch if best_miou < val_iou else best_epoch
            best_miou = val_iou if best_miou < val_iou else best_miou

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_miou": best_miou}
            pkl_name = "checkpoint_faultseg_{}_{}_{}.pkl".format(epoch,
                                                     opt.arch,description) if epoch == opt.nepoch - 1 else "checkpoint_best_faultseg_{}_{}.pkl".format(opt.arch,description)


            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)

        total_training_time = time.time() - t_start_train
        logger.info("total training time: {}".format(total_training_time / 3600))

if __name__ == '__main__':
    main()