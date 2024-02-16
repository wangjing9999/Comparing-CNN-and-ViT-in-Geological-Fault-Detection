import torch
import os
from collections import OrderedDict
from models.unet import UNet
from configs.config import get_config
from models.swinunet import SwinUnet
from models.TransUnet import VisionTransformer,CONFIGS
from models.TransAttUnet import UNet_Attention_Transformer_Multiscale
from models.resnet50unet import UNetWithResnet50Encoder
from models.swindeeplab import SwinDeepLab
import numpy as np
import models.TransUnet_vit_seg_configs as configs
import importlib

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)


def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir, "model_epoch_{}_{}.pth".format(epoch, session))
    torch.save(state, model_out_path)


def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except:
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr


def get_arch(opt):
    arch = opt.arch
    print('You choose ' + arch + '...')
    if arch == 'unet':
        model = UNet()
    elif arch == 'TransUnet':
        CONFIGS = {
            'R50-ViT-B_16': configs.get_r50_b16_config(),
        }

        vit_name = "R50-ViT-B_16"
        img_size = 224
        vit_patches_size = 16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformer(config_vit)
        model.load_from(weights=np.load(config_vit.pretrained_path))
    elif arch == 'swinunet':
        cfg = get_config()
        model = SwinUnet(cfg, img_size=224).cuda()
        model.load_from(cfg)
    elif arch == 'swindeeplab':
        model_config = importlib.import_module('models.configs.swin_224_7_2level')

        model = SwinDeepLab(
            model_config.EncoderConfig,
            model_config.ASPPConfig,
            model_config.DecoderConfig
        )
        if model_config.EncoderConfig.encoder_name == 'swin' and model_config.EncoderConfig.load_pretrained:
            model.encoder.load_from('../pretrainmodel/swin_tiny_patch4_window7_224.pth')
        if model_config.ASPPConfig.aspp_name == 'swin' and model_config.ASPPConfig.load_pretrained:
            model.aspp.load_from('../pretrainmodel/swin_tiny_patch4_window7_224.pth')
        if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and not model_config.DecoderConfig.extended_load:
            model.decoder.load_from('../pretrainmodel/swin_tiny_patch4_window7_224.pth')
        if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and model_config.DecoderConfig.extended_load:
            model.decoder.load_from_extended('../pretrainmodel/swin_tiny_patch4_window7_224.pth')
    elif arch == 'TransAttnUnet':
        model = UNet_Attention_Transformer_Multiscale(1,1)
    elif arch=='resnetunet':
        model=UNetWithResnet50Encoder()

    else:
        raise Exception("Arch error!")

    return model

def create_model_faultseg(model_name):

    if model_name=='unet':
        model=UNet(in_channels=1)
        weights_path="/data/wangjing/faultdetect/dataEfficient/thebe/swinunet/dataefficient_0.1/checkpoint_best_swinunet_dataefficient_0.1.pkl"
        weights = torch.load(weights_path)['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict)
    elif model_name == 'resnetunet':
        model = UNetWithResnet50Encoder()
        weights_path = "/data/wangjing/faultdetect/faultseg/resnetunet/1/checkpoint_best_faultseg_resnetunet_1.pkl"
        weights = torch.load(weights_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)

    elif model_name == "TransUnet":
        vit_name = "R50-ViT-B_16"
        img_size = 224
        vit_patches_size = 16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformer(config_vit)
        weights_path = "/data/wangjing/faultdetect/faultseg/TransUnet/3_lowaug_seed12/checkpoint_best_faultseg_TransUnet_3_lowaug_seed12.pkl"
        weights = torch.load(weights_path, map_location="cuda:1")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)
    elif model_name=='transattunet':
        model_nestunet_path = "/data/wangjing/faultdetect/faultseg/TransAttnUnet/1/checkpoint_best_faultseg_TransAttnUnet_1.pkl"
        model = UNet_Attention_Transformer_Multiscale(1,1)
        weights = torch.load(model_nestunet_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict)

    elif model_name=="swinunet":
        cfg = get_config()
        model = SwinUnet(cfg, img_size=224)
        weights_path = "/data/wangjing/faultdetect/dataEfficient/thebe/swinunet/dataefficient_0.06/checkpoint_best_swinunet_dataefficient_0.06.pkl"
        weights = torch.load(weights_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)
    elif model_name == 'swindeeplab':
        model_config = importlib.import_module('models.configs.swin_224_7_2level')

        model = SwinDeepLab(
            model_config.EncoderConfig,
            model_config.ASPPConfig,
            model_config.DecoderConfig
        )
        weights_path = "/data/wangjing/faultdetect/faultseg/swindeeplab/1/checkpoint_best_faultseg_swindeeplab_1.pkl"
        weights = torch.load(weights_path)['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)

    return model

def create_model_thebe(opt,model_name):
    checkpoint_dir=opt.save_dir
    weights_path = "{}/{}.pkl".format(checkpoint_dir, model_name)
    if model_name=='unet':
        model=UNet(in_channels=1)
        weights=torch.load(weights_path)['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict)
    elif model_name=='resnetunet':
        model=UNetWithResnet50Encoder()
        weights = torch.load(weights_path)['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)
    elif model_name == 'TransUnet':
        vit_name = "R50-ViT-B_16"
        img_size = 224
        vit_patches_size = 16
        config_vit = CONFIGS[vit_name]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        model = VisionTransformer(config_vit)
        weights = torch.load(weights_path)['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)

    elif model_name == 'swinunet':
        cfg = get_config()
        model = SwinUnet(cfg, img_size=224)
        weights = torch.load(weights_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict)

    elif model_name=='transattunet':
        model = UNet_Attention_Transformer_Multiscale(1,1)
        weights = torch.load(weights_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict)
    elif model_name == 'swindeeplab':

        model_config = importlib.import_module('models.configs.swin_224_7_2level')

        model = SwinDeepLab(
            model_config.EncoderConfig,
            model_config.ASPPConfig,
            model_config.DecoderConfig
        )
        weights = torch.load(weights_path, map_location="cuda")['model_state_dict']
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict)
    return model