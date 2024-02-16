from models.unet_parts import *
from models.unet_parts_att_transformer import *
from models.unet_parts_att_multiscale import *


class UNet_Attention_Transformer_Multiscale(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Attention_Transformer_Multiscale, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)

        self.up2 = Up(1024, 256 // factor, bilinear)
        self.up3 = Up(512, 128 // factor, bilinear)
        self.up4 = Up(256, 64, bilinear)
        self.outc = OutConv(128, n_classes)

        '''位置编码'''
        self.pos = PositionEmbeddingLearned(512 // factor)

        '''空间注意力机制'''
        self.pam = PAM_Module(512)

        '''自注意力机制'''
        self.sdpa = ScaledDotProductAttention(512)

        '''残差多尺度连接'''
        self.fuse1 = MultiConv(768, 256)
        self.fuse2 = MultiConv(384, 128)
        self.fuse3 = MultiConv(192, 64)
        self.fuse4 = MultiConv(128, 64)

    def forward(self, x):
        if x.size()[1] == 1 and self.n_channels == 3:  # 如果channel 是1，变成3
            x = x.repeat(1, 3, 1, 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        '''Setting 1'''
        x5_pam = self.pam(x5)
        '''Setting 2'''
        x5_pos = self.pos(x5)
        x5 = x5 + x5_pos
        x5_sdpa = self.sdpa(x5)
        x5 = x5_sdpa + x5_pam
        x6 = self.up1(x5, x4)
        x5_scale = F.interpolate(x5, size=x6.shape[2:], mode='bilinear', align_corners=True)
        x6_cat = torch.cat((x5_scale, x6), 1)

        x7 = self.up2(x6_cat, x3)
        x6_scale = F.interpolate(x6, size=x7.shape[2:], mode='bilinear', align_corners=True)
        x7_cat = torch.cat((x6_scale, x7), 1)

        x8 = self.up3(x7_cat, x2)
        x7_scale = F.interpolate(x7, size=x8.shape[2:], mode='bilinear', align_corners=True)
        x8_cat = torch.cat((x7_scale, x8), 1)

        x9 = self.up4(x8_cat, x1)
        x8_scale = F.interpolate(x8, size=x9.shape[2:], mode='bilinear', align_corners=True)
        x9 = torch.cat((x8_scale, x9), 1)

        logits = self.outc(x9)
        logits = torch.sigmoid(logits)
        return logits

if __name__ == '__main__':
    x=torch.randn(1,1,224,224)
    model=UNet_Attention_Transformer_Multiscale(1,1)
    y=model(x)
    print(y.shape)