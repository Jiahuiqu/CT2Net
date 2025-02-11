import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet with PixelShuffle & MaxPool

class double_conv(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_features, output_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout2d(p=p_drop),
            nn.Conv2d(output_features, output_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class down_block(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(down_block, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=p_drop),
            double_conv(input_features, output_features, negative_slope, p_drop)
        )
    def forward(self, x):
        x = self.down(x)
        return x

class up_block(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(up_block, self).__init__()
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Sequential(
            nn.Dropout2d(p=p_drop),
            double_conv(int(input_features/4+output_features), output_features, negative_slope, p_drop)
        )
    def forward(self, x, x_pre):
        x = self.up(x)
        x = torch.cat((x, x_pre), 1)
        x = self.conv(x)
        return x

class SpeSRModel(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(SpeSRModel, self).__init__()
        model = []

        self.in_conv = double_conv(input_features, 64, negative_slope, p_drop)
        self.in_conv1 = nn.Conv2d(64, 64, kernel_size=2, padding=3, stride=1)
        self.down1 = down_block(64, 128, negative_slope, p_drop) # W/2
        self.down2 = down_block(128, 256, negative_slope, p_drop) # W/4
        self.down3 = down_block(256, 512, negative_slope, p_drop) # W/8
        self.bottleneck = down_block(512, 1024, negative_slope, p_drop) # W/16
        self.up1 = up_block(1024, 512, negative_slope, p_drop) # W/8
        self.up2 = up_block(512, 256, negative_slope, p_drop) # W/4
        self.up3 = up_block(256, 128, negative_slope, p_drop) # W/2
        self.out_conv = up_block(128, 64, negative_slope, p_drop) # W
        self.out_conv1 = nn.Conv2d(64, 64, kernel_size=6, padding=0, stride=1)
        self.out = nn.Conv2d(64, output_features, kernel_size=1, padding=0, bias=False)

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(output_features, output_features, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):

        conv = self.in_conv(x) # W 64
        conv = nn.functional.interpolate(conv, (32, 32))
        down1 = self.down1(conv) # W/2 128
        down2 = self.down2(down1) # W/4 256
        down3 = self.down3(down2) # W/8 512
        bottleneck = self.bottleneck(down3) # W/16 1024
        up = self.up1(bottleneck, down3) # W/8 512
        up = self.up2(up, down2) # W/4 256
        up = self.up3(up, down1) # W/2 128
        up = self.out_conv(up, conv) # W 64
        # up = self.out_conv1(up)
        up = nn.functional.interpolate(up, (27, 27))
        out = self.out(up)
        out = self.model(out)

        return out


class SpaSRModel(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(SpaSRModel, self).__init__()
        in_model = []
        out_model = []

        in_features = input_features
        out_features = 128
        for _ in range(2):
            in_model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                         nn.InstanceNorm2d(out_features),
                         nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        self.in_conv = double_conv(in_features, 64, negative_slope, p_drop)
        self.in_conv1 = nn.Conv2d(64, 64, kernel_size=2, padding=3, stride=1)
        self.down1 = down_block(64, 128, negative_slope, p_drop)  # W/2
        self.down2 = down_block(128, 256, negative_slope, p_drop)  # W/4
        self.down3 = down_block(256, 512, negative_slope, p_drop)  # W/8
        self.bottleneck = down_block(512, 1024, negative_slope, p_drop)  # W/16
        self.up1 = up_block(1024, 512, negative_slope, p_drop)  # W/8
        self.up2 = up_block(512, 256, negative_slope, p_drop)  # W/4
        self.up3 = up_block(256, 128, negative_slope, p_drop)  # W/2

        # Output layer
        out_model += [nn.ReflectionPad2d(3),
                      nn.Conv2d(128, output_features, 7),
                      nn.Tanh()]

        self.in_model = nn.Sequential(*in_model)
        self.out_model = nn.Sequential(*out_model)

    def forward(self, x):
        x = self.in_model(x)
        conv = self.in_conv(x)  # W 64
        conv = nn.functional.interpolate(conv, (128, 128))
        down1 = self.down1(conv)  # W/2 128
        down2 = self.down2(down1)  # W/4 256
        down3 = self.down3(down2)  # W/8 512
        bottleneck = self.bottleneck(down3)  # W/16 1024
        up = self.up1(bottleneck, down3)  # W/8 512
        up = self.up2(up, down2)  # W/4 256
        up = self.up3(up, down1)  # W/2 128
        up = nn.functional.interpolate(up, (81, 81))
        out = self.out_model(up)

        return out

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():  # 此处遍历model所有层
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2 * m.num_features]
            # print("mean:{} std:{}".format(mean,std))
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            # print('m.weight:{}'.format(m.weight))
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features:]

def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params

class GFNet(nn.Module):
    def __init__(self, channel=512):
        super(GFNet, self).__init__()
        self.g1 = nn.Conv2d(channel, channel, 1, 1, 0)
        self.f = nn.Conv2d(channel, channel, 1, 1, 0)
        self.h = nn.Conv2d(channel, channel, 1, 1, 0)
        self.softmax1 = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=-1)
        self.gamma = nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        self.gamma.weight.data = torch.Tensor(torch.full((1, channel, 1, 1), 0.0001))
        self.beta = nn.Conv2d(channel, channel, 1, 1, 0, bias=False)
        self.beta.weight.data = torch.Tensor(torch.full((1, channel, 1, 1), 0.0001))
        self.out_conv = nn.Conv2d(channel, channel, 1, 1, 0)

    def forward(self, content, guidance):
        G1 = self.g1(mean_variance_norm(content))
        F = self.f(mean_variance_norm(guidance))
        H = self.h(guidance)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G1.size()
        G1 = G1.view(b, -1, w * h)
        Atten = torch.bmm(F, G1)
        Atten = self.softmax1(Atten)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        G2 = torch.bmm(H, Atten.permute(0, 2, 1))
        G2 = self.softmax2(G2)
        b, c, h, w = guidance.size()
        G1 = G1.view(b, c, h, w)
        G2 = G2.view(b, c, h, w)
        gamma = self.gamma(G2)
        beta = self.beta(G2)
        Out = torch.matmul(G1, gamma)+beta
        Out = self.out_conv(Out)
        return Out

class guided_SpaSRModel(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(guided_SpaSRModel, self).__init__()
        in_model = []
        out_model = []

        in_features = input_features
        out_features = 128
        for _ in range(2):
            in_model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                         nn.InstanceNorm2d(out_features),
                         nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        self.in_conv = double_conv(in_features, 64, negative_slope, p_drop)
        self.down1 = down_block(64, 128, negative_slope, p_drop)  # W/2
        self.down2 = down_block(128, 256, negative_slope, p_drop)  # W/4
        self.down3 = down_block(256, 512, negative_slope, p_drop)  # W/8

        guided_features = output_features
        self.guided_in_conv = double_conv(guided_features, 64, negative_slope, p_drop)

        self.GFNet = GFNet(channel=512)

        self.bottleneck = down_block(512, 1024, negative_slope, p_drop)  # W/16
        self.up1 = up_block(1024, 512, negative_slope, p_drop)  # W/8
        self.up2 = up_block(512, 256, negative_slope, p_drop)  # W/4
        self.up3 = up_block(256, 128, negative_slope, p_drop)  # W/2

        self.STBlock = STBlock(128, 128, 3, 1, 1)
        # MLP to generate AdaIN parameters by style code
        self.mlp = MLP(input_dim=512 * 16 * 16, output_dim=get_num_adain_params(self.STBlock), dim=128,
                       n_blk=1, norm='none', activ='relu')

        # Output layer
        out_model += [nn.ReflectionPad2d(3),
                      nn.Conv2d(128, output_features, 7),
                      nn.Tanh()]

        # self.para = torch.nn.Parameter(torch.tensor([0.05]))
        self.in_model = nn.Sequential(*in_model)
        self.out_model = nn.Sequential(*out_model)



    def forward(self, x, guide_y):
        x = self.in_model(x)
        conv = self.in_conv(x)  # W 64
        conv = nn.functional.interpolate(conv, (128, 128))
        down1 = self.down1(conv)  # W/2 128
        down2 = self.down2(down1)  # W/4 256
        content = self.down3(down2)  # W/8 512

        guided_conv = self.guided_in_conv(guide_y)  # W 64
        guided_conv = nn.functional.interpolate(guided_conv, (128, 128))
        guided_down1 = self.down1(guided_conv)  # W/2 128
        guided_down2 = self.down2(guided_down1)  # W/4 256
        Dominant_features = self.down3(guided_down2)  # W/8 512

        G = self.GFNet(content, Dominant_features)
        down3 = content + G

        bottleneck = self.bottleneck(down3)  # W/16 1024

        up = self.up1(bottleneck, down3)  # W/8 512
        up = self.up2(up, down2)  # W/4 256
        up = self.up3(up, down1)  # W/2 128
        up = nn.functional.interpolate(up, (81, 81))

        style_params = self.mlp(Dominant_features)
        assign_adain_params(style_params, self.STBlock)

        # 重建图像
        up = self.STBlock(up)
        out = self.out_model(up)
        return out

class guided_SpeSRModel(nn.Module):
    def __init__(self, input_features, output_features, negative_slope=0, p_drop=0):
        super(guided_SpeSRModel, self).__init__()
        model = []

        self.in_conv = double_conv(input_features, 64, negative_slope, p_drop)
        self.down1 = down_block(64, 128, negative_slope, p_drop)  # W/2
        self.down2 = down_block(128, 256, negative_slope, p_drop)  # W/4
        self.down3 = down_block(256, 512, negative_slope, p_drop)  # W/8

        guided_features = output_features
        self.guided_in_conv = double_conv(guided_features, 128, negative_slope, p_drop)
        self.guided_down1 = down_block(128, 128, negative_slope, p_drop)  # W/2
        self.guided_down2 = down_block(128, 256, negative_slope, p_drop)  # W/4
        self.guided_down3 = down_block(256, 512, negative_slope, p_drop)  # W/8

        self.GFNet = GFNet(channel=512)

        self.bottleneck = down_block(512, 1024, negative_slope, p_drop)  # W/16
        self.up1 = up_block(1024, 512, negative_slope, p_drop)  # W/8
        self.up2 = up_block(512, 256, negative_slope, p_drop)  # W/4
        self.up3 = up_block(256, 128, negative_slope, p_drop)  # W/2
        self.out_conv = up_block(128, 64, negative_slope, p_drop)  # W
        self.out_conv1 = nn.Conv2d(64, 64, kernel_size=6, padding=0, stride=1)

        self.STBlock = STBlock(64, 64, 3, 1, 1)
        # MLP to generate AdaIN parameters by style code
        self.mlp = MLP(input_dim=512 * 4 * 4, output_dim=get_num_adain_params(self.STBlock), dim=128,
                       n_blk=1, norm='none', activ='relu')

        # Output layer
        self.out = nn.Conv2d(64, output_features, kernel_size=1, padding=0, bias=False)
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(output_features, output_features, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, guide_y):
        conv = self.in_conv(x)  # W 64
        conv = nn.functional.interpolate(conv, (32, 32))
        down1 = self.down1(conv)  # W/2 128
        down2 = self.down2(down1)  # W/4 256
        content = self.down3(down2)  # W/8 512

        guide_conv = self.guided_in_conv(guide_y)  # W 64
        guide_conv = nn.functional.interpolate(guide_conv, (32, 32))
        guide_down1 = self.guided_down1(guide_conv)  # W/2 128
        guide_down2 = self.guided_down2(guide_down1)  # W/4 256
        Dominant_feature = self.guided_down3(guide_down2)  # W/8 512

        G = self.GFNet(content, Dominant_feature)
        down3 = content+G

        bottleneck = self.bottleneck(down3)  # W/16 1024
        up = self.up1(bottleneck, down3)  # W/8 512
        up = self.up2(up, down2)  # W/4 256
        up = self.up3(up, down1)  # W/2 128
        up = self.out_conv(up, conv)  # W 64
        up = nn.functional.interpolate(up, (27, 27))

        style_params = self.mlp(Dominant_feature)
        assign_adain_params(style_params, self.STBlock)
        # 重建图像
        up = self.STBlock(up)

        out = self.out(up)
        out = self.model(out)
        return out

class STBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding):
        super(STBlock, self).__init__()
        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(num_blocks=1, dim=output_dim, norm='adain', activation='relu', pad_type='zero')]
        self.model = nn.Sequential(*self.model)

        self.use_bias = True
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)
        self.activation = nn.Tanh()


    def forward(self, x):
        out = self.conv(x)
        out = self.model(out)
        # out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Sequential Models
##################################################################################
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        # self.fc.weight = nn.Parameter(torch.Tensor(torch.full((output_dim, input_dim), 0.0001)))

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='adain', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        residual = x
        x = self.model(x)
        out = x + residual
        return out

##################################################################################
# Basic Blocks
##################################################################################
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        self.conv.weight.data = torch.Tensor(torch.full((input_dim, output_dim, kernel_size, kernel_size), 0.000001))


    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        # if self.activation:
        #     x = self.activation(x)
        return x

##################################################################################
# Discriminator
##################################################################################
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 3, stride=1, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 3, stride=1, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 3, stride=1, padding=1),

                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 3, stride=1, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 1, 1, padding=0)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # print("fake_hsi:{}".format(x.shape))
        x = self.model(x)
        # print("result:{}".format(x.shape))
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

if __name__ == '__main__':
    A = torch.randn((1, 194, 27, 27))
    B = torch.randn((1, 3, 81, 81))

    netG_hsi2rgb = guided_SpaSRModel(input_features=194, output_features=3)
    syn_rgb = netG_hsi2rgb(A, B)
    print(syn_rgb.shape)

    netG_rgb2hsi = guided_SpeSRModel(input_features=3, output_features=194)
    syn_hsi = netG_rgb2hsi(B, A)
    print(syn_hsi.shape)