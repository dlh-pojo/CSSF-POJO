
import torch
import torch.nn as nn
import numpy as np
import util
from util import *



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):#指归一化操作
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)#减去均值/方差
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def adain(content_features, style_features):
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

projection_style = nn.Sequential(
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=128)
)

projection_content = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=128)
)


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs




class AdaAttN(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(AdaAttN, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.conv0 = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(mean_variance_norm(content_key))
        G = self.g(mean_variance_norm(style_key))
        H = self.h(mean_variance_norm(style))#风格图
        b, _, h_g, w_g = G.size()#b,c,h,w
        G = G.view(b, -1, w_g * h_g).contiguous()#使用contiguous()不改变原值
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)#产生随机数
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]#[:,2]：是取全部的意思
            G = G[:, :, index]#调整图像大小
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()#V，transpose()函数的作用就是调换数组的行列值的索引值，类似于求矩阵的转置：
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)#.permute是将维度进行换位
        S = torch.bmm(F, G)#相乘操作
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)#S*V
        # std: b, n_c, c
        # std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        mean = self.conv0(mean)
        # std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return adain(content , mean)

class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes,
                                      key_planes=key_planes + 512 if shallow_layer else key_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))  # 填充
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1,
                content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):

        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed) +
            self.upsample5_1(self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed))))

class Decoder(nn.Module):

    def __init__(self, skip_connection_3=False):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256 + 256 if skip_connection_3 else 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is None:
            cs = self.decoder_layer_2(cs)
        else:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        return cs

class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter,skip_connection, shallow_layer, device):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.device = device
        self.visual_names = ['c', 'cs', 's']
        self.model_names = ['decoder', 'transformer']
        parameters = []
        self.max_sample = 64 * 64
        self.seed = 6666
        self.lambda_local = 1.
        self.lambda_global = 10.
        if skip_connection:
            self.net_adaattn_3 = AdaAttN(in_planes=256, key_planes=256 + 128 +64 if shallow_layer else 256,
                                max_sample=self.max_sample)
            self.model_names.append('adaattn_3')

        if shallow_layer:
            channels = 512 + 256 + 128 + 64
        else:
            channels = 512

        #projection
        self.proj_style = projection_style
        self.proj_content = projection_content

        #transform
        self.transform = Transformer(in_planes = 512, key_planes=channels, shallow_layer=shallow_layer)
        self.decoder = Decoder(skip_connection)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        if(start_iter > 0):
            self.transform.load_state_dict(torch.load('./experiments/transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('./experiments/decoder_iter_' + str(start_iter) + '.pth'))
            self.net_adaattn_3.load_state_dict(torch.load('./experiments/net_adaattn_3_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm = False):
        if(norm == False):
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)



    def mean_variance_norm(self, feat):  # 指归一化操作
        size = feat.size()
        mean, std = calc_mean_std(feat)
        normalized_feat = (feat - mean.expand(size)) / std.expand(size)  # 减去均值/方差
        return normalized_feat

    def get_key(self, feats, last_layer_idx, need_shallow=True):  # feats指的是特征
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))  # 将图片上/下采样到指定的大小
            results.append(mean_variance_norm(feats[last_layer_idx]))  # 拼接操作
            return torch.cat(results, dim=1)
        else:
            return mean_variance_norm(feats[last_layer_idx])

    def compute_style_loss(self, s_feats, c_feats, stylized_feats,shallow_layer):
        if self.lambda_global > 0:
            s_feats_mean, s_feats_std = calc_mean_std(s_feats[1])
            stylized_feats_mean, stylized_feats_std = calc_mean_std(stylized_feats[1])
            loss_global = 4 *self.mse_loss(
                    stylized_feats_mean, s_feats_mean) + self.mse_loss(stylized_feats_std, s_feats_std)
            s_feats_mean, s_feats_std = calc_mean_std(s_feats[2])
            stylized_feats_mean, stylized_feats_std = calc_mean_std(stylized_feats[2])
            loss_global += 4 *self.mse_loss(
                    stylized_feats_mean, s_feats_mean) + self.mse_loss(stylized_feats_std, s_feats_std)
            for i in range(3, 5):
                s_feats_mean, s_feats_std = calc_mean_std(s_feats[i])
                stylized_feats_mean, stylized_feats_std = calc_mean_std(stylized_feats[i])
                loss_global += self.mse_loss(
                    stylized_feats_mean, s_feats_mean) + self.mse_loss(stylized_feats_std, s_feats_std)
        if self.lambda_local > 0:
            c_key = self.get_key(c_feats, 1, shallow_layer)
            s_key = self.get_key(s_feats, 1, shallow_layer)
            s_value = s_feats[1]
            b, _, h_s, w_s = s_key.size()
            s_key = s_key.view(b, -1, h_s * w_s).contiguous()
            if h_s * w_s > self.max_sample:
                torch.manual_seed(self.seed)
                index = torch.randperm(h_s * w_s)[:self.max_sample]
                s_key = s_key[:, :, index]
                style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
            else:
                style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
            b, _, h_c, w_c = c_key.size()
            c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
            attn = torch.bmm(c_key, s_key)
            # S: b, n_c, n_s
            attn = torch.softmax(attn, dim=-1)
            # mean: b, n_c, c
            mean = torch.bmm(attn, style_flat)
            # std: b, n_c, c
            std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
            # mean, std: b, c, h, w
            mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
            std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
            loss_local = self.mse_loss(stylized_feats[1], adain(c_feats[1] , mean))
            for i in range(2, 5):
                c_key = self.get_key(c_feats, i, shallow_layer)
                s_key = self.get_key(s_feats, i, shallow_layer)
                s_value = s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    torch.manual_seed(self.seed)
                    index = torch.randperm(h_s * w_s)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                loss_local += self.mse_loss(stylized_feats[i], adain(c_feats[i] , mean))
        return loss_local * self.lambda_local + loss_global * self.lambda_global


    def forward(self, content, style, batch_size,shallow_layer,skip_connection):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        content3_1_key = self.get_key(content_feats,2,shallow_layer)
        style3_1_key = self.get_key(style_feats,2,shallow_layer)
        content4_1_key = self.get_key(content_feats,3, shallow_layer)
        style4_1_key = self.get_key(style_feats, 3, shallow_layer)
        content5_1_key = self.get_key(content_feats, 4, shallow_layer)
        style5_1_key = self.get_key(style_feats, 4, shallow_layer)

        if skip_connection:
            c_adain_feat_3 = self.net_adaattn_3(content_feats[2], style_feats[2], content3_1_key,style3_1_key, self.seed)
        else:
            c_adain_feat_3 = None

        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4], content4_1_key, style4_1_key, content5_1_key, style5_1_key)
        g_t = self.decoder(stylized,c_adain_feat_3)
        
        
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[0], content_feats[0], norm=False) + self.calc_content_loss(g_t, content, norm=False)
        for i in range(1,5):
            loss_c += self.calc_content_loss(g_t_feats[i], content_feats[i], norm=False)
        # loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(g_t_feats[4], content_feats[4], norm=True) + self.calc_content_loss(g_t_feats[2], content_feats[2],
        #                                                                          norm=True)
        # loss_s = self.calc_content_loss(g_t_feats[0], style_feats[0]) + self.calc_content_loss(g_t,style)
        # for i in range(1, 5):
        #     loss_s += self.calc_content_loss(g_t_feats[i], style_feats[i])
        loss_s = self.compute_style_loss(style_feats, content_feats, g_t_feats, shallow_layer)
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4],content4_1_key,content4_1_key,content5_1_key,content5_1_key),c_adain_feat_3)
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4],style4_1_key,style4_1_key,style5_1_key,style5_1_key),c_adain_feat_3)
        l_identity1 = self.calc_content_loss(Icc, content)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],style_feats[i])
        return g_t, loss_c, loss_s, l_identity1, l_identity2
