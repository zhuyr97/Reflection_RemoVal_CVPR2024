import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/ghome/zhuyr/Deref_RW/')
print(f'Reflection Detection:  path: {sys.path}')
# from networks.efficientnet_pytorch import EfficientNet
from networks.efficientnet_pytorch import EfficientNet
import numpy as np
import os,cv2
import matplotlib.pyplot as plt
def feature_save(tensor,name):
    if not os.path.exists(str(name)):
        os.makedirs(str(name))
    for i in range(tensor.shape[1]):
        inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
        inp = np.clip(inp,0,1)
        inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp =np.squeeze(inp)
        plt.figure()
        plt.imshow(inp)
        plt.savefig(str(name) + '/' + str(i) + '.png')
        #cv2.imwrite(str(name)+'/'+str(i)+'.png',inp*255.0)
def feature_save1(tensor,name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    # tensor = torch.mean(tensor,dim=1).repeat(3,1,1)
    if not os.path.exists(str(name)):
        os.makedirs(str(name))
    for i in range(tensor.shape[1]):
        inp = tensor[:,i,:,:].detach().cpu().numpy().transpose(1,2,0)
        inp = np.clip(np.abs(inp),0,1)
        inp = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
        inp = np.squeeze(inp)
        #cv2.imwrite(str(name)+'/'+str(i)+'.png',inp*255.0)
        plt.figure()
        plt.imshow(inp)
        plt.savefig(str(name) + '/' + str(i) + '.png')

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConstantNormalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ConstantNormalize, self).__init__()
        mean = torch.Tensor(mean).view([1, 3, 1, 1])
        std = torch.Tensor(std).view([1, 3, 1, 1])
        # https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-5)


class Conv1x1(nn.Sequential):
    def __init__(self, in_planes, out_planes=16, has_se=False, se_reduction=None):
        if has_se:
            if se_reduction is None:
                # se_reduction= int(math.sqrt(in_planes))
                se_reduction = 2
            super(Conv1x1, self).__init__(SELayer(in_planes, se_reduction),
                                          nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          nn.ReLU()
                                          )
        else:
            super(Conv1x1, self).__init__(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          nn.ReLU()
                                          )


class ResBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion=1, stride=1):
        super(ResBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        # if stride == 1 and in_planes != out_planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, out_planes, kernel_size=1,
        #                   stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_planes),
        #     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + x #self.shortcut(x) if self.stride==1 else out
        return out

ml_features = []


def feature_hook(module, fea_in, fea_out):
    #     print("hooker working")
    # module_name.append(module.__class__)
    # features_in_hook.append(fea_in)
    global ml_features
    ml_features.append(fea_out)
    return None



class RefDet(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4):

        super(RefDet, self).__init__()

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone) # backbone : 'efficientnet-b3'
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)
        print("Total number of paramerters in EfficientNet networks is {} ".format(
            sum(x.numel() for x in self.feat_net.parameters())))
        print("Total number of requires_grad paramerters in EfficientNet networks is {} ".format(
            sum(p.numel() for p in self.feat_net.parameters() if p.requires_grad)))

        # remove classification head to get correct param count
        self.feat_net._avg_pooling = None
        self.feat_net._dropout = None
        self.feat_net._fc = None
        self.expansion = expansion
        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)

        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False

        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes]
        self.proj_convs = nn.ModuleList(proj_convs)

        if backbone == 'efficientnet-b0':
            channel_factor = 8
        elif backbone == 'efficientnet-b1':
            channel_factor = 12
        elif backbone == 'efficientnet-b2':
            channel_factor = 12
        elif backbone == 'efficientnet-b3':
            channel_factor = 13
        self.temp_conv = Conv1x1(proj_planes * channel_factor,proj_planes * channel_factor, has_se=has_se)

        # two stream feature
        self.stem_conv = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)  # 1*1
        convs = []
        for i in range(num_of_layers):
           convs.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        self.convs = nn.Sequential(*convs)

        # prediction
        pred_layers = []
        pred_layers.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_conv = nn.Sequential(*pred_layers)



        # for m in self.modules():
        #     if isinstance(m, nn.ReLU):
        #         m.inplace = False

    def forward(self, x):
        global ml_features

        b, c, h, w = x.size()
        ml_features = []

        _ = self.feat_net.extract_features(self.norm(x))
        h_f, w_f = ml_features[0].size()[2:]
        proj_features = []
        for i in range(len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature = torch.cat(proj_features, dim=1)  # cat_feature [N,13*16,h_f, w_f ]
        #print(f'cat_feature:{cat_feature.size()}' )
        cat_feature = self.temp_conv(cat_feature)
        out_l = self.stem_conv(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        #print(f'out_l (before convs):{out_l.size()}' )

        for conv in self.convs:
           out_l = conv(out_l)
        #print(f'out_l (after convs):{out_l.size()}' )
        Sparse_out = F.interpolate(self.pred_conv(out_l), size=(h, w), mode='bilinear')

        return Sparse_out  #F.sigmoid(S_out)


class RefDetDual(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4):

        super(RefDetDual, self).__init__()

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone) # backbone : 'efficientnet-b3'
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)
        print("Total number of paramerters in EfficientNet networks is {} ".format(
            sum(x.numel() for x in self.feat_net.parameters())))
        print("Total number of requires_grad paramerters in EfficientNet networks is {} ".format(
            sum(p.numel() for p in self.feat_net.parameters() if p.requires_grad)))

        # remove classification head to get correct param count
        self.feat_net._avg_pooling = None
        self.feat_net._dropout = None
        self.feat_net._fc = None
        self.expansion = expansion
        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)

        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False

        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes]
        self.proj_convs = nn.ModuleList(proj_convs)

        if backbone == 'efficientnet-b0':
            channel_factor = 8
        elif backbone == 'efficientnet-b1':
            channel_factor = 12
        elif backbone == 'efficientnet-b2':
            channel_factor = 12
        elif backbone == 'efficientnet-b3':
            channel_factor = 13
        self.temp_conv = Conv1x1(proj_planes * channel_factor,proj_planes * channel_factor, has_se=has_se)

        # two stream feature
        #  stream1
        self.stem_conv = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)  # 1*1
        convs = []
        for i in range(num_of_layers):
           convs.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        self.convs = nn.Sequential(*convs)

        # prediction
        pred_layers = []
        pred_layers.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_conv = nn.Sequential(*pred_layers)


        #  stream2
        self.stem_conv1 = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)  # 1*1
        convs1 = []
        for i in range(num_of_layers):
           convs1.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        self.convs1 = nn.Sequential(*convs1)

        # prediction
        pred_layers1 = []
        pred_layers1.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_conv1 = nn.Sequential(*pred_layers1)


        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        global ml_features

        b, c, h, w = x.size()
        ml_features = []

        _ = self.feat_net.extract_features(self.norm(x))
        h_f, w_f = ml_features[0].size()[2:]
        proj_features = []
        for i in range(len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature = torch.cat(proj_features, dim=1)  # cat_feature [N,13*16,h_f, w_f ]
        #print(f'cat_feature:{cat_feature.size()}' )
        cat_feature = self.temp_conv(cat_feature)

        out_l = self.stem_conv(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        #print(f'out_l (before convs):{out_l.size()}' )

        for conv in self.convs:
           out_l = conv(out_l)
        #print(f'out_l (after convs):{out_l.size()}' )
        Dense_out = F.interpolate(self.pred_conv(out_l), size=(h, w), mode='bilinear')


        out_r = self.stem_conv1(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        for conv in self.convs1:
           out_r = conv(out_r)
        Sparse_out = F.interpolate(self.pred_conv1(out_r), size=(h, w), mode='bilinear')


        return Dense_out, Sparse_out  #F.sigmoid(S_out)



class RefDetDual_V2(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4):
        super(RefDetDual_V2, self).__init__()

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone) # backbone : 'efficientnet-b3'
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)
        print("Total number of paramerters in EfficientNet networks is {} ".format(
            sum(x.numel() for x in self.feat_net.parameters())))
        print("Total number of requires_grad paramerters in EfficientNet networks is {} ".format(
            sum(p.numel() for p in self.feat_net.parameters() if p.requires_grad)))

        # remove classification head to get correct param count
        self.feat_net._avg_pooling = None
        self.feat_net._dropout = None
        self.feat_net._fc = None
        self.expansion = expansion
        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)

        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False

        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes]
        self.proj_convs = nn.ModuleList(proj_convs)

        if backbone == 'efficientnet-b0':
            channel_factor = 8
        elif backbone == 'efficientnet-b1':
            channel_factor = 12
        elif backbone == 'efficientnet-b2':
            channel_factor = 12
        elif backbone == 'efficientnet-b3':
            channel_factor = 13
        self.temp_conv = Conv1x1(proj_planes * channel_factor,proj_planes * channel_factor, has_se=has_se)


        # two stream feature
        #  stream1
        self.stem_conv = Conv1x1(proj_planes * len(in_planes), pred_planes*2, has_se=has_se)  # 1*1
        self.pred_dense_head = nn.Linear(pred_planes * 2, 1, bias=False)
        # convs = []
        # for i in range(num_of_layers):
        #    convs.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        # self.convs = nn.Sequential(*convs)

        # prediction
        # pred_layers = []
        # pred_layers.append(nn.Conv2d(pred_planes, 1, 1))
        # self.pred_conv = nn.Sequential(*pred_layers)


        #  stream2
        self.stem_conv1 = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)  # 1*1
        convs1 = []
        for i in range(num_of_layers):
           convs1.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        self.convs1 = nn.Sequential(*convs1)

        # prediction
        pred_layers1 = []
        pred_layers1.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_sparse_head = nn.Sequential(*pred_layers1)




        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        global ml_features

        b, c, h, w = x.size()
        ml_features = []

        _ = self.feat_net.extract_features(self.norm(x))
        h_f, w_f = ml_features[0].size()[2:]
        proj_features = []
        for i in range(len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature = torch.cat(proj_features, dim=1)  # cat_feature [N,13*16,h_f, w_f ]
        #print(f'cat_feature:{cat_feature.size()}' )
        cat_feature = self.temp_conv(cat_feature)

        #print(f'out_l (before convs):{out_l.size()}' )

        # for conv in self.convs:
        #    out_l = conv(out_l)
        #print(f'out_l (after convs):{out_l.size()}' )

        out_l = self.stem_conv(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        avgP_feas = F.adaptive_avg_pool2d(out_l, 1)
        Dense_out = self.pred_dense_head(avgP_feas.view(x.shape[0], -1))  #F.interpolate(self.pred_conv(out_l), size=(h, w), mode='bilinear')


        out_r = self.stem_conv1(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        for conv in self.convs1:
           out_r = conv(out_r)
        Sparse_out = F.interpolate(self.pred_sparse_head(out_r), size=(h, w), mode='bilinear')

        return Dense_out, Sparse_out #F.interpolate(out_l, size=(h, w), mode='bilinear')   #F.sigmoid(S_out)




class RefDetDual_V2_loss(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4):
        super(RefDetDual_V2_loss, self).__init__()

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone) # backbone : 'efficientnet-b3'
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)
        print("Total number of paramerters in EfficientNet networks is {} ".format(
            sum(x.numel() for x in self.feat_net.parameters())))
        print("Total number of requires_grad paramerters in EfficientNet networks is {} ".format(
            sum(p.numel() for p in self.feat_net.parameters() if p.requires_grad)))

        # remove classification head to get correct param count
        self.feat_net._avg_pooling = None
        self.feat_net._dropout = None
        self.feat_net._fc = None
        self.expansion = expansion
        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)

        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False

        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes]
        self.proj_convs = nn.ModuleList(proj_convs)

        if backbone == 'efficientnet-b0':
            channel_factor = 8
        elif backbone == 'efficientnet-b1':
            channel_factor = 12
        elif backbone == 'efficientnet-b2':
            channel_factor = 12
        elif backbone == 'efficientnet-b3':
            channel_factor = 13
        self.temp_conv = Conv1x1(proj_planes * channel_factor,proj_planes * channel_factor, has_se=has_se)

        # two stream feature
        #  stream1
        self.stem_conv = Conv1x1(proj_planes * len(in_planes), pred_planes*2, has_se=has_se)  # 1*1
        self.pred_dense_head = nn.Linear(pred_planes * 2, 1, bias=False)

        #  stream2
        self.stem_conv1 = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)  # 1*1
        convs1 = []
        for i in range(num_of_layers):
           convs1.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        self.convs1 = nn.Sequential(*convs1)

        # prediction
        pred_layers1 = []
        pred_layers1.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_sparse_head = nn.Sequential(*pred_layers1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        global ml_features

        b, c, h, w = x.size()
        ml_features = []

        _ = self.feat_net.extract_features(self.norm(x))
        h_f, w_f = ml_features[0].size()[2:]
        proj_features = []
        for i in range(len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature = torch.cat(proj_features, dim=1)  # cat_feature [N,13*16,h_f, w_f ]
        cat_feature = self.temp_conv(cat_feature)

        out_l = self.stem_conv(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        avgP_feas = F.adaptive_avg_pool2d(out_l, 1)
        Dense_out = self.pred_dense_head(avgP_feas.view(x.shape[0], -1))  #F.interpolate(self.pred_conv(out_l), size=(h, w), mode='bilinear')


        out_r = self.stem_conv1(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        for conv in self.convs1:
           out_r = conv(out_r)
        Sparse_out = F.interpolate(self.pred_sparse_head(out_r), size=(h, w), mode='bilinear')

        return Dense_out, Sparse_out , cat_feature  #F.sigmoid(S_out)



class RefDetDual_V3(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4):
        super(RefDetDual_V3, self).__init__()

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone) # backbone : 'efficientnet-b3'
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)
        print("Total number of paramerters in EfficientNet networks is {} ".format(
            sum(x.numel() for x in self.feat_net.parameters())))
        print("Total number of requires_grad paramerters in EfficientNet networks is {} ".format(
            sum(p.numel() for p in self.feat_net.parameters() if p.requires_grad)))

        # remove classification head to get correct param count
        self.feat_net._avg_pooling = None
        self.feat_net._dropout = None
        self.feat_net._fc = None
        self.expansion = expansion
        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)

        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False

        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes]
        self.proj_convs = nn.ModuleList(proj_convs)

        if backbone == 'efficientnet-b0':
            channel_factor = 8
        elif backbone == 'efficientnet-b1':
            channel_factor = 12
        elif backbone == 'efficientnet-b2':
            channel_factor = 12
        elif backbone == 'efficientnet-b3':
            channel_factor = 13
        self.temp_conv = Conv1x1(proj_planes * channel_factor,proj_planes * channel_factor, has_se=has_se)


        # two stream feature
        #  stream1
        self.stem_conv = Conv1x1(proj_planes * len(in_planes), pred_planes*2, has_se=has_se)  # 1*1
        self.pred_dense_head = nn.Linear(pred_planes * 2, 1, bias=False)
        # convs = []
        # for i in range(num_of_layers):
        #    convs.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        # self.convs = nn.Sequential(*convs)

        # prediction
        # pred_layers = []
        # pred_layers.append(nn.Conv2d(pred_planes, 1, 1))
        # self.pred_conv = nn.Sequential(*pred_layers)


        #  stream2
        self.stem_conv1 = Conv1x1(proj_planes * len(in_planes), pred_planes, has_se=has_se)  # 1*1
        convs1 = []
        for i in range(num_of_layers):
           convs1.append(ResBlock(in_planes=pred_planes,out_planes=pred_planes,expansion= self.expansion, stride=1))
        self.convs1 = nn.Sequential(*convs1)

        # prediction
        pred_layers1 = []
        pred_layers1.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_sparse_head = nn.Sequential(*pred_layers1)




        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    def forward(self, x):
        global ml_features

        b, c, h, w = x.size()
        ml_features = []

        _ = self.feat_net.extract_features(self.norm(x))
        h_f, w_f = ml_features[0].size()[2:]
        proj_features = []
        for i in range(len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature = torch.cat(proj_features, dim=1)  # cat_feature [N,13*16,h_f, w_f ]
        #print(f'cat_feature:{cat_feature.size()}' )
        cat_feature = self.temp_conv(cat_feature)

        #print(f'out_l (before convs):{out_l.size()}' )

        # for conv in self.convs:
        #    out_l = conv(out_l)
        #print(f'out_l (after convs):{out_l.size()}' )

        out_l = self.stem_conv(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        avgP_feas = F.adaptive_avg_pool2d(out_l, 1)
        Dense_out = self.pred_dense_head(avgP_feas.view(x.shape[0], -1))  #F.interpolate(self.pred_conv(out_l), size=(h, w), mode='bilinear')


        out_r = self.stem_conv1(cat_feature)  # stem_feat [N,32,h_f, w_f ]
        for conv in self.convs1:
           out_r = conv(out_r)
        Sparse_out = F.interpolate(self.pred_sparse_head(out_r), size=(h, w), mode='bilinear')

        return Dense_out, Sparse_out , F.interpolate(out_l, size=(h, w), mode='bilinear')

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = RefDet(backbone='efficientnet-b3',use_pretrained=False, num_of_layers=4).to(device)
    # #print(model)
    # input = torch.randn(1, 3, 256, 256).to(device)
    # S_out= model(input)
    # print('-'*50)
    # print(f'S_out: {S_out.shape}')
    # print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    #
    # model1 = RefDetDual_V3(backbone='efficientnet-b3',use_pretrained=False, num_of_layers=4,
    #                        proj_planes=16,
    #                        pred_planes=72,
    #
    #                        fix_backbone=False,
    #                        has_se=False,
    #                        expansion=4).to(device)
    # #print(model)
    # input = torch.randn(1, 3, 256, 256).to(device)
    # Dense_out, Sparse_out, dense_fea  = model1(input)
    # print('-'*50)
    # print(f'Dense_out: {Dense_out.shape} || dense_fea: {dense_fea.shape}')
    # print(f'Sparse_out: {Sparse_out.shape}')
    #
    # print('#generator parameters:', sum(param.numel() for param in model1.parameters()))

    input = torch.randn(1, 3, 256, 256)#.to(device)
    net_Det = RefDet(backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=False,
                 fix_backbone=False,
                 has_se=False,
                 num_of_layers=6,
                 expansion = 4)

    from thop import profile
    flops, params = profile(net_Det, inputs=(input,))
    print(flops, params, '----', flops / 1000000000, params / 1000000)



