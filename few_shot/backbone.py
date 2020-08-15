# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x, return_activation_l1=False):
        if not return_activation_l1:
            return x.view(x.size(0), -1)
        else:
            return x.view(x.size(0), -1), 0


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

dropout_layers = 2
finetune_wo_infodrop = False  # when finetuning without InfoDrop, turn this on

class Info_Dropout(nn.Module):
    def __init__(self, indim, outdim, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, if_pool=False, pool_kernel_size=2, pool_stride=None,
                 pool_padding=0, pool_dilation=1):
        super(Info_Dropout, self).__init__()
        if groups != 1:
            raise ValueError('InfoDropout only supports groups=1')

        self.indim = indim
        self.outdim = outdim
        self.if_pool = if_pool
        self.drop_rate = 0.1
        self.temperature = 0.5
        self.band_width = 1.0
        self.radius = 3

        self.patch_sampling_num = 9

        self.all_one_conv_indim_wise = nn.Conv2d(self.patch_sampling_num, self.patch_sampling_num,
                                                 kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation,
                                                 groups=self.patch_sampling_num, bias=False)
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise = nn.Conv2d(self.patch_sampling_num, outdim, kernel_size=1, padding=0, bias=False)
        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


        if if_pool:
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding, pool_dilation)

        self.padder = nn.ConstantPad2d((padding + self.radius, padding + self.radius + 1,
                                         padding + self.radius, padding + self.radius + 1), 0)

    def initialize_parameters(self):
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


    def forward(self, x_old, x):
        if finetune_wo_infodrop:
            return x

        with torch.no_grad():
            distances = []
            padded_x_old = self.padder(x_old)
            sampled_i = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).tolist()
            sampled_j = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).tolist()
            for i, j in zip(sampled_i, sampled_j):
                tmp = padded_x_old[:, :, self.radius: -self.radius - 1, self.radius: -self.radius - 1] - \
                      padded_x_old[:, :, self.radius + i: -self.radius - 1 + i,
                      self.radius + j: -self.radius - 1 + j]
                distances.append(tmp.clone())
            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance = (distance**2).view(-1, self.indim, h_dis, w_dis).sum(dim=1).view(batch_size, -1, h_dis, w_dis)
            distance = self.all_one_conv_indim_wise(distance)
            distance = torch.exp(
                -distance / distance.mean() / 2 / self.band_width ** 2)  # using mean of distance to normalize
            prob = (self.all_one_conv_radius_wise(distance) / self.patch_sampling_num) ** (1 / self.temperature)

            if self.if_pool:
                prob = -self.pool(-prob)  # min pooling of probability
            prob /= prob.sum(dim=(-2, -1), keepdim=True)


            batch_size, channels, h, w = x.shape

            random_choice = random_sample(prob, sampling_num=int(self.drop_rate * h * w))

            random_mask = torch.ones((batch_size * channels, h * w), device='cuda:0')
            random_mask[torch.arange(batch_size * channels, device='cuda:0').view(-1, 1), random_choice] = 0

        return x * random_mask.view(x.shape)


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1, if_dropout = False):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.if_dropout = if_dropout
        self.if_pool = pool
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        # if if_dropout:
        #     self.info_dropout = Info_Dropout(indim, outdim, 3, padding=padding)
        #     self.parametrized_layers = [self.C, self.BN, self.relu, self.info_dropout]
        # else:
        #     self.parametrized_layers = [self.C, self.BN, self.relu]

        self.parametrized_layers = [self.C, self.BN, self.relu]

        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        if if_dropout:
            self.info_dropout = Info_Dropout(indim, outdim, 3, padding=padding, if_pool=pool)
            # self.info_dropout = nn.Dropout(0.1)
            self.parametrized_layers.append(self.info_dropout)

        # self.trunk = nn.Sequential(*self.parametrized_layers)
        self.trunk = nn.ModuleList(self.parametrized_layers)

        # build a all-ones conv_kernel for computing norm of x
        self.all_one_conv = nn.Conv2d(indim, outdim, 3, padding=padding, bias=False)
        self.all_one_conv.weight.data = torch.ones_like(self.C.weight, dtype=torch.float)
        self.all_one_conv.weight.requires_grad = False

    def forward(self, x, return_activation_l1=False):
        # out = self.trunk(x)
        # return out
        x_old = x.clone()
        for i, module in enumerate(self.trunk):
            if self.if_dropout: # Info_Dropout Layer
                if self.if_pool and i == 4:
                    x = module(x_old, x)
                elif (not self.if_pool) and i == 3:
                    x = module(x_old, x)
                else:
                    x = module(x)
            else:
                x = module(x)
            # x = module(x)

        if return_activation_l1:
            return x, 0
        else:
            return x


# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res, if_dropout=False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.if_dropout = if_dropout
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        if if_dropout:
            self.info_dropout1 = Info_Dropout(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1)
            self.info_dropout2 = Info_Dropout(outdim, outdim,kernel_size=3, padding=1)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x, return_activation_l1=False):
        x_old = x.clone()
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        if self.if_dropout:
            out = self.info_dropout1(x_old, out)

        x_old = out.clone()
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        if self.if_dropout:
            out = self.info_dropout2(x_old, out)

        if return_activation_l1:
            return out, 0
        else:
            return out


# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True, pool=True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ), if_dropout=(i < dropout_layers)) #only pooling for first 4 layers
            trunk.append(B)

        # trunk.append(nn.AvgPool2d(5))
        if flatten:
            trunk.append(Flatten())

        # self.trunk = nn.Sequential(*trunk)
        self.trunk = nn.ModuleList(trunk)
        self.final_feat_dim = 1600
        # if pool:
        #     self.final_feat_dim = 64
        # else:
        #     self.final_feat_dim = 1600

    def forward(self, x, return_activation_l1=False):
        for module in self.trunk:
            x = module(x)
        if return_activation_l1:
            return x, 0
        else:
            return x

class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1, if_dropout=(i < dropout_layers)  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

    def forward(self,x, return_activation_l1=False):
        out = self.trunk(x)
        if return_activation_l1:
            return out, 0
        else:
            return out

class ConvNetS(nn.Module): #For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten = True):
        super(ConvNetS,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ConvNetSNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,5,5]

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ResNet(nn.Module):
    maml = False #Default
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            self.conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            self.bn1 = BatchNorm2d_fw(64)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if dropout_layers > 0:
            self.info_dropout = Info_Dropout(3, 64, kernel_size=7, stride=2, padding=3, if_pool=True,
                                             pool_kernel_size=3, pool_stride=2, pool_padding=1)

        init_layer(self.conv1)
        init_layer(self.bn1)


        trunk = []
        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x, return_activation_l1=False):
        x_old = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        if dropout_layers > 0:
            x = self.info_dropout(x_old, x)
        out = self.trunk(x)

        if return_activation_l1:
            return out, 0
        else:
            return out

def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)

def Conv4NP():
    return ConvNetNopool(4)

def Conv6NP():
    return ConvNetNopool(6)

def Conv4S():
    return ConvNetS(4)

def Conv4SNP():
    return ConvNetSNopool(4)

def ResNet10( flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34( flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50( flatten = True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101( flatten = True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)




