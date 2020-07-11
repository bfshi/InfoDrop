import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Id(nn.Module):
    def __init__(self):
        super(Id, self).__init__()

    def forward(self, x):
        return x


def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

dropout_layers = 0

class Info_Dropout(nn.Module):  # slow version
    def __init__(self, indim, outdim, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, if_pool=False, pool_kernel_size=2, pool_stride=None,
                 pool_padding=0, pool_dilation=1):
        super(Info_Dropout, self).__init__()
        if groups != 1:
            raise ValueError('InfoDropout only supports groups=1')

        self.indim = indim
        self.outdim = outdim
        self.if_pool = if_pool
        self.drop_rate = 1.5
        self.temperature = 0.03
        self.band_width = 1.0
        self.radius = 3

        self.all_one_conv_indim_wise = nn.Conv2d(9, 9,
                                                 kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation,
                                                 groups=9, bias=False)
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise = nn.Conv2d(9, outdim, kernel_size=1, padding=0, bias=False)
        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


        if if_pool:
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding, pool_dilation)

        self.padder2 = nn.ConstantPad2d((padding + self.radius, padding + self.radius + 1,
                                         padding + self.radius, padding + self.radius + 1), 0)

    def initialize_parameters(self):
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


    def forward(self, x_old, x):
        # return self.random_prop_dropout(x)
        # return F.dropout(x, p=0.1, training=self.training)
        # return x

        with torch.no_grad():
            distances = []
            padded_x_old = self.padder2(x_old)
            sampled_i = torch.randint(-self.radius, self.radius + 1, size=(9,)).tolist()
            sampled_j = torch.randint(-self.radius, self.radius + 1, size=(9,)).tolist()
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
            prob = (self.all_one_conv_radius_wise(distance) / 9) ** (1 / self.temperature)

            if self.if_pool:
                prob = -self.pool(-prob)  # min pooling of probability
            prob /= prob.sum(dim=(-2, -1), keepdim=True)


            batch_size, channels, h, w = x.shape

            if not self.training:
                with torch.enable_grad():
                    return x * torch.exp(-self.drop_rate * prob * h * w)

            random_choice = random_sample(prob, sampling_num=int(self.drop_rate * h * w))

            random_mask = torch.ones((batch_size * channels, h * w), device='cuda:0')
            random_mask[torch.arange(batch_size * channels, device='cuda:0').view(-1, 1), random_choice] = 0

        return x * random_mask.view(x.shape)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=True, jigsaw_classes=30):
        super(AlexNet, self).__init__()

        self.feature_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
        )
        self.infodrop_1 = Info_Dropout(3, 64, kernel_size=11, stride=4, padding=2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.feature_2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.infodrop_2 = Info_Dropout(64, 192, kernel_size=5, padding=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.features = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout() if dropout else Id(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout() if dropout else Id(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.jigsaw_classifier = nn.Linear(256 * 6 * 6, jigsaw_classes)

    def is_patch_based(self):
        return False

    def forward(self, x, if_get_feature=False,):
        # for parameter in self.feature_1.parameters():
        #     print(parameter)

        x_old = x.clone()
        x = self.feature_1(x)
        if dropout_layers >= 1:
            x = self.infodrop_1(x_old, x)
        x = self.max_pool_1(x)

        x_old = x.clone()
        x = self.feature_2(x)
        if dropout_layers >= 2:
            x = self.infodrop_2(x_old, x)
        x = self.max_pool_2(x)

        x = self.features(x)

        if if_get_feature:
            return x
        else:
            return self.jigsaw_classifier(x.view(x.size(0), 256 * 6 * 6)),\
                   self.classifier(x.view(x.size(0), 256 * 6 * 6))


def alexnet(classes, pretrained=False, jigsaw_classes=30):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(classes, True, jigsaw_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    model.classifier[-1] = nn.Linear(4096, classes)
    nn.init.xavier_uniform_(model.classifier[-1].weight, .1)
    nn.init.constant_(model.classifier[-1].bias, 0.)
    return model
