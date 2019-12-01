"""
Models that support quantized gradients
"""
import torch
import torch.nn as nn


def get_model(model_name, *args, **kwargs):
    """
    User interface to models.
    Use "vgg{11,13,16,19}"" for CIFAR10,
    and "lenet" for MNIST
    """
    if "vgg" in model_name.lower():
        return VGG(model_name.upper())
    elif model_name == "lenet":
        return LeNet(*args, **kwargs)
    else:
        raise ValueError("Unknown model {}".format(model_name))


class ModelBase(nn.Module):
    def get_gradients(self):
        """
        Returns dictionary of layer name -> gradient wrt x mappings.
        Should be called after .backward() call.
        """
        grad_dict = {}
        for k, v in self.named_parameters():
            if v.grad is not None:
                grad_dict[k] = v.grad
        return grad_dict

    def set_weights(self, weights):
        """
        Set weights of the model, given state dict
        """
        state_dict = self.state_dict()
        for k in weights:  # only update the weights that require_grad
            state_dict[k] = weights[k]
        self.load_state_dict(state_dict)


class LeNet(ModelBase):
    def __init__(self, in_channels=1, img_rows=28, num_classes=10):
        """ Parameters set for MNIST, change in constructor if used with other data """
        super(LeNet, self).__init__()
        self.model_name = 'LeNet'
        self.out_rows = ((img_rows - 4)//2 - 4)//2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 20, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 50, 5),
            nn.Dropout2d(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.out_rows * self.out_rows * 50, 500),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(500, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.out_rows*self.out_rows*50)
        x = self.classifier(x)
        return x


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(ModelBase):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

