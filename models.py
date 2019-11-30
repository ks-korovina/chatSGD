"""
Models that support quantized gradients
"""
import torch
import torch.nn as nn


class QuantizedModelBase(nn.Module):
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


class QuantizedLeNet(QuantizedModelBase):
    def __init__(self, in_channels=1, img_rows=28, num_classes=10):
        """ Parameters set for MNIST, change in constructor if used with other data """
        super(QuantizedLeNet, self).__init__()
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
