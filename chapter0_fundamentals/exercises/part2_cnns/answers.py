import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple, Dict, List
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
import functools
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
from dataclasses import dataclass
from IPython.display import display
from PIL import Image
import json


# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent
assert exercises_dir.name == "exercises", f"This file should be run inside 'exercises/part2_cnns', not '{section_dir}'"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
import part2_cnns.tests as tests
from part2_cnns.utils import print_param_count

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")



class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        ret = x.clone()
        ret[x < 0] = 0
        return ret


tests.test_relu(ReLU)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        weights = t.empty((out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(nn.init.kaiming_uniform_(weights))
        if bias:
            self.bias = nn.Parameter(nn.init.uniform_(t.empty((out_features, ))) / np.sqrt(out_features))
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        if self.bias is not None:
            return einops.einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features") + self.bias
        else:
            return einops.einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"



tests.test_linear_forward(Linear)
tests.test_linear_parameters(Linear)
tests.test_linear_no_bias(Linear)


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        input_shape = input.shape
        end = self.end_dim + 1
        if end == 0:
            output_shape = input.shape[:self.start_dim] + tuple([np.prod(input.shape[self.start_dim:])])
        else:
            output_shape = input.shape[:self.start_dim] + tuple([np.prod(input.shape[self.start_dim:end])]) + input_shape[end:]
        return input.reshape(output_shape)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"

if MAIN:
    tests.test_flatten(Flatten)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dims = (28, 28)

        self.layers = nn.Sequential(Flatten(1, -1), Linear(28 * 28, 100), ReLU(), Linear(100, 10))

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)



tests.test_mlp(SimpleMLP)



MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset



@dataclass
class SimpleMLPTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = SimpleMLPTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    subset: int = 10


def train(args: SimpleMLPTrainingArgs):
    '''
    Trains the model, using training parameters from the `args` object.
    '''
    model = SimpleMLP().to(device)

    mnist_trainset, mnist_testset = get_mnist(subset=args.subset)
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []
    accuracy_list = []

    for epoch in tqdm(range(args.epochs)):
        # Training loop!
        for imgs, labels in mnist_trainloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())   
        # Val loop
        total = 0
        corr = 0
        with t.inference_mode():
            for imgs, labels in mnist_testloader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                _, preds = t.max(logits, dim=1)
                total += len(preds)
                corr += t.sum(labels == preds)
        accuracy_list.append(corr / total * 100)
        print(f"Epoch: {epoch}. Accuracy: {corr / total * 100}") 

    line(
        loss_list, 
        yaxis_range=[0, max(loss_list) + 0.1],
        labels={"x": "Num batches seen", "y": "Cross entropy loss"}, 
        title="SimpleMLP training on MNIST",
        width=700
    )
    line(
        accuracy_list, 
        yaxis_range=[0, 100],
        labels={"x": "Epoch", "y": "Accuracy"}, 
        title="SimpleMLP training on MNIST",
        width=700
    )


args = SimpleMLPTrainingArgs()
train(args)




# class Conv2d(nn.Module):
#     def __init__(
#         self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
#     ):
#         '''
#         Same as torch.nn.Conv2d with bias=False.

#         Name your weight field `self.weight` for compatibility with the PyTorch version.
#         '''
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#         self.m = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.stride, padding = self.padding)


#     def forward(self, x: t.Tensor) -> t.Tensor:
#         '''Apply the functional conv2d, which you can import.'''
#         return self.m(x)


#     def extra_repr(self) -> str:
#         return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, stride={self.stride}"


# tests.test_conv2d_module(Conv2d)
# m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
# print(f"Manually verify that this is an informative repr: {m}")

