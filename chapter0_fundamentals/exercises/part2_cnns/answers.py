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


# args = SimpleMLPTrainingArgs()
# train(args)




class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        range_ = 2 * np.sqrt(1 / (in_channels * kernel_size * kernel_size))

        self.weight = nn.Parameter((t.rand(self.out_channels, self.in_channels, kernel_size, kernel_size) - 0.5) * range_)


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d, which you can import.'''
        
        return t.nn.functional.conv2d(x, self.weight, stride = self.stride, padding = self.padding)


    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, stride={self.stride}"


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: Optional[int] = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of max_pool2d.'''
        return t.nn.functional.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        keys = ["kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])



tests.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")



class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, "num_features"]
    running_var: Float[Tensor, "num_features"]
    num_batches_tracked: Int[Tensor, ""] # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

        self.register_buffer("running_mean", t.zeros((num_features)))
        self.register_buffer("running_var", t.ones((num_features)))
        self.register_buffer("num_batches_tracked", t.tensor(0))
        
        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        if self.training:
            mean = t.mean(x, dim=(0, 2, 3), keepdim=True)
            var = t.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            self.num_batches_tracked += 1
        else:
            mean = einops.rearrange(self.running_mean, "n_channels -> 1 n_channels 1 1")
            var = einops.rearrange(self.running_var, "n_channels -> 1 n_channels 1 1")
        weight = einops.rearrange(self.weight, "channels -> 1 channels 1 1")
        bias = einops.rearrange(self.bias, "channels -> 1 channels 1 1")
        
        return ((x - mean) / t.sqrt(var + self.eps)) * weight + bias

    def extra_repr(self) -> str:
        keys = ["num_features", "eps", "momentum"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)


class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return einops.reduce(x, "batch channels height width -> batch channels", "mean")



class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = first_stride
        
        self.left_branch = nn.Sequential(
            Conv2d(in_feats, out_feats, 3, first_stride, padding = 1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, 3, padding = 1),
            BatchNorm2d(out_feats)
        )

        if first_stride > 1:
            self.right_branch = nn.Sequential(
                Conv2d(in_feats, out_feats, 1, first_stride),
                BatchNorm2d(out_feats)
            )
        else:
            self.right_branch = nn.Identity()
        self.add_module("last_relu", ReLU())

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)

        If no downsampling block is present, the addition should just add the left branch's output to the input.
        '''
        left = self.left_branch(x)
        right = self.right_branch(x)
        return self.last_relu(left + right)

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(in_feats, out_feats, first_stride),
            *[ResidualBlock(out_feats, out_feats) for _ in range(n_blocks - 1)]
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.blocks(x)

class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(3, 64, 7, 2, 3),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(3, 2),
            *[BlockGroup(n_blocks, in_feats, out_feats, first_stride) 
              for n_blocks, in_feats, out_feats, first_stride 
              in zip(n_blocks_per_group, [64] + out_features_per_group, out_features_per_group, first_strides_per_group)
              ],
            AveragePool(),
            Flatten(),
            Linear(out_features_per_group[-1], n_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        '''
        return self.model(x)


my_resnet = ResNet34()


def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    # Get the state dictionaries for each model, check they have the same number of parameters & buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

my_resnet = copy_weights(my_resnet, pretrained_resnet)


print_param_count(my_resnet, pretrained_resnet)

IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0)

assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)

def predict(model, images: t.Tensor) -> t.Tensor:
    '''
    Returns the predicted class for each image (as a 1D array of ints).
    '''
    with t.inference_mode():
        bois = model(images)
        return t.argmax(bois, dim=1)


with open(section_dir / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# Check your predictions match those of the pretrained model
my_predictions = predict(my_resnet, prepared_images)
pretrained_predictions = predict(pretrained_resnet, prepared_images)

print(my_predictions, pretrained_predictions)
assert all(my_predictions == pretrained_predictions)
print("All predictions match!")

# Print out your predictions, next to the corresponding images
for img, label in zip(images, my_predictions):
    print(f"Class {label}: {imagenet_labels[label]}")
    display(img)
    print()

