
import os
import sys
import numpy as np
import einops
from typing import Union, Optional, Tuple, List, Dict
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
from PIL import Image
import json

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
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
        pass

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


tests.test_flatten(Flatten)

