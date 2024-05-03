import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
from torch import Tensor, optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional
from jaxtyping import Float
from dataclasses import dataclass, replace
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display, HTML
import wandb

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part3_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer
from part2_cnns.solutions import IMAGENET_TRANSFORM, ResNet34
from part2_cnns.solutions_bonus import get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
import part3_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"


def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


# plot_fn(pathological_curve_loss)


def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    opt = t.optim.SGD([xy,], lr, momentum)
    rets = t.empty((n_iters, 2))
    for i in range(n_iters):
        rets[i] = (xy.detach())
        out = fn(*xy)
        out.backward()
        opt.step()
        opt.zero_grad()
    return rets


points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])

    points.append((xys, optimizer_class, params))

# plot_fn_with_points(pathological_curve_loss, points=points)



class SGD:
    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float = 0.0, 
        weight_decay: float = 0.0
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        self.params = list(params) # turn params into a list (because it might be a generator)

        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay

        self.grads = [t.zeros_like(param.data) for param in self.params]

    def zero_grad(self) -> None:
        '''Zeros all gradients of the parameters in `self.params`.
        '''
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        '''Performs a single optimization step of the SGD algorithm.
        '''
        for param, grad in zip(self.params, self.grads):
            if self.lmda:
                param_grad = param.grad + self.lmda * param.data
            else:
                param_grad = param.grad
            grad *= self.mu
            grad += param_grad
            param.data -= self.lr * grad

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


tests.test_sgd(SGD)

class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        '''
        self.params = list(params) # turn params into a list (because it might be a generator)

        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay
        self.eps = eps
        self.alpha = alpha

        self.grads = [t.zeros_like(param.data) for param in self.params]
        self.square_averages = [t.zeros_like(param.data) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param, grad, square_average in zip(self.params, self.grads, self.square_averages):
            param_grad = param.grad + self.lmda * param.data

            square_average *= self.alpha
            square_average += (1 - self.alpha) * (param_grad * param_grad)

            grad *= self.mu
            grad += param_grad / (t.sqrt(square_average) + self.eps)

            param.data -= self.lr * grad

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"


tests.test_rmsprop(RMSprop)


class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        self.params = list(params) # turn params into a list (because it might be a generator)

        self.lr = lr
        (self.beta1, self.beta2) = betas
        self.lmda = weight_decay
        self.eps = eps

        self.first_moments = [t.zeros_like(param.data) for param in self.params]
        self.second_moments = [t.zeros_like(param.data) for param in self.params]
        self.ep_count = 0 

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.ep_count += 1
        for param, first_moment, second_moment in zip(self.params, self.first_moments, self.second_moments):
            param_grad = param.grad + self.lmda * param.data

            first_moment *= self.beta1
            first_moment += (1 - self.beta1) * param_grad
            first_moment_unbiased = first_moment / (1 - self.beta1 ** self.ep_count)

            second_moment *= self.beta2
            second_moment += (1 - self.beta2) * (param_grad * param_grad)
            second_moment_unbiased = second_moment / (1 - self.beta2 ** self.ep_count)

            param.data -= self.lr * first_moment_unbiased / (t.sqrt(second_moment_unbiased) + self.eps)

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adam(Adam)


class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        self.params = list(params) # turn params into a list (because it might be a generator)

        self.lr = lr
        (self.beta1, self.beta2) = betas
        self.lmda = weight_decay
        self.eps = eps

        self.first_moments = [t.zeros_like(param.data) for param in self.params]
        self.second_moments = [t.zeros_like(param.data) for param in self.params]
        self.ep_count = 0 

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        self.ep_count += 1
        for param, first_moment, second_moment in zip(self.params, self.first_moments, self.second_moments):
            param.data -= self.lmda * self.lr * param.data
            first_moment *= self.beta1
            first_moment += (1 - self.beta1) * param.grad
            first_moment_unbiased = first_moment / (1 - self.beta1 ** self.ep_count)

            second_moment *= self.beta2
            second_moment += (1 - self.beta2) * (param.grad * param.grad)
            second_moment_unbiased = second_moment / (1 - self.beta2 ** self.ep_count)

            param.data -= self.lr * first_moment_unbiased / (t.sqrt(second_moment_unbiased) + self.eps)

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adamw(AdamW)


def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    opt = optimizer_class([xy,], **optimizer_hyperparams)
    rets = t.empty((n_iters, 2))
    for i in range(n_iters):
        rets[i] = (xy.detach())
        out = fn(*xy)
        out.backward()
        opt.step()
        opt.zero_grad()
    return rets

points = []

optimizer_list = [
    (SGD, {"lr": 0.03, "momentum": 0.99}),
    (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
    (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn(pathological_curve_loss, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
    points.append((xys, optimizer_class, params))

# plot_fn_with_points(pathological_curve_loss, points=points)


def get_cifar(subset: int = 1):
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)
    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
    return cifar_trainset, cifar_testset


cifar_trainset, cifar_testset = get_cifar()

# imshow(
#     cifar_trainset.data[:15],
#     facet_col=0,
#     facet_col_wrap=5,
#     facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
#     title="CIFAR-10 images",
#     height=600
# )

@dataclass
class ResNetTrainingArgs():
    batch_size: int = 64
    epochs: int = 3
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10


class ResNetTrainer:
    def __init__(self, args: ResNetTrainingArgs):
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = t.optim.Adam(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        self.logged_variables = {"loss": [], "accuracy": []}

    def to_device(self, *args):
        return [x.to(device) for x in args]

    def training_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
        imgs, labels = self.to_device(imgs, labels)
        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @t.inference_mode()
    def validation_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
        imgs, labels = self.to_device(imgs, labels)
        logits = self.model(imgs)
        return (logits.argmax(dim=1) == labels).sum()

    def train(self):

        for epoch in range(self.args.epochs):

            # Load data
            train_dataloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
            val_dataloader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
            progress_bar = tqdm(total=len(train_dataloader))

            # Training loop (includes updating progress bar, and logging loss)
            self.model.train()
            for imgs, labels in train_dataloader:
                loss = self.training_step(imgs, labels)
                self.logged_variables["loss"].append(loss.item())
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}")

            # Compute accuracy by summing n_correct over all batches, and dividing by number of items
            self.model.eval()
            accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in val_dataloader) / len(self.testset)

            # Update progress bar description to include accuracy, and log accuracy
            progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}")
            self.logged_variables["accuracy"].append(accuracy.item())

# args = ResNetTrainingArgs()
# trainer = ResNetTrainer(args)
# trainer.train()

# plot_train_loss_and_test_accuracy_from_trainer(trainer, title="Feature extraction with ResNet34")

def test_resnet_on_random_input(model: ResNet34, n_inputs: int = 3):
    indices = np.random.choice(len(cifar_trainset), n_inputs).tolist()
    classes = [cifar_trainset.classes[cifar_trainset.targets[i]] for i in indices]
    imgs = cifar_trainset.data[indices]
    device = next(model.parameters()).device
    with t.inference_mode():
        x = t.stack(list(map(IMAGENET_TRANSFORM, imgs)))
        logits: t.Tensor = model(x.to(device))
    probs = logits.softmax(-1)
    if probs.ndim == 1: probs = probs.unsqueeze(0)
    for img, label, prob in zip(imgs, classes, probs):
        display(HTML(f"<h2>Classification probabilities (true class = {label})</h2>"))
        imshow(
            img, 
            width=200, height=200, margin=0,
            xaxis_visible=False, yaxis_visible=False
        )
        bar(
            prob,
            x=cifar_trainset.classes,
            template="ggplot2", width=600, height=400,
            labels={"x": "Classification", "y": "Probability"}, 
            text_auto='.2f', showlegend=False,
        )


# test_resnet_on_random_input(trainer.model)

@dataclass
class ResNetTrainingArgsWandb(ResNetTrainingArgs):
    wandb_project: Optional[str] = 'day3-resnet'
    wandb_name: Optional[str] = None


class ResNetTrainerWandb(ResNetTrainer):
    def __init__(self, args: ResNetTrainingArgsWandb):
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)

        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        wandb.watch(self.model.out_layers[-1], log="all", log_freq = 50)
        self.optimizer = t.optim.Adam(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        self.logged_variables = {"loss": [], "accuracy": []}

    def train(self):
        try:
            step = 0
            for epoch in range(self.args.epochs):

                # Load data
                train_dataloader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
                val_dataloader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)
                progress_bar = tqdm(total=len(train_dataloader))

                # Training loop (includes updating progress bar, and logging loss)
                self.model.train()
                for imgs, labels in train_dataloader:
                    step += 1
                    loss = self.training_step(imgs, labels)
                    progress_bar.update()
                    progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}")
                    wandb.log({"loss": loss.item()}, step)

                # Compute accuracy by summing n_correct over all batches, and dividing by number of items
                self.model.eval()
                accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in val_dataloader) / len(self.testset)

                # Update progress bar description to include accuracy, and log accuracy
                progress_bar.set_description(f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}")

                wandb.log({"accuracy": accuracy.item()}, step)
        finally:
            wandb.finish()

# args = ResNetTrainingArgsWandb()
# trainer = ResNetTrainerWandb(args)
# trainer.train()



sweep_config = {
    "name": "imgonnastartsweepin",
    "method": "random",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "learning_rate": {"min": 1e-4, "max": 1e-1, "distribution": "log_uniform_values"},
        "batch_size": {"values": [32, 64, 128, 256]},
        "epochs": {"values": [1, 2, 3]},
    },
}

tests.test_sweep_config(sweep_config)


# (2) Define a training function which takes no arguments, and uses `wandb.config` to get hyperparams

class ResNetTrainerWandbSweeps(ResNetTrainerWandb):
    '''
    New training class made specifically for hyperparameter sweeps, which overrides the values in
    `args` with those in `wandb.config` before defining model/optimizer/datasets.
    '''
    def __init__(self, args: ResNetTrainingArgsWandb):
        self.args = args
        # Initialize
        wandb.init(name=args.wandb_name)

        # Update args with the values in wandb.config
        self.args.batch_size = wandb.config["batch_size"]
        self.args.epochs = wandb.config["epochs"]
        self.args.learning_rate = wandb.config["learning_rate"]

        # Perform the previous steps (initialize model & other important objects)
        self.model = get_resnet_for_feature_extraction(self.args.n_classes).to(device)
        self.optimizer = t.optim.Adam(self.model.out_layers[-1].parameters(), lr=self.args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=self.args.subset)
        self.step = 0
        wandb.watch(self.model.out_layers[-1], log="all", log_freq=20)


def train():
    args = ResNetTrainingArgsWandb()
    trainer = ResNetTrainerWandbSweeps(args)
    trainer.train()


sweep_id = wandb.sweep(sweep=sweep_config, project='day3-resnet-sweep')
wandb.agent(sweep_id=sweep_id, function=train, count=3)
wandb.finish()

