from functools import partial
import torch

import datasets
from diffusion.ddpm import DDPM, DiscreteDDPM
from diffusion.unet import bpUNET #, hUNET, hUNETTimeChan, hUNETFullEmb, bpUNET
from datasets.BP_utils import BpRhm
from diffusion.evaluate_model import (
    compute_d3pm_loss_per_time,
    check_rules,
    compare_with_trainset,
)


def init_data(args, trainloader_shuffle=True):
    """
    Initialise dataset.

    Returns:
        Two dataloaders for train and test set, rules, and features.
    """
    if args.dataset == "rhm":

        dataset = datasets.RandomHierarchyModel(
            num_features=args.num_features,  # vocabulary size
            num_synonyms=args.num_synonyms,  # features multiplicity
            num_layers=args.num_layers,  # number of layers
            num_classes=args.num_classes,  # number of classes
            tuple_size=args.tuple_size,  # number of branches of the tree
            seed_rules=args.seed_rules,
            train_size=args.train_size,
            test_size=args.test_size,
            seed_sample=args.seed_sample,
            input_format=args.input_format,
            whitening=args.whitening,
            replacement=args.replacement,
        )
        rules = dataset.get_rules()

    else:
        raise ValueError("dataset argument is invalid!")

    dataset.features, dataset.labels = dataset.features.to(
        args.device
    ), dataset.labels.to(
        args.device
    )  # move to device when using cuda

    trainset = torch.utils.data.Subset(dataset, range(args.train_size))
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=trainloader_shuffle, num_workers=0
    )

    if args.test_size:
        testset = torch.utils.data.Subset(
            dataset, range(args.train_size, args.train_size + args.test_size)
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1024, shuffle=False, num_workers=0
        )
    else:
        test_loader = None

    return train_loader, test_loader, rules, dataset.features


def init_model(args):
    """
    Initialise machine-learning model.
    """
    torch.manual_seed(args.seed_model)

    if args.model == "bpUnet":
        model = bpUNET(
            input_dim=args.tuple_size**args.num_layers,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            width=args.width,
            num_layers=args.depth,
            bias=args.bias,
            process=args.process,
        )
    else:
        raise ValueError("model argument is invalid!")

    model = model.to(args.device)

    if args.process == "discrete":
        ddpm = DiscreteDDPM(
            model=model,
            betas=(args.beta1, args.beta2),
            n_T=args.nT,
            model_type=args.model_type,
            model_output=args.model_output,
        )
    else:
        ddpm = DDPM(
            model=model,
            betas=(args.beta1, args.beta2),
            n_T=args.nT,
            model_type=args.model_type,
            model_output=args.model_output,
        )

    return ddpm


class NoOpScheduler:
    def step(self):
        pass


def init_optimizer(model, args):
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError(f"Invalid optimizer {args.optim}")

    if args.warmup_steps > 0:
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                return float(current_step) / float(max(1, args.warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = NoOpScheduler()

    return optimizer, scheduler


def init_eval_func(bp, args, dataset):
    return {
        "Train_losses_per_time": partial(compute_d3pm_loss_per_time, 10, 1),
        "Test_losses_per_time": partial(compute_d3pm_loss_per_time, 10, 1),
        "Fraction_of_copies": partial(
            compare_with_trainset, dataset[: args.train_size]
        ),
        "Valid_samples": partial(check_rules, bp=bp),
    }

def init_bp(args, rules):
    bp = BpRhm(
        args.num_features,
        args.tuple_size,
        args.num_synonyms,
        args.num_layers,
        rules,
        args.device,
    )
    return bp