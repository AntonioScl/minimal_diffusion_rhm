import argparse
import torch
import init
from train import train


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train DDPM on the Random Hierarchy Model"
    )

    parser.add_argument("--device", type=str, default="cuda")

    ### Process
    parser.add_argument(
        "--process",
        type=str,
        default="discrete",
        help="Process to use (implemented: continuous, discrete)",
    )
    parser.add_argument(
        "--nT",
        type=int,
        default=200,
        help="Number of time steps for the diffusion process",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.001, help="beta1 for the diffusion process"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.1, help="beta2 for the diffusion process"
    )

    ### Dataset parameters
    parser.add_argument(
        "--dataset", type=str, default="rhm", help="Dataset to reproduce"
    )
    parser.add_argument(
        "--num_features", metavar="v", type=int, help="number of features"
    )
    parser.add_argument(
        "--num_classes", metavar="n", type=int, help="number of classes", default=None
    )
    parser.add_argument(
        "--num_synonyms",
        metavar="m",
        type=int,
        help="multiplicity of low-level representations",
    )
    parser.add_argument(
        "--tuple_size", metavar="s", type=int, help="size of low-level representations"
    )
    parser.add_argument("--num_layers", metavar="L", type=int, help="number of layers")
    parser.add_argument("--seed_rules", type=int, help="seed for the dataset")
    parser.add_argument(
        "--train_size", metavar="Ptr", type=int, help="training set size"
    )
    parser.add_argument("--batch_size", metavar="B", type=int, help="batch size")
    parser.add_argument(
        "--test_size", metavar="Pte", default=256, type=int, help="test set size"
    )
    parser.add_argument(
        "--generate_all",
        default=False,
        action="store_true",
        help="generate all the dataset",
    )
    parser.add_argument(
        "--seed_sample", type=int, help="seed for the sampling of train and testset"
    )
    parser.add_argument(
        "--replacement",
        default=False,
        action="store_true",
        help="sample with replacement for the rhm dataset",
    )
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--whitening", type=int, default=0)

    """
    Architecture args
    """
    parser.add_argument(
        "--model", type=str, default='bpUnet', help="architecture (implemented: bpUnet)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="start",
        help="If architecture predicts noise or the starting point",
    )
    parser.add_argument("--model_output", type=str, default="logits")
    parser.add_argument("--depth", type=int, default=None, help="depth of the network")
    parser.add_argument("--width", type=int, default=1024, help="width of the network")
    parser.add_argument("--filter_size", type=int, default=None)
    parser.add_argument(
        "--num_heads", type=int, help="number of heads (transformer Unet only)"
    )
    parser.add_argument(
        "--embedding_dim", type=int, help="embedding dimension (transformer only)"
    )
    parser.add_argument("--bias", default=False, action="store_true")
    parser.add_argument("--seed_model", type=int, default=2, help="seed for model initialization")

    """
        Training args
    """
    parser.add_argument("--optim", type=str, default="adam", help="optimizer to use [adam, sgd]")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--accumulation", default=False, action="store_true")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--scheduler_time", type=int, default=None)
    parser.add_argument("--n_epoch", type=int, default=10000)
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=1,
        help="number of forward process trajectories used in the loss",
    )

    """
	Output args
    """
    parser.add_argument("--print_period", type=int, help="period of prints, linearly spaced.", default=1000)
    parser.add_argument("--save_freq", type=int, help="frequency of saves, logarithmically spaced.", default=1)
    parser.add_argument("--loss_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--output", type=str, required=True, help="path of the output file"
    )

    args = parser.parse_args()
    if args.num_classes is None:
        args.num_classes = args.num_features
    if args.depth is None:
        args.depth = args.num_layers
    if args.filter_size is None:
        args.filter_size = args.tuple_size
    if args.generate_all:
        Pmax = args.num_classes * args.num_synonyms ** (
            (args.tuple_size**args.num_layers - 1) // (args.tuple_size - 1)
        )
        if Pmax > 1e8:
            print("Pmax is too large, impossible to generate all the dataset")
            args.generate_all = False
        else:
            args.test_size = Pmax - args.train_size
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU")
        args.device = "cpu"

    return args


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    args = parse_args() # Parse arguments

    train_loader, test_loader, rules, dataset = init.init_data(args) # Initialize data
    ddpm = init.init_model(args)                       # Initialize diffusion model and neural network
    ddpm.to(args.device)
    optim_sched = init.init_optimizer(ddpm, args)      # Initialize optimizer and scheduler
    bp = init.init_bp(args, rules)                     # Initialize belief propagation if needed
    eval_func = init.init_eval_func(bp, args, dataset) # Initialize evaluation functions to test the model

    # Training loop
    for data in train(train_loader, test_loader, ddpm, optim_sched, args, eval_func=eval_func):
        pass
