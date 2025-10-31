import torch
import torch.nn as nn
from datasets.BP_utils import BP_countcorrect_upward

import numpy as np


def check_rules(samples, bp):
    """
    Check if the generated data is consistent with the rules.

    Args:
        samples (torch.Tensor): generated samples in one-hot encoding. Shape (B, v, d)
        bp (BP object): belief propagation object

    Returns:
        frac_correct (float): fraction of samples that are consistent with the rules.
        frac_correct (dict): fraction of samples that are consistent with the rules for every layer.
    """

    x = torch.argmax(samples, dim=1) # B, d
    frac_correct, rules_frequencies = BP_countcorrect_upward(x, bp)

    frac_valid = frac_correct[0].item()
    frac_valid_per_layer = {bp.L-key : frac_correct[key].cpu() for key in frac_correct.keys()} # Count the layers starting from the leaves (layer 0)
    frac_valid_per_layer = dict(sorted(frac_valid_per_layer.items()))
    rules_frequencies_per_layer = {bp.L-key : rules_frequencies[key].cpu() for key in rules_frequencies.keys()} # Count the layers starting from the leaves (layer 0)
    rules_frequencies_per_layer = dict(sorted(rules_frequencies_per_layer.items()))

    return frac_valid, frac_valid_per_layer, rules_frequencies_per_layer


def compute_d3pm_loss_per_time(n_windows, points_per_window, model, x0):

    with torch.no_grad():
        model.eval()

        time_losses = {}
        for time_window in range(n_windows):
            n_trajectories = points_per_window
            v, d = x0.shape[1], x0.shape[2]
            x = (
                x0.unsqueeze(1)
                .repeat(1, n_trajectories, 1, 1)
                .view(-1, v, d)
                .to(x0.device)
            )
            B = x.shape[0]
            _ts = torch.randint(
                model.n_T // n_windows * time_window,
                model.n_T // n_windows * (time_window + 1),
                (B,),
            ).to(x.device)

            proba = model.alphabar_t[_ts, None, None] * x + (
                1 - model.alphabar_t[_ts, None, None]
            ) / v * torch.ones_like(x)
            proba = proba.permute(0, 2, 1).reshape(-1, v)

            x_t = torch.multinomial(proba, num_samples=1)
            x_t = nn.functional.one_hot(x_t, v)
            x_t = x_t.reshape(-1, d, v).permute(0, 2, 1)

            if model.model_type == "start":
                true_p = model.proba_posterior_t_1(x_t, _ts, x) # B, v, d
                true_p = true_p.permute(0, 2, 1).reshape(-1, true_p.shape[1]) # B*d, v
                true_p = true_p / true_p.sum(1, keepdim=True)
                model_p = model.proba_posterior_t_1(
                    x_t, _ts, model.readout(model.model(x_t, _ts / model.n_T))
                ) # B, v, d
                model_p = model_p.permute(0, 2, 1).reshape(-1, model_p.shape[1]) # B*d, v
                model_p = model_p / model_p.sum(1, keepdim=True)
                log_model_p = torch.log(model_p + 1e-8)
                log_true_p = torch.log(true_p + 1e-8)
                t_loss = (true_p * (log_true_p - log_model_p)).sum() / model_p.shape[0]
                time_losses[time_window] = t_loss.item()

            else:
                raise NotImplementedError

        return time_losses


def compare_with_trainset(trainset, samples):
    """
    Check if the generated data is consistent with the rules.

    Args:
        trainset (torch.Tensor): dataset used for training. Shape (P, v, d)
        samples (torch.Tensor): generated samples. Shape (B, v, d)
        
    Returns:
        frac_copies (float): fraction of samples that are copies of the training set.

    """
    # samples = torch.argmax(samples, dim=1).reshape(samples.shape[0], -1)
    # dataset = dataset.argmax(dim=1).reshape(dataset.shape[0], -1)
    # hamming = (samples != dataset).float().mean()

    d = trainset.shape[2]

    # print(samples.argmax(1)[:10])

    samples = samples.reshape(samples.shape[0], -1) # B, v*d
    trainset = trainset.reshape(trainset.shape[0], -1) # P, v*d

    frac_copies = (samples @ trainset.T == d).sum() / samples.shape[0]

    return frac_copies.item()