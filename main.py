"""
Runner script
"""

from datasets import get_data_loaders_per_machine
from models import QuantizedLeNet
from coding import uniform_reconstruct

import torch
import torch.optim as optim
import torch.nn as nn

import argparse
import ast
import math
from copy import deepcopy


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model to use')
    parser.add_argument('--dataset', type=str, help='dataset to use')

    # Key experimental parameters
    parser.add_argument('--n_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--quant_levels', default="10", type=str, help='number of quantization levels, either an int or list[int] in str format')

    # Training settings
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--n_epochs', default=15, type=int, help='number of training epochs')

    # Evaluation settings
    parser.add_argument('--eval_freq', default=10, type=int, help='evaluate every `x` epochs')
    parser.add_argument('--verbose', default=False, type=bool, help='whether to display training statistics')
    
    args = parser.parse_args()
    return args


def add_dicts(d, add_d, weight=1.0):
    for key in add_d:
        d[key] += weight * add_d[k]
    return d


def aggregate_gradients(all_grads, s, agg_args):
    agg_grads = {k: 0.0 for k in all_grads[0].keys()}
    if agg_args["mode"] == "simple":
        for m, grad in enumerate(all_grads):
            quantized_grad = uniform_reconstruct(grad, s[m])
            agg_grads = add_dicts(agg_grads, quantized_grad)
        return agg_grads
    elif agg_args["mode"] == "grouped":
        s_prime = agg_args["in_group_quant"]
        assert isinstance(int, s), "Outer quantization level must be an integer"
        assert s_prime >= s, "In-group allowed quantization level should be higher than out-group"
        group_size = len(all_grads) // agg_args["n_groups"]
        for leader in range(0, n_workers, group_size):  # iterate over groups
            agg_grads_group = {k: 0.0 for k in all_grads[0].keys()}
            for m in range(leader, leader+group_size):
                # for each machine in group, quantize a little bit (simulate sending to leader):
                quantized_grad = uniform_reconstruct(all_grads[m], s_prime)
                agg_grads_group = add_dicts(agg_grads_group, quantized_grad)
            # Simulate sending from the leader to the master node:
            agg_grads_group = uniform_reconstruct(agg_grads_group, s)
            agg_grads = add_dicts(agg_grads, agg_grads_group)
        return agg_grads
    else:
        raise ValueError(f"Not implemented mode {agg_args['mode']}")


def train_epoch(model, data_loaders, lr, s, sync_mode="sync",
    agg_args={"mode": "simple", "n_groups": None, "in_group_quant": None}):
    """
    Function simulating distributed training for one epoch.

    Arguments:
        model
        data_loaders
        lr
        s {int or list[int]} -- quantization levels per machine
        sync_mode
    Returns:
        loss

    Particular settings:

    Experiment 1. Use agg_args["mode"] = "grouped"

    Experiment 2. Use different values in list s, leave default agg_args

    """
    assert sync_mode in ("sync", "async")
    n_workers = len(data_loaders)

    # If async, get all gradients and then make an update
    if sync_mode == "sync":
        all_grads = []
        for m_id, data_loader in enumerate(data_loaders):
            gradients_m = {k: 0.0 for k in new_state_dict.keys()}
            for xs, ys in data_loader:
                logits = model(xs)
                loss = nn.CrossEntropyLoss()(logits, ys)
                model.zero_grad()
                loss.backward()
                grads = model.get_gradients()
                gradients_m = add_dicts(gradients_m, grads)
            # now gradients_m has gradients computed on machine m
            all_grads.append(gradients_m)
        # In synchronous regime, apply the update after all gradients are computed
        new_state_dict = deepcopy(model.state_dict())
        agg_grad = aggregate_gradients(all_grads, s, agg_args)
        new_state_dict = add_dicts(new_state_dict, agg_grad, weight=-lr)
        model.set_weights(new_state_dict)

    # Otherwise, make an update after every gradient computation
    elif sync_mode == "async":
        # TODO
        pass
    return None


def evaluate(model, data_loaders):
    pass


if __name__ == "__main__":
    args = parse_args()

    train_data_loaders = get_data_loaders_per_machine(args.dataset, "train", args.n_workers, args.batch_size)
    eval_data_loaderst = get_data_loaders_per_machine(args.dataset, "val", args.n_workers, args.batch_size)
    test_data_loaderst = get_data_loaders_per_machine(args.dataset, "test", args.n_workers, args.batch_size)

    model = QuantizedLeNet()

    # Parse quant levels
    s = ast.literal_eval(args.quant_levels)  # int or list[int]
    # TODO: add args parameters for agg_args

    curr_lr = args.lr  # TODO: this may need tuning and lr decay

    for epoch in range(args.n_epochs):
        train_epoch(model, train_data_loaders, curr_lr, s)
        if epoch % args.eval_freq == 0:
            evaluate(model, eval_data_loaders)

    # test
    evaluate(model, test_data_loaders)
