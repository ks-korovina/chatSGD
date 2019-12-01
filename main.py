"""
Runner script.
Workers communicate quantized gradients with specified level of quantization after every batch round.

TODO:
* add GPU training (transfers to device)
"""

from datasets import get_data_loaders_per_machine
from models import get_model
from coding import uniform_reconstruct

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import argparse
import ast
import math
from copy import deepcopy

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CPU = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="lenet", type=str, help='model to use')
    parser.add_argument('--dataset', default="mnist", type=str, help='dataset to use')

    # Key experimental parameters
    parser.add_argument('--n_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--quant_levels', default="10000", type=str,
        help='number of quantization levels, either an int or list[int] in str format; log2(2*s)=num of bits')

    # Training settings
    parser.add_argument('--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--n_epochs', default=15, type=int, help='number of training epochs')

    # Evaluation settings
    parser.add_argument('--eval_freq', default=1, type=int, help='evaluate every `x` epochs')
    parser.add_argument('--verbose', default=False, type=bool, help='whether to display training statistics')

    args = parser.parse_args()
    return args


def add_dicts(d, add_d, weight=1.0):
    for key in add_d:
        if isinstance(d[key], torch.LongTensor):
            continue
        d[key] += weight * add_d[key]
    return d


def aggregate_gradients(all_grads, s, agg_args):
    agg_grads = {k: 0.0 for k in all_grads[0].keys()}
    
    if agg_args["mode"] == "simple":
        n_active_workers = len([g for g in all_grads if g is not None])
        for m, grad in enumerate(all_grads):
            quantized_grad = uniform_reconstruct(grad, s if isinstance(s, int) else s[m])
            agg_grads = add_dicts(agg_grads, quantized_grad, weight=1/n_active_workers)
        return agg_grads
    
    elif agg_args["mode"] == "grouped":
        s_prime = agg_args["in_group_quant"]
        assert isinstance(s, int), "Outer quantization level must be an integer"
        assert s_prime >= s, "In-group allowed quantization level should be higher than out-group"
        n_workers = len(all_grads)
        group_size = math.ceil(len(all_grads) / agg_args["n_groups"])  # groups are chosen greedily
        for leader in range(0, n_workers, group_size):  # iterate over groups
            agg_grads_group = {k: 0.0 for k in all_grads[0].keys()}
            for m in range(leader, leader+group_size):
                if m >= n_workers:
                    break
                # for each machine in group, quantize a little bit (simulate sending to leader):
                quantized_grad = uniform_reconstruct(all_grads[m], s_prime)
                agg_grads_group = add_dicts(agg_grads_group, quantized_grad, weight=1/group_size)
            # Simulate sending from the leader to the master node:
            agg_grads_group = uniform_reconstruct(agg_grads_group, s)
            agg_grads = add_dicts(agg_grads, agg_grads_group, weight=1/agg_args["n_groups"])
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
        average per-batch loss

    Particular settings:

    Experiment 1. Use agg_args["mode"] = "grouped"

    Experiment 2. Use different values in list s, leave default agg_args

    NOTE: if we update once an epoch, it is actually too slow to converge -- doesn't work!

    """
    assert sync_mode in ("sync", "async")
    n_workers = len(data_loaders)
    model.train()

    # If async, get all gradients and then make an update
    if sync_mode == "sync":
        all_losses = []
        iterators = [iter(data_loader) for data_loader in data_loaders]
        done = [False] * len(iterators)
        while not all(done):
            all_grads = []  # PER-BATCH gradients from m workers
            for m_id, data_loader_it in enumerate(iterators):
                if not done[m_id]:
                    # Try to fetch the next batch on this machine:
                    try:
                        xs, ys = next(data_loader_it)
                        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
                    except StopIteration:
                        done[m_id] = True
                if done[m_id]:  # either previously was done, or set just now
                    # append empty message from that node
                    zero_message = {k: torch.tensor(0.0, device=DEVICE) for k in model.state_dict().keys()}
                    all_grads.append(zero_message)
                    # don't update all_loss to not screw up averaging
                    continue
                logits = model(xs)
                loss = nn.CrossEntropyLoss()(logits, ys)
                model.zero_grad()
                loss.backward()
                gradient = model.get_gradients()
                all_grads.append(gradient)
                all_losses.append(loss.item())
                # print(loss.item())

                ############ for per-epoch update ############
                # gradients_m = {k: 0.0 for k in model.state_dict().keys()}
                # [... looping over all (xs,ys) batches of local data and putting gradients into gradients_m ...]
                # grads = model.get_gradients() 
                # gradients_m = add_dicts(gradients_m, grads)
                # all_losses.append(loss.item())
                # # now gradients_m has gradients computed on machine m
                # all_grads.append(gradients_m)
                ###############################################

            # In synchronous regime, apply the update after all gradients are computed (*after every round of batches*)
            new_state_dict = deepcopy(model.state_dict())
            agg_grad = aggregate_gradients(all_grads, s, agg_args)
            new_state_dict = add_dicts(new_state_dict, agg_grad, weight=-lr)  # step against the gradient
            model.set_weights(new_state_dict)  # everything should live on the same device
        return np.mean(all_losses)  # per batch

    # Otherwise, make an update after every gradient computation
    elif sync_mode == "async":
        # TODO
        pass

    else:
        raise ValueError(f"Invalid argument sync_mode={sync_mode}")


def evaluate(model, data_loader):
    model.eval()
    accs = []
    for (xs, ys) in data_loader:
        logits = model(xs)
        y_pred = logits.argmax(dim=1)
        batch_acc = (y_pred == ys).float().mean().item()
        accs.append(batch_acc)
    print("Validation accuracy: {:.3f}".format(np.mean(accs)))


if __name__ == "__main__":
    args = parse_args()

    train_data_loaders = get_data_loaders_per_machine(args.dataset, "train", args.n_workers, args.batch_size)
    eval_data_loaders = get_data_loaders_per_machine(args.dataset, "val", args.n_workers, args.batch_size)
    test_data_loaders = get_data_loaders_per_machine(args.dataset, "test", args.n_workers, args.batch_size)

    model = get_model(args.model)
    model.to(DEVICE)

    # Parse quant levels
    s = ast.literal_eval(args.quant_levels)  # int or list[int]
    # TODO: add args parameters for agg_args instead of this:
    # agg_args_grouped = {
    #     "mode": "grouped",
    #     "n_groups": 3,
    #     "in_group_quant": s * 2000
    # }

    curr_lr = args.lr  # TODO: this may need tuning and lr decay

    for epoch in range(args.n_epochs):
        loss = train_epoch(model, train_data_loaders, curr_lr, s)  #, agg_args=agg_args_grouped)
        print(f"Loss: {loss}")
        if epoch % args.eval_freq == 0:
            model.eval()
            evaluate(model, eval_data_loaders)

    # test
    evaluate(model, test_data_loaders)
