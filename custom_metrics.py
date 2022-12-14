import torch
from xautodl.utils import obtain_accuracy

def valid_acc_metric(network, inputs, targets):
    with torch.no_grad():
        network.eval()
        _, logits = network(inputs)
        val_top1, val_top5 = obtain_accuracy(
            logits.data, targets.data, topk=(1, 5)
        )
    return val_top1.item()