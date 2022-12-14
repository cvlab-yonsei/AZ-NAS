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

def acc_confidence_robustness_metrics(network, inputs, targets):
    with torch.no_grad():
        network.eval()
        # accuracy
        _, logits = network(inputs)
        val_top1, val_top5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
        acc = val_top1

        # confidence
        prob = torch.nn.functional.softmax(logits, dim=1)
        one_hot_idx = torch.nn.functional.one_hot(targets)
        confidence = (prob[one_hot_idx==1].sum()) / inputs.size(0) * 100 # in percent
        
        # robustness
        _, noisy_logits = network(inputs + torch.randn_like(inputs)*0.1)
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        robustness = -kl_loss(torch.nn.functional.log_softmax(noisy_logits, dim=1), torch.nn.functional.softmax(logits, dim=1))
        
        return acc.item(), confidence.item(), robustness.item()