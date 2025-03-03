import torch

def accuracy(output, target):
    # I can compute the top-1 accuracy here.
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred)).sum()
        acc = correct.float() / target.size(0)
        return acc
