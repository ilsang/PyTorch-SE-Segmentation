import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)
