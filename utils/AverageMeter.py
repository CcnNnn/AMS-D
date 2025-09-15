# Copyright (c) 2021-present LG CNS Corp.
# original code from:  
# "Self-Knowledge Distillation with Progressive Refinement of Targets"
# Kyungyul Kim, ByeongMoon Ji, Doyoung Yoon, Sangheum Hwang, ICCV 2021
# GitHub: https://github.com/lgcnsai/PS-KD-Pytorch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    