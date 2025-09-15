# author: Nuo Chen (2025)

import torch
import torch.nn as nn
from torch.nn import functional as F


class Custom_Loss(nn.Module):
    def __init__(self, distillation_type, tau=1.0, alpha=1.0, beta=3e-2, gamma=3e-1):
        super(Custom_Loss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.kldivloss  = nn.KLDivLoss(reduction='batchmean')
        self.distillation_type = distillation_type
        self.tau = tau
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
    
    
    def forward(self, s_output, t_output, weighted_output, targets, mse):
        """
        Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
        """
        ### task loss
        log_probs = self.logsoftmax(s_output)
        loss = (-targets * log_probs).mean(0).sum()
        
        ### token weights loss
        if weighted_output is not None:
            weight_log_probs = self.logsoftmax(weighted_output)
            weight_loss = (-targets * weight_log_probs).mean(0).sum()
        else:
            weight_loss = 0
        
        ### self-disillation loss
        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = self.kldivloss(F.log_softmax(s_output/T, dim=1), 
                                               F.softmax(t_output/T, dim=1)) * (T * T)
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(s_output, t_output.argmax(dim=1))
        
        pskd_loss = loss + distillation_loss * self.alpha + weight_loss * self.beta + mse * self.gamma
        
        return pskd_loss, loss, weight_loss, distillation_loss
        
         