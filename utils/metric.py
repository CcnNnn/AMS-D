import numpy as np
from sklearn import metrics

__all__ = ['calculate_stats']


def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    target_max = np.argmax(target, 1)
    output_max = np.argmax(output, 1)
    
    # 4-class accuracy
    acc = metrics.accuracy_score(target_max, output_max)
    
    num_neg = len(target_max[target_max==0])
    num_pos = len(target_max) - num_neg
    
    A_multiple = target_max * output_max
    tp = sum([1 for element in A_multiple if element!=0])
    
    B_multiple = (target_max + 1) * (output_max + 1)
    tn = sum([1 for element in B_multiple if element==1])
        
    tpr = tp / num_pos if num_pos != 0 else 0  #sensitivity 
    tnr = tn / num_neg if num_neg != 0 else 0 #specificity

    metric_dict = {
            'tpr': tpr,       #sensitivity
            'tnr': tnr,       #specificity
            'acc': acc        # 4-class accuracy
            }

    return metric_dict
