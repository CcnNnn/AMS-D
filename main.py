# Copyright (c) 2021-present LG CNS Corp.
# original PS-KD code from:  
# "Self-Knowledge Distillation with Progressive Refinement of Targets"
# Kyungyul Kim, ByeongMoon Ji, Doyoung Yoon, Sangheum Hwang, ICCV 2021
# GitHub: https://github.com/lgcnsai/PS-KD-Pytorch

# =====================================================================
# further modified by Nuo Chen (2025)


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from .loader.myloader import my_dataloader
from .models.ams_d import AMSD
from .custom_loss import Custom_Loss

from .utils.AverageMeter import AverageMeter
from .utils.dir_maker import DirectroyMaker
from .utils.metric import calculate_stats
from .utils.color import Colorer
from .utils.etc import is_main_process, check_args, init_seed
from .utils.etc import save_on_master, paser_config_save, set_logging_defaults, save_data_pandas, save_data_jpg

import yaml
import os, logging
import argparse
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Progressive Self-Knowledge Distillation : PS-KD')
    
    ### AMS-D setteings
    parser.add_argument('--num-classes', type=int, default=4, help='number of classes')
    parser.add_argument('--model-size', type=str, default='base384', help='model size')
    parser.add_argument('--num-mel-bins', type=int, default=128, help='number of mel bins, frequency axis length of spectrograms')  
    parser.add_argument('--target-length', type=int, default=798, help='time axis length of spectrograms')  
    parser.add_argument('--fstride', type=int, default=16, help='patch stride in frequency axis')  
    parser.add_argument('--tstride', type=int, default=16, help='patch stride in time axis')     
    parser.add_argument('--fshape', type=int, default=16, help='patch size in frequency axis') 
    parser.add_argument('--tshape', type=int, default=16, help='patch size in time axis') 
    
    ## training settings
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay-rate', default=0.2, type=float, help='learning rate decay rate')
    parser.add_argument('--lr-decay-schedule', default=[15, 25, 30, 35, 40, 45], nargs='*', type=int, help='when to drop lr')
    parser.add_argument('--weight-decay', default=5e-7, type=float, help='weight decay')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch number')
    parser.add_argument('--end-epoch', default=50, type=int, help='number of training epoch to run')
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--experiments-dir', type=str, default=r'./AMS-D/log',help='directory name to save the model, log, config')
    parser.add_argument('--data-root', type=int, default=[r'./ICBHI/dataset', r'./ICBHI/dataset_aug'], help='dataset path')
    parser.add_argument('--data-type', type=str, default='ICBHI', help='type of dataset')
    parser.add_argument('--saveckp-freq', default=10, type=int, help='save checkpoint every x epochs')
    parser.add_argument('--num-workers', default=8, type=int, help='number of workers for dataloader')
    parser.add_argument('--gpu', type=int, default=0, help='gpu code')
    
    ### mask settings
    parser.add_argument('--mask-ratio', type=float, default=0.39, help='masking ratio')    
    parser.add_argument('--mask-dense', type=float, default=0, help='masking dense')    
    parser.add_argument('--mask-mode', type=str, default='attentive', help='mask strategy (attentive, time_inter, freq_inter, random)')    
    parser.add_argument('--alpha', type=float, default=1.0, help='weight of distillation loss')
    parser.add_argument('--beta', type=float, default=3e-2, help='weight of token-weight-module loss')  
    parser.add_argument('--gamma', type=float, default=3e-1, help='weight of reconstruction loss')
    parser.add_argument('--tau', type=float, default=1, help='temperature parameter of distillation loss')
    parser.add_argument('--resume', type=str, default=r'', help='load model path')
    
    args = parser.parse_args()
    return check_args(args)


def get_shape(fshape, tshape, fstride, tstride, input_fdim=128, input_tdim=1024):
    test_input = torch.randn(1, 1, input_fdim, input_tdim)
    test_proj = nn.Conv2d(1, 1, kernel_size=(fshape, tshape), stride=(fstride, tstride))
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return f_dim, t_dim    


#  Adjust_learning_rate & get_learning_rate  
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr

    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def get_learning_rate(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list += [param_group['lr']]
    return lr_list


#  Top-1 / Top -5 accuracy
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    
      
      
C = Colorer.instance()


def main():
    args = parse_args()
    config_file = r'./my_kd/config_train.yaml'
    with open(config_file, 'r') as file:
        audio_config = yaml.safe_load(file)

    print(C.green("[!] Start AMS-D."))
    
    dir_maker = DirectroyMaker(root=args.experiments_dir, save_model=True, save_log=True, save_config=True)
    model_dir, log_dir, config_dir = dir_maker.experiments_dir_maker(args)

    paser_config_save(args, config_dir)

    main_worker(model_dir, log_dir, args, audio_config)
    print(C.green("[!] All Single GPU Training Done"))
    print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
    print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
    print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir))
        

def main_worker(model_dir, log_dir, args, audio_config):
    
    best_acc = 0
    best_score = 0

    net = AMSD(label_dim=args.num_classes, fstride=args.fstride, tstride=args.tstride, fshape=args.fshape, tshape=args.tshape,
               input_fdim=args.num_mel_bins, input_tdim=args.target_length, model_size=args.model_size,
               mask_ratio=args.mask_ratio, mask_dense=args.mask_dense, mask_mode=args.mask_mode,
               imagenet_pretrain=True, audioset_pretrain=True, verbose=True)
    
    if not torch.cuda.is_available():
        print(C.red2("[Warnning] Using CPU, this will be slow."))            
    elif args.gpu is not None:
        print(C.underline(C.yellow("[Info] Use GPU : {} for training".format(args.gpu))))
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)       
    else:
        net = torch.nn.DataParallel(net).cuda()
        
    #  Set logger
    set_logging_defaults(log_dir, args)
    
    #  Load Dataset
    train_loader, valid_loader = my_dataloader(audio_config, noise=False, args=args, train=True, test=False, epoch=0)
    
    #  Define loss function (criterion) and optimizer
    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_CE_my = Custom_Loss(alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                                  tau=args.tau, distillation_type='soft').cuda(args.gpu) 

    trainables = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))

    #  Empty matrix for store predictions, masked token indices
    all_predictions = torch.zeros(len(train_loader.dataset)+len(valid_loader.dataset), args.num_classes, dtype=torch.float32)

    f_dim, t_dim = get_shape(args.fshape, args.tshape, args.fstride, args.tstride, args.num_mel_bins, args.target_length)
    num_masked = round(f_dim * t_dim * args.mask_ratio)
    if args.mask_mode == 'attentive':
        ids_masked  = torch.zeros(len(train_loader.dataset)+len(valid_loader.dataset), num_masked, dtype=torch.int64)
    else:
        ids_masked = None
    
    #  load status & Resume Learning
    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
        
        args.start_epoch = checkpoint['epoch'] + 1 
        best_acc = checkpoint['best_acc']
        best_score = checkpoint['best_score']
        all_predictions = checkpoint['prev_predictions'].cpu()
        ids_masked = checkpoint['prev_ids_masked'].cpu()

        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint


    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.end_epoch):
        
        #  Load Dataset
        if epoch >= 10:
            train_loader, valid_loader = my_dataloader(audio_config, noise=True, args=args, train=True, test=False, epoch=epoch)
        
        adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_acc, all_predictions, ids_masked = train(
                                            all_predictions,
                                            ids_masked, 
                                            criterion_CE_my,
                                            optimizer,
                                            net,
                                            epoch,
                                            train_loader,
                                            args)

        val_metrics = val(
                    criterion_CE,
                    net,
                    epoch,
                    valid_loader,
                    args)


        val_acc = val_metrics['acc']
        val_score = val_metrics['score']
        save_dict = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'accuracy' : val_acc,
            'prev_predictions': all_predictions,
            'prev_ids_masked': ids_masked,
            }

        if val_acc > best_acc:
            best_acc = val_acc
            save_on_master(save_dict, os.path.join(model_dir, 'checkpoint_best_acc.pth'))
            if is_main_process():
                print(C.green("[!] Save best acc checkpoint."))
                
        if val_score > best_score:
            best_score = val_score
            save_on_master(save_dict, os.path.join(model_dir, 'checkpoint_best_score.pth'))
            if is_main_process():
                print(C.green("[!] Save best score checkpoint."))

  
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            save_on_master(save_dict, os.path.join(model_dir, f'checkpoint_{epoch:03}.pth'))
            if is_main_process():
                print(C.green("[!] Save checkpoint."))
        else:
            save_on_master(save_dict, os.path.join(model_dir, f'checkpoint_{epoch:03}.pth'))
            if is_main_process():
                print(C.green("[!] Save checkpoint."))
                
        data_dict = {
            'train_loss': round(train_loss, 5),
            'train_acc':  round(train_acc, 4),
            'val_acc':    val_acc,
            'val_Sen':   val_metrics['Sen'],
            'val_Spe':   val_metrics['Spe'],
            'val_score': val_score,
        }
         
        full_data_dict = save_data_pandas(data_dict, os.path.join(log_dir, 'train_data.csv'))
        save_data_jpg(full_data_dict, ['train_loss', 'val_acc'], 'traing log', {'x': 'Epoch', 'y': 'Loss/Accuracy'}, os.path.join(log_dir, 'train_curve.png'))


#-------------------------------
# Train 
#------------------------------- 
def train(all_predictions,
          ids_masked, 
          criterion_CE_my,
          optimizer,
          net,
          epoch,
          train_loader,
          args):
    
    train_acc = AverageMeter()
    train_losses = AverageMeter()
    
    correct = 0
    total = 0

    net.train()
    current_LR = get_learning_rate(optimizer)[0]
    
    for batch_idx, (inputs, targets, input_indices) in enumerate(train_loader):
        
        if args.mask_mode == 'attentive':
            id_masked = ids_masked[input_indices]
        else:
            id_masked = None

        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            if args.mask_mode == 'attentive':
                id_masked  = id_masked.cuda(non_blocking=True) 
        
        targets_numpy = targets.cpu().detach().numpy()
        identity_matrix = torch.eye(args.num_classes) 
        targets_one_hot = identity_matrix[targets_numpy]
        
        if epoch == 0:
            all_predictions[input_indices] = targets_one_hot

        # create new soft-targets
        if torch.cuda.is_available():
            targets_one_hot = targets_one_hot.cuda()  
            
        t_outputs = all_predictions[input_indices].cuda()
            
        # student model output
        weighted_outputs, mse, id_masked, s_outputs = net(inputs, epoch, id_masked,  mode='train')
        softmax_output = F.softmax(s_outputs, dim=1) 
        loss, _, _, _ = criterion_CE_my(s_outputs, t_outputs, weighted_outputs, targets_one_hot, mse)    ### outputs: before softmax, soft_targets: after softmax 

        train_losses.update(loss.item(), inputs.size(0))
        acc = accuracy(s_outputs.data, targets, topk=(1,))
        train_acc.update(acc[0].item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(s_outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        all_predictions[input_indices] = softmax_output.cpu().detach()
        if args.mask_mode == 'attentive':
            ids_masked[input_indices] = id_masked.cpu().detach()
        
        print('Epoch: [{:3d}/{:3d}] Iter: [{:4d}/{:4d}] || lr: {:.1e} || train_loss: {:.3f} || train_acc: {:.3f} || correct/total({:4d}/{:4d})'.format(
                epoch, args.end_epoch, batch_idx+1, len(train_loader), current_LR, train_losses.avg, train_acc.avg, correct, total))
    
    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [alpha {}] [beta {}] [gamma {}] [lr {:.1e}] [train_loss {:.3f}] [train_acc {:.3f}] ] [correct/total {:4d}/{:4d}]'.format(
        epoch,
        args.alpha,
        args.beta,
        args.gamma,
        current_LR,
        train_losses.avg,
        train_acc.avg,
        correct,
        total))
    
    return train_losses.avg, train_acc.avg, all_predictions, ids_masked

#-------------------------------          
# Validation
#------------------------------- 
def val(criterion_CE,
        net,
        epoch,
        val_loader,
        args):

    val_losses = AverageMeter()

    A_targets = []
    A_predictions = []

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):              
                    
            if torch.cuda.is_available():
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
                                
            outputs = net(inputs, None, None, mode='eval')[0]
            
            targets_numpy = targets.cpu().detach().numpy()
            identity_matrix = torch.eye(args.num_classes) 
            targets_one_hot = identity_matrix[targets_numpy]   

            A_targets.append(targets_one_hot.to(torch.long))

            softmax_predictions = F.softmax(outputs, dim=1)
            A_predictions.append(softmax_predictions.to('cpu').detach())
            
            audio_predictions = torch.cat(A_predictions)
            audio_targets = torch.cat(A_targets)
            val_stats = calculate_stats(audio_predictions, audio_targets)
            
            Sen = val_stats['tpr']
            Spe = val_stats['tnr']
            acc = val_stats['acc']     # 4-class accuracy
            
            val_data_dict = {
                'acc': round(acc, 4),
                'Sen': round(Sen, 4),
                'Spe': round(Spe, 4),
                'score': round((Sen+Spe)/2, 4)
            }
            
            # loss  
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = criterion_CE(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))

            print('Epoch: [{:3d}/{:3d}] Iter: [{:4d}/{:4d}] || val_loss: {:.3f} || val_acc: {:.3f} || val_Sen: {:.3f} || val_Spe: {:.3f} || val_score: {:.3f} || correct/total({:4d}/{:4d})'.format(
                epoch, args.end_epoch, batch_idx+1, len(val_loader), val_losses.avg, acc, Sen, Spe, (Sen+Spe)/2, correct, total))

    if is_main_process():
        logger = logging.getLogger('val')
        logger.info('[Epoch {:3d}] [val_loss {:.3f}] [val_acc {:.3f}] [val_Sen {:.3f}] [val_Spe {:.3f}] [correct/total {:4d}/{:4d}]'.format(
                    epoch, val_losses.avg, acc, Sen, Spe, correct, total))

    return  val_data_dict


if __name__ == '__main__':
    init_seed(2345)
    main()
    