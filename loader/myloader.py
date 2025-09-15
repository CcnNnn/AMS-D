# author: Nuo Chen (2025)

import os
import torch
from .mydataset import ICBHIDataset
from torch.utils.data import DataLoader, random_split, ConcatDataset

SEED = 2345

def my_dataloader(audio_config, noise, args, epoch, train:True, test:False):
    A_dataloader = []
    
    if train:
        if len(args.data_root) > 1:
            dataset = []
            for data_path in args.data_root:
                dataset.append(ICBHIDataset(os.path.join(data_path, 'train'), audio_config=audio_config, noise=noise, epoch=epoch))
            dataset = ConcatDataset(dataset)
        else:    
            dataset = ICBHIDataset(os.path.join(args.data_root[0], 'train'), audio_config=audio_config, noise=noise, epoch=epoch)
        n_train = int(len(dataset) * 0.8)
        n_valid = len(dataset) - n_train
        train_dataset, valid_dataset = random_split(dataset, [n_train, n_valid], generator=torch.Generator().manual_seed(0))

        print("------ load train dataset ------")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, generator=torch.Generator().manual_seed(SEED))
        print("train_dataset:", n_train)

        print("------ load valid dataset ------")
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, generator=torch.Generator().manual_seed(SEED))
        print("valid_dataset:", n_valid)

        A_dataloader.append(train_loader)
        A_dataloader.append(valid_loader)
    
    if test:
        test_dataset = ICBHIDataset(os.path.join(args.data_root, 'test'), audio_config=audio_config, noise=noise, epoch=epoch)
        print('------ load test dataset ------')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        print("test_dataset:", len(test_dataset))

        A_dataloader.append(test_loader)
    
    return A_dataloader


