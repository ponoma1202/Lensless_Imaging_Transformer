import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import sys
sys.path.append('/home/ponoma/workspace/Pan_Transformer/utils/')  # TODO: would not recognize local path to data_prepare
from data_prepare import get_data

def get_loader(cfg):
    train_dataset = get_data(
        input_size=(cfg.basic.H, cfg.basic.W),
        output_size=(cfg.basic.H, cfg.basic.W),
        save_dir=cfg.dir.dataset_dir,
        split='train',
        dataset=cfg.basic.dataset,
        downsize_coeff=cfg.basic.downsize_coeff
    )

    val_dataset = get_data(
        input_size=(cfg.basic.H, cfg.basic.W),
        output_size=(cfg.basic.H, cfg.basic.W),
        save_dir=cfg.dir.dataset_dir,
        split='val',
        dataset=cfg.basic.dataset,
        downsize_coeff=cfg.basic.downsize_coeff
    )

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.train_batch_size,
                              #num_workers=cfg.train.GPU_num*4,
                              num_workers=cfg.train.GPU_num*1,
                              pin_memory=True,
                              drop_last=False,
                              sampler=train_sampler,
                              prefetch_factor=2)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            #num_workers=cfg.train.GPU_num*4,
                            num_workers=cfg.train.GPU_num*1,
                            pin_memory=True,
                            drop_last=False)

    return train_loader, val_loader