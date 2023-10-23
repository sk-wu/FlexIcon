import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np
import random

def CreateDataset(opt):
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateDataset_test(opt):
    from data.aligned_dataset import AlignedDataset_test
    dataset = AlignedDataset_test()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        if opt.isTrain:
            self.dataset = CreateDataset(opt)
        else:
            self.dataset = CreateDataset_test(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
        worker_init_fn=lambda _: np.random.seed())

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def random_sampler(size):
    while True:
        yield random.randrange(size)

class CustomDatasetDataLoader_1(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader_1'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        if opt.isTrain:
            self.dataset = CreateDataset(opt)
        else:
            self.dataset = CreateDataset_test(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=random_sampler(len(self.dataset)),
            pin_memory=True,
            num_workers=int(opt.nThreads),
        worker_init_fn=lambda _: np.random.seed())

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


def CreateDataset_palette(opt):
    from data.aligned_dataset import AlignedDataset_palette
    dataset = AlignedDataset_palette()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateDataset_test_palette(opt):
    from data.aligned_dataset import AlignedDataset_test_palette
    dataset = AlignedDataset_test_palette()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader_palette(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader_Palette'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        if opt.isTrain:
            self.dataset = CreateDataset_palette(opt)
        else:
            self.dataset = CreateDataset_test_palette(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            worker_init_fn=lambda _: np.random.seed())

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)