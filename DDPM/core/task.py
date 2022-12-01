from abc import ABC

from torch.utils.data import DataLoader

from DDPM.data.empatheric_dialogues import EmpatheticDialogues


class Task(ABC):

    def __init__(self, args):
        pass

    def train_epoch(self, model, loader):
        for step, inputs in enumerate(loader):
            outputs = model.train_step(inputs)

    def get_data_loader(self, args, phase='train'):
        sampler = None
        dataset = EmpatheticDialogues(args=args, phase=phase)
        return DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.pad_batch_seq)
