"""datasets.py"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUN, ImageFolder

__datasets__ = ['cifar10', 'celeba', 'lsun']


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    if not is_power_of_2(image_size) or image_size < 32:
        raise ValueError('image size should be 32, 64, 128, ...')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if name.lower() == 'cifar10':
        root = Path(dset_dir).joinpath('CIFAR10')
        train_kwargs = {'root':root, 'train':True, 'transform':transform, 'download':True}
        dset = CIFAR10
    elif name.lower() == 'celeba':
        root = Path(dset_dir).joinpath('CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = ImageFolder
    elif name.lower() == 'lsun':
        raise NotImplementedError('{} is not supported yet'.format(name))
        root = Path(dset_dir).joinpath('LSUN')
        train_kwargs = {'root':str(root), 'classes':'train', 'transform':transform}
        dset = LSUN
    else:
        root = Path(dset_dir).joinpath(name)
        train_kwargs = {'root':root, 'transform':transform}
        dset = ImageFolder

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = dict()
    data_loader['train'] = train_loader

    return data_loader


if __name__ == '__main__':
    import argparse
    #os.chdir('..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--dset_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    data_loader = return_data(args)
