from torchvision import datasets
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

from isl.datasets.dataset import Dataset


class CIFAR10(Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super(CIFAR10, self).__init__(*args, **kwargs)

        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self._train = datasets.CIFAR10(self._cache_dir, train=True, transform=transform, download=self._download)
        self._test = datasets.CIFAR10(self._cache_dir, train=False, transform=transform, download=False)

        self._n_class = len(self._train.classes)
        self._n_train = len(self._train)
        self._n_test = len(self._test)


class CIFAR100(Dataset):

    def __init__(self, *args, **kwargs) -> None:
        super(CIFAR100, self).__init__(*args, **kwargs)

        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self._train = datasets.CIFAR100(self._cache_dir, train=True, transform=transform, download=self._download)
        self._test = datasets.CIFAR100(self._cache_dir, train=False, transform=transform, download=False)

        self._n_class = len(self._train.classes)
        self._n_train = len(self._train)
        self._n_test = len(self._test)
