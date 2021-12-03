from torchvision import datasets

from isl.datasets.dataset import Dataset


class CIFAR10(Dataset):

    def __init__(self, cache_dir: str = '.cisl') -> None:
        super(CIFAR10, self).__init__(cache_dir)

        self._train = datasets.CIFAR10(self._cache_dir, train=True, download=True)
        self._test = datasets.CIFAR10(self._cache_dir, train=False, download=True)

        self._n_class = len(self._train.classes)
        self._n_train = len(self._train)
        self._n_test = len(self._test)

class CIFAR100(Dataset):

    def __init__(self, cache_dir: str = '.cisl') -> None:
        super(CIFAR100, self).__init__(cache_dir)

        self._train = datasets.CIFAR100(self._cache_dir, train=True, download=True)
        self._test = datasets.CIFAR100(self._cache_dir, train=False, download=True)

        self._n_class = len(self._train.classes)
        self._n_train = len(self._train)
        self._n_test = len(self._test)
