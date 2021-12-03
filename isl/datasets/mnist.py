from torchvision import datasets

from .dataset import Dataset


class MNIST(Dataset):

    def __init__(self, cache_dir: str = '.cisl'):
        super(MNIST, self).__init__(cache_dir)

        self._train = datasets.MNIST(self._cache_dir, train=True, download=True)
        self._test = datasets.MNIST(self._cache_dir, train=False, download=True)

        self._n_class = len(self._train.classes)
        self._n_train = len(self._train)
        self._n_test = len(self._test)