from torchvision import datasets
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize

from isl.datasets.dataset import Dataset


class MNIST(Dataset):

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

        transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self._train = datasets.MNIST(self._cache_dir, train=True, transform=transform, download=self._download)
        self._test = datasets.MNIST(self._cache_dir, train=False, transform=transform, download=False)

        self._n_class = len(self._train.classes)
        self._n_train = len(self._train)
        self._n_test = len(self._test)
