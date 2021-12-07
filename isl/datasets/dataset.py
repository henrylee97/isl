from abc import ABC, abstractmethod
from typing import Iterable

from isl.utils.string import StringBuilder


class Dataset(ABC):

    @abstractmethod
    def __init__(self, cache_dir: str = '.isl', download=False) -> None:
        self._cache_dir = cache_dir
        self._download = download

        self._train: Iterable = None
        self._test: Iterable = None

        self._n_class: int = None
        self._n_train: int = None
        self._n_test: int = None

    @property
    def train(self) -> Iterable:
        return self._train

    @property
    def test(self) -> Iterable:
        return self._test

    def __str__(self):
        builder = StringBuilder()
        builder.append_line(f'Dataset {self.__class__.__name__}')
        builder.indent().append_line(f'Root location: {self._cache_dir}')
        builder.append_line(f'Number of classes: {self._n_class}')
        builder.append_line('Train split')
        builder.indent().append_line(f'Number of datapoints: {self._n_train}')
        builder.dedent().append_line('Test split')
        builder.indent().append_line(f'Number of datapoints: {self._n_test}')
        return str(builder)
