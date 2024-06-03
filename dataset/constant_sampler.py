""" Samples data in a constant manner. """
import torch 

from torch.utils.data.sampler import Sampler
from typing import Iterator, Sized




class ConstantRandomSampler(Sampler[int]):
    def __init__(self, data_source: Sized) -> None:
        self.num_samples = len(data_source)
        generator = torch.Generator()

        self.shuffled_list = torch.randperm(self.num_samples, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        yield from self.shuffled_list

    def __len__(self) -> int:
        return self.num_samples

