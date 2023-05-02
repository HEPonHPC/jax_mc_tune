from typing import Optional

from .reader import InputReader
from .filter import Filters
from .surrogate import SurrogateModel

class InnerLoopOptimizer:
    def __init__(self,
                 input_reader: InputReader,
                 surrogate_model: SurrogateModel,
                 outdir: str,
                 filters: Optional[Filters] = None,
                 name: str = "InnerLoopOptimizer"):
        self.reader = input_reader
        self.surrogate = surrogate_model
        self.outdir = outdir
        self.filters = filters
        self.name = name

    def optimize(self):
        if self.filters is not None:
            self.reader = self.filters.apply(self.reader)

        num_of_bins = self.reader.num_of_bins
        for bin_idx in range(num_of_bins):
            self.surrogate.fit(self.reader, bin_idx=bin_idx)

        self.surrogate.save(self.outdir + "/surrogate_model")
