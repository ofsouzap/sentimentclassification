from typing import List, Tuple;

Token = str; # A token for input data
TokenSequence = List[str]; # The tokens for some input data
DataLabel = int; # The correct label associated for some input data

class DataSet:

    def __init__(self,
        train_data: List[Tuple[TokenSequence, DataLabel]],
        dev_data: List[Tuple[TokenSequence, DataLabel]],
        test_data: List[Tuple[TokenSequence, DataLabel]]):

        self.train_data = train_data;
        self.dev_data = dev_data;
        self.test_data = test_data;

    @property
    def train_data_size(self):
        return len(self.train_data);

    @property
    def dev_data_size(self):
        return len(self.dev_data);

    @property
    def test_data_size(self):
        return len(self.test_data);

    @property
    def train_data_solutions(self):
        return [c for _, c in self.train_data];

    @property
    def dev_data_solutions(self):
        return [c for _, c in self.dev_data];

    @property
    def test_data_solutions(self):
        return [c for _, c in self.test_data];
