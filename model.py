from typing import List, Tuple, Union;
from dataset import DataLabel, TokenSequence;
from abc import ABC, abstractmethod;
from progress.bar import ShadyBar;

class RunningUntrainedModelException(Exception):
    pass;

class Model(ABC):

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        pass;

    @abstractmethod
    def train(self, data: List[Tuple[TokenSequence, DataLabel]], use_progress_bar: bool = False) -> None:
        pass;

    @abstractmethod
    def run(self, tokens: TokenSequence) -> DataLabel:
        pass;

    def multi_run(self, data: List[TokenSequence]) -> List[DataLabel]:

        predictions: List[DataLabel] = [];

        for item in data:
            predictions.append(self.run(item));

        return predictions;

    def train_progress_init(self, max: int) -> None:
        self.__train_progress_bar = ShadyBar("Training model", max=max);

    def train_progress_next(self) -> None:
        self.__train_progress_bar.next();

    def train_progress_finish(self) -> None:
        self.__train_progress_bar.finish();
