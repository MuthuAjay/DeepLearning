from abc import abstractmethod
from typing import Optional


class ModelHandler:

    def __init__(self):
        pass

    def __add__(self):
        pass

    @abstractmethod
    def process(self,
                method: Optional[int],
                input_shape: Optional[int] = 3,
                hidden_units: Optional[int] = 10,
                epochs: int = 5,
                num_of_layers: int = 2
                ):
        """
        :return:
        """
        raise NotImplemented

