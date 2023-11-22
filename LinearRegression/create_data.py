import torch

class DataSet():
    
    def __init__(self, weights, bias) -> None:
        self.inputs : torch.Tensor = torch.Tensor
        self.weights = weights
        self.bias = bias
    
    def create_train_test_split(self):
        self.train_split = int(0.8 * len(self.inputs))
        self.x_train, self.x_test = self.inputs[:self.train_split] , self.inputs[self.train_split:]
        self.y_train , self.y_test = self.target[:self.train_split], self.target[self.train_split:]

    def create_1D_data_set(self, start: int|float , end: int|float, step : int|float) -> torch.Tensor:
        
        self.inputs : torch.Tensor = torch.arange(start, end, step).unsqueeze(dim = 1)
        self.target : torch.Tensor = self.inputs * self.weights + self.bias
        self.create_train_test_split()

        return self.x_train, self.x_test, self.y_train, self.y_test

