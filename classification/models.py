import torch 
from torch import nn


class CircleModel(nn.Module):
    def __init__(self, input_features: int, output_features :int, hidden_units:int = 8  ) -> None:
        super().__init__()
        self.linear_stack_layers = nn.Sequential(
            nn.Linear(in_features=input_features, out_features= hidden_units ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features= hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features= output_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_stack_layers(x)
    

class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_stack_layers = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )
    
    def forward(self, x: torch.Tensor) -> None:
        return self.linear_stack_layers(x)        

