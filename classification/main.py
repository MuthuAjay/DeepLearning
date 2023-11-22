#%%
import torch 
from torch import nn 
from datasets import *
from train import *
from models import *
from utils import *


class main(TrainModel):
    def __init__(self, epochs: int = 1000, manual_seed: int | float = 42) -> None:
        super().__init__(epochs, manual_seed)

    
    def plot(self,model, X,y, title, n):
        plt.subplot(1,2,n)
        plt.title(title)
        Utils.plot_decision_boundary(model, X, y)

    def process(self):
        print("Model You wanna Create")
        print("1. Binary Classification | 2. Multi Class Classification")

        # model = int(input())
        model = 2

        if model == 1:
            print("Binary Classification")
            print("Create Dataset")
            dataset = Datasets(test_size= 0.2, random_state= 42)
            X_train, X_test, y_train, y_test = dataset.circle_dataset(n_samples= 1000,
                                                                        noise= 0.02,
                                                                        random_state= 42)        
            model = CircleModel(input_features= 2,
                                output_features= 1,
                                hidden_units= 20).to(Utils.set_device())
            
            print(model)
            
            
            model = self.train_binary_class_model(model= model,
                                    X_train=X_train, X_test=X_test,
                                    y_train=y_train, y_test =y_test,
                                    learning_rate= 0.1)
            
            self.plot(model,X_train, y_train, "Train", 1)
            self.plot(model,X_test, y_test, "Test", 2)

        elif model == 2:
            print("Multi class Classification")
            print("Create Dataset")
            dataset = Datasets(test_size= 0.2, random_state= 42)
            X_train, X_test, y_train, y_test = dataset.blob_dataset(n_samples= 1000,
                                                                    num_features= 2,
                                                                    num_classes=4,
                                                                    cluster_std= 1,
                                                                    random_state= 42)
            
            model = BlobModel(input_features = 2,
                              output_features = 4,
                              hidden_units = 10).to(Utils.set_device())
            
            model = self.train_multiclass_model(model = model,
                                    X_train=X_train, X_test=X_test,
                                    y_train=y_train, y_test =y_test,
                                    learning_rate  = 0.1
                                        )
            self.plot(model,X_train, y_train, "Train", 1)
            self.plot(model,X_test, y_test, "Test", 2)

if __name__ == '__main__':
    models = main()
    models.epochs = 1000    
    models.process()
#%%