import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


BATCH_SIZE = 128

def download_dataset():
	train_data = datasets.MNIST(
		root = "data",
		download = False,
		train = True,
		transform = ToTensor()
		
	validation_data = datasets.MNIST(
		root = "data",
		download = False,
		train = False,
		transform = ToTensor()		
		
	)
	
	return train_data, validation_data
	
if __name__ == "__main__":
	
	train_data
	print("MNIST_dataset")
	
	train_data_loader = DataLoader(train_data, batch_size = BATCH_SIZE)
	

class FeedForwardNet(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.dense_layers = nn.Sequential(
			nn.Linear(28*28, 256),
			nn.ReLU(),
			nn.Linear(256, 10)
			
		)
		self.softmax = nn.Softmax(dim=1)
		
	def forward(self, input_data):
		flatten_data = self.flatten(input_data)
		logit
		
		
		

