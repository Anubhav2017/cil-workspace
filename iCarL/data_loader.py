from torchvision.datasets import  MNIST
import torchvision.transforms as transforms
import numpy as np
import torch
import time
from random import sample
import genomics

class genomics_data():
	def __init__(self, 
				 classes=range(66),
				 extra_classes=None,
				 nsamples=None,
				 train=True
				 ):	
		self.train = train
		
		xtrain,ytrain,xtest,ytest= genomics.load_dataset()
		# print("dataset loaded")
		

		# Select subset of classes
		if self.train:
			self.train_data = xtrain
			self.train_labels = ytrain

			extra_train_data=dict()
			for cl in extra_classes:
				extra_train_data[cl] = []

			train_data = []
			train_labels = []

			for i in range(len(self.train_data)):
				if self.train_labels[i] in classes:	
					train_data.append(self.train_data[i])
					train_labels.append((self.train_labels[i]))

				if self.train_labels[i] in extra_classes:
					extra_train_data[self.train_labels[i]] += [self.train_data[i]]
			
			for k in extra_train_data.keys():
				if nsamples is not None:
					train_data += sample(extra_train_data[k], nsamples)
					train_labels += [k]*nsamples
				
					
			self.train_data = np.array(train_data, dtype = np.float32)
			self.train_labels = np.array(train_labels)

		else:
			self.test_data = xtest
			self.test_labels = ytest
			test_data = []
			test_labels = []

			for i in range(len(self.test_data)):
				if self.test_labels[i] in classes:
					test_data.append(self.test_data[i])
					test_labels.append((self.test_labels[i]))
					
			self.test_data = np.array(test_data, dtype = np.float32)
			self.test_labels = test_labels


	def __getitem__(self, index):
		# print(index)
		if self.train:
			xdata = self.train_data[index]
			
			xdata = torch.FloatTensor(xdata)
			target = self.train_labels[index]
		else:
			xdata, target = self.test_data[index], self.test_labels[index]

		xdata = torch.FloatTensor(xdata)
		
		return index, xdata, target

	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.test_data)
		


