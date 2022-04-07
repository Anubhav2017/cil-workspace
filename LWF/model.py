import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy

import torchvision.models as models
import torchvision.transforms as transforms
device="cpu"
def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
	labels = Variable(labels.data, requires_grad=False).to(device)
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return Variable(outputs.data, requires_grad=True).to(device)

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class Model(nn.Module):
	def __init__(self, classes, classes_map, args):
		# Hyper Parameters
		self.init_lr = args.init_lr

		self.lr=self.init_lr
		self.num_epochs = args.num_epochs
		self.batch_size = args.batch_size
		self.lower_rate_epoch = np.arange(int(0.1 * self.num_epochs), int(0.9 * self.num_epochs),10) #hardcoded decay schedule
		self.lr_dec_factor = 1.5
		
		self.pretrained = False
		self.momentum = 0.0
		self.weight_decay = 0.0005
		# Constant to provide numerical stability while normalizing
		self.epsilon = 1e-16

		# Network architecture
		super(Model, self).__init__()
		model = nn.Sequential(
		nn.Conv1d(1, 6, kernel_size=6),
		nn.ReLU(),
		nn.Conv1d(6, 3, kernel_size=6),
		nn.ReLU(),
		nn.Flatten(),
		nn.Linear(738,512),
		nn.ReLU()
		)

		self.feature_extractor=model

		self.feature_extractor.apply(kaiming_normal_init)
		self.fc = nn.Linear(512, classes, bias=False)

		self.n_classes = 0
		self.n_known = 0
		self.classes_map = classes_map

	def forward(self, x):
		x = self.feature_extractor(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

	def increment_classes(self, new_classes):
		"""Add n classes in the final fc layer"""
		n = len(new_classes)
		print('new classes: ', n)
		in_features = self.fc.in_features
		out_features = self.fc.out_features
		weight = self.fc.weight.data

		if self.n_known == 0:
			new_out_features = n
		else:
			new_out_features = out_features + n
		print('new out features: ', new_out_features)
		self.fc = nn.Linear(in_features, new_out_features, bias=False)
		
		kaiming_normal_init(self.fc.weight)
		self.fc.weight.data[:out_features] = weight
		self.n_classes += n

	def classify(self, images):
		"""Classify images by softmax

		Args:
			x: input image batch
		Returns:
			preds: Tensor of size (batch_size,)
		"""
		_, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

		return preds

	def update(self, dataset, class_map):

		self.compute_means = True
		self.lr=self.init_lr

		# Save a copy to compute distillation outputs
		prev_model = copy.deepcopy(self)
		prev_model.to(device)

		classes = list(set(dataset.train_labels))
		#print("Classes: ", classes)
		print('Known: ', self.n_known)
		# if self.n_classes == 1 and self.n_known == 0:
		# 	new_classes = [classes[i] for i in range(1,len(classes))]
		# else:
		new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

		print('New classes: ', new_classes)

		# print(self.lower_rate_epoch)
		

		if len(new_classes) > 0:
			self.increment_classes(new_classes)
			self.to(device)

		

		loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
											   shuffle=True, num_workers=8)

		print("Batch Size (for n_classes classes) : ", len(dataset))
		optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum = self.momentum, weight_decay=self.weight_decay)

		# for param in self.feature_extractor.parameters():
		# 	param.requires_grad = False

		
		# print("warming up fc layer")
		# for _ in range(5):
		# 	for i, (indices, images, labels) in enumerate(loader):
		# 		seen_labels = []
		# 		images = Variable(torch.FloatTensor(images)).to(device)
		# 		# print(labels)
		# 		seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
		# 		labels = Variable(seen_labels).to(device)
		# 		# indices = indices.cuda()

		# 		optimizer.zero_grad()
		# 		logits = self.forward(images)

		# 		loss = nn.CrossEntropyLoss()(logits, labels)

		# 		loss.backward()
		# 		optimizer.step()
		
		# print("warmup done")
		# for param in self.feature_extractor.parameters():
		# 	param.requires_grad = True



		with tqdm(total=self.num_epochs) as pbar:
			for epoch in range(self.num_epochs):
				
				# Modify learning rate
				# if (epoch+1) in self.lower_rate_epoch:
				# 	self.lr = self.lr * 1.0/self.lr_dec_factor
				# 	for param_group in optimizer.param_groups:
				# 		param_group['lr'] = self.lr

				# print((self.n_classes-self.n_known))

				
				for i, (indices, images, labels) in enumerate(loader):
					seen_labels = []
					images = Variable(torch.FloatTensor(images)).to(device)
					seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
					labels = Variable(seen_labels).to(device)
					# indices = indices.cuda()

					optimizer.zero_grad()
					logits = self.forward(images)
					# print(logits.size())
					# print(logits)
					# print(labels)
					cls_loss = nn.CrossEntropyLoss()(logits, labels)
					loss = cls_loss
					if len(new_classes) > 0:
						dist_target = prev_model.forward(images)
						logits_dist = logits[:,:-(self.n_classes-self.n_known)]
						dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
						# dist_loss=nn.CrossEntropyLoss()(logits_dist, dist_target)
						loss = dist_loss+0.1*cls_loss
					else:
						loss = cls_loss

					loss.backward()
					optimizer.step()

					if (i) == 0:
						tqdm.write('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, lr = %.6f' 
							   %(epoch+1, self.num_epochs, i+1, np.ceil(len(dataset)/self.batch_size), loss.data, self.lr))

				pbar.update(1)






