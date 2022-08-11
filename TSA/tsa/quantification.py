#Py2/Py3 Compatibility
from __future__ import print_function, unicode_literals, absolute_import, division
#from cProfile import label
#from csv import list_dialects

import warnings

#Matplotlib library for plots
import matplotlib
#from matplotlib import testing
#from sklearn.cluster import mean_shift
#from sklearn.linear_model import PassiveAggressiveClassifier
matplotlib.use('Agg') # Crashes with SSH connections if this isn't set.
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.colors

#Python packages

import time
import copy
import bisect
import re
import subprocess
import importlib
import os
import datetime
import sys
import collections
import random
import functools
import math
from operator import itemgetter

#Linking from other files
from .sniffer import Transform
from .sniffer import Utils
from .sniffer import Sniffer
from .alignment import Alignment
from .sniffer import Packet

#Numpy, Scipy and Scikit-learn for statistical modeling and information quantification
import numpy as np
#import scipy.stats
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture#, GaussianMixture
from sklearn.preprocessing import normalize#, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold#,ShuffleSplit, LeaveOneOut, RepeatedKFold
#from sklearn.metrics import log_loss

#Cross Validation for Models for Train/Test Split (No Parameter Search yet)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score
#from imblearn.metrics import geometric_mean_score, specificity_score

#Packages for Classifications (SVM and Logistic Regression)
#import sklearn
#from sklearn.svm import LinearSVC, SVC, NuSVC # Support Vector Machines
#from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

#Deep Learning Support
#import pytorch
#import torchvision
#import torchvision.transforms as transforms
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

#Dataset setup
#from torch.utils.data import Dataset, DataLoader
#from torchvision import utils
#import pandas as pd
#from skimage import io, transform

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torch.utils.data import Dataset#, DataLoader
#import skorch
#from skorch import NeuralNetClassifier
#import skorch.callbacks

# License: http://creativecommons.org/publicdomain/zero/1.0/
colors = ['#acc2d9', '#56ae57', '#b2996e', '#a8ff04', '#69d84f',
	'#894585', '#70b23f','#d4ffff','#65ab7c','#952e8f','#96f97b',
	'#fcfc81','#a5a391','#388004','#4c9085','#5e9b8a','#efb435',
	'#d99b82','#01386a','#25a36f','#59656d','#75fd63','#21fc0d',
	'#0a5f38','#0c06f7','#61de2a','#3778bf','#2242c7','#533cc6',
	'#9bb53c','#05ffa6','#1f6357','#017374','#0cb577','#ff0789',
	'#afa88b','#08787f','#dd85d7','#a6c875','#a7ffb5','#c2b709',
	'#e78ea5','#966ebd','#ccad60','#ac86a8','#947e94','#983fb2',
	'#ff63e9','#b2fba5','#63b365','#8ee53f','#b7e1a1','#ff6f52',
	'#bdf8a3','#d3b683','#fffcc4','#430541','#ffb2d0','#997570',
	'#ad900d','#c48efd','#507b9c','#7d7103','#fffd78','#da467d',
	'#410200','#c9d179','#fffa86','#5684ae','#6b7c85','#6f6c0a',
	'#7e4071','#009337','#d0e429','#fff917','#1d5dec','#054907',
	'#b5ce08','#8fb67b','#c8ffb0','#fdde6c','#ffdf22','#a9be70',
	'#6832e3','#fdb147','#c7ac7d','#fff39a','#850e04','#efc0fe',
	'#40fd14','#b6c406','#9dff00','#3c4142','#f2ab15','#ac4f06',
	'#c4fe82','#2cfa1f','#9a6200','#ca9bf7','#875f42','#3a2efe',
	'#fd8d49','#8b3103','#cba560','#698339','#0cdc73','#b75203',
	'#7f8f4e','#26538d','#63a950','#c87f89','#b1fc99','#ff9a8a',
	'#f6688e','#76fda8','#53fe5c','#4efd54','#a0febf','#7bf2da',
	'#bcf5a6','#ca6b02','#107ab0','#2138ab','#719f91','#fdb915',
	'#fefcaf','#fcf679','#1d0200','#cb6843','#31668a','#247afd',
	'#ffffb6','#90fda9','#86a17d','#fddc5c','#78d1b6','#13bbaf',
	'#fb5ffc','#20f986','#ffe36e','#9d0759','#3a18b1','#c2ff89',
	'#d767ad','#720058','#ffda03','#01c08d','#ac7434','#014600',
	'#9900fa','#02066f','#8e7618','#d1768f','#96b403','#fdff63',
	'#95a3a6','#7f684e','#751973','#089404','#ff6163','#598556',
	'#214761','#3c73a8','#ba9e88','#021bf9','#734a65','#23c48b',
	'#4b57db','#d90166','#015482','#9d0216','#728f02','#ffe5ad',
	'#4e0550','#f9bc08','#ff073a','#c77986','#d6fffe','#fe4b03',
	'#fd5956','#fce166','#8fae22','#e6f2a2','#89a0b0','#7ea07a',
	'#1bfc06','#b9484e','#647d8e','#bffe28','#d725de','#886806',
	'#b2713d','#1f3b4d','#699d4c','#56fca2','#fb5581','#3e82fc',
	'#a0bf16','#d6fffa','#4f738e','#ffb19a','#5c8b15','#54ac68',
	'#cafffb','#b6ffbb','#a75e09','#152eff','#8d5eb7','#5f9e8f',
	'#63f7b4','#606602','#fc86aa','#8c0034','#758000','#ab7e4c',
	'#030764','#fe86a4','#d5174e','#fed0fc','#680018','#fedf08',
	'#fe420f','#6f7c00','#ca0147','#1b2431','#00fbb0','#db5856',
	'#ddd618','#41fdfe','#cf524e','#21c36f','#a90308','#6e1005',
	'#fe828c','#4b6113','#4da409','#beae8a','#0339f8','#a88f59',
	'#5d21d0','#feb209','#4e518b','#964e02','#85a3b2','#ff69af',
	'#c3fbf4','#2afeb7','#005f6a','#0c1793','#ffff81','#fd4659',
	'#f0833a','#f1f33f','#b1d27b','#fc824a','#71aa34','#b7c9e2',
	'#4b0101','#a552e6','#af2f0d','#8b88f8','#9af764','#a6fbb2',
	'#ffc512','#750851','#c14a09','#fe2f4a','#0203e2','#0a437a',
	'#a50055','#ae8b0c','#fd798f','#bfac05','#3eaf76','#c74767',
	'#b29705','#673a3f','#a87dc2','#fafe4b','#c0022f','#0e87cc',
	'#8d8468','#ad03de','#8cff9e','#94ac02','#c4fff7','#fdee73',
	'#33b864','#fff9d0','#758da3','#f504c9','#adf802','#c1c6fc',
	'#35ad6b','#fffd37','#a442a0','#f36196','#c6f808','#f43605',
	'#77a1b5','#8756e4','#889717','#c27e79','#017371','#9f8303',
	'#f7d560','#bdf6fe','#75b84f','#9cbb04','#29465b','#696006',
	'#947706','#fff4f2','#1e9167','#b5c306','#feff7f','#cffdbc',
	'#0add08','#87fd05','#1ef876','#7bfdc7','#bcecac','#bbf90f',
	'#ab9004','#1fb57a','#00555a','#a484ac','#c45508','#3f829d',
	'#548d44','#c95efb','#3ae57f','#016795','#87a922','#f0944d',
	'#5d1451','#25ff29','#d0fe1d','#ffa62b','#01b44c','#ff6cb5',
	'#6b4247','#c7c10c','#b7fffa','#aeff6e','#ec2d01','#76ff7b',
	'#730039','#040348','#df4ec8','#6ecb3c','#8f9805','#5edc1f',
	'#d94ff5','#c8fd3d','#070d0d','#4984b8','#51b73b','#ac7e04',
	'#4e5481','#876e4b','#58bc08','#2fef10','#2dfe54','#0aff02',
	'#9cef43','#18d17b','#35530a','#ef4026','#3c9992','#d0c101',
	'#1805db','#6258c4','#ff964f','#ffab0f','#8f8ce7','#24bca8',
	'#3f012c','#cbf85f','#ff724c','#280137','#b36ff6','#48c072',
	'#bccb7a','#a8415b','#06b1c4','#cd7584','#f1da7a','#ff0490',
	'#805b87','#50a747','#a8a495','#cfff04','#ffff7e','#ff7fa7',
	'#04f489','#fef69e','#cfaf7b','#3b719f','#fdc1c5','#20c073',
	'#9b5fc0','#0f9b8e','#742802','#9db92c','#a4bf20','#cd5909',
	'#ada587','#be013c','#b8ffeb','#dc4d01','#a2653e','#638b27',
	'#419c03','#b1ff65','#9dbcd4','#fdfdfe','#77ab56','#464196',
	'#990147','#befd73','#32bf84','#af6f09','#a0025c','#ffd8b1',
	'#7f4e1e','#bf9b0c','#6ba353','#f075e6','#7bc8f6','#475f94',
	'#f5bf03','#fffeb6','#fffd74','#895b7b','#436bad','#05480d',
	'#c9ae74','#60460f','#98f6b0','#8af1fe','#2ee8bb','#03719c',
	'#02c14d','#b25f03','#2a7e19','#490648','#536267','#5a06ef',
	'#cf0234','#c4a661','#978a84','#1f0954','#03012d','#2bb179',
	'#c3909b','#a66fb5','#770001','#922b05','#7d7f7c','#990f4b',
	'#8f7303','#c83cb9','#fea993','#acbb0d','#c071fe','#ccfd7f',
	'#00022e','#828344','#ffc5cb','#ab1239','#b0054b','#99cc04',
	'#937c00','#019529','#ef1de7','#000435','#42b395','#9d5783',
	'#c8aca9','#c87606','#aa2704','#e4cbff','#fa4224','#0804f9',
	'#5cb200','#76424e','#6c7a0e','#fbdd7e','#2a0134','#044a05',
	'#0d75f8','#fe0002','#cb9d06','#fb7d07','#b9cc81','#edc8ff',
	'#61e160','#8ab8fe','#920a4e','#fe02a2','#9a3001','#65fe08',
	'#befdb7','#b17261','#885f01','#02ccfe','#c1fd95','#836539',
	'#fb2943','#84b701','#b66325','#7f5112','#5fa052','#6dedfd',
	'#0bf9ea','#c760ff','#ffffcb','#f6cefc','#155084','#f5054f',
	'#645403','#7a5901','#a8b504','#3d9973','#000133','#76a973',
	'#2e5a88','#0bf77d','#bd6c48','#ac1db8','#2baf6a','#26f7fd',
	'#aefd6c','#9b8f55','#ffad01','#c69c04','#f4d054','#de9dac',
	'#11875d','#fdb0c0','#b16002','#f7022a','#d5ab09','#86775f',
	'#c69f59','#7a687f','#042e60','#c88d94','#a5fbd5','#fffe71',
	'#6241c7','#fffe40','#d3494e','#985e2b','#a6814c','#ff08e8',
	'#9d7651','#feffca','#98568d','#9e003a','#287c37','#b96902',
	'#ba6873','#ff7855','#94b21c','#c5c9c7','#661aee','#6140ef',
	'#9be5aa','#7b5804','#276ab3','#feb308','#5a86ad','#fec615',
	'#8cfd7e','#6488ea','#056eee','#b27a01','#0ffef9','#fa2a55',
	'#820747','#7a6a4f','#f4320c','#a13905','#6f828a','#a55af4',
	'#ad0afd','#004577','#658d6d','#ca7b80','#005249','#2b5d34',
	'#bff128','#b59410','#2976bb','#014182','#bb3f3f','#fc2647',
	'#a87900','#82cbb2','#667c3e','#658cbb','#749551','#cb7723',
	'#05696b','#ce5dae','#c85a53','#96ae8d','#1fa774','#40a368',
	'#fe46a5','#fe83cc','#94a617','#a88905','#7f5f00','#9e43a2',
	'#062e03','#8a6e45','#cc7a8b','#9e0168','#fdff38','#c0fa8b',
	'#eedc5b','#7ebd01','#3b5b92','#01889f','#3d7afd','#5f34e7',
	'#6d5acf','#748500','#706c11','#3c0008','#cb00f5','#002d04',
	'#b9ff66','#9dc100','#faee66','#7efbb3','#7b002c','#c292a1',
	'#017b92','#fcc006','#657432','#d8863b','#738595','#aa23ff',
	'#08ff08','#9b7a01','#f29e8e','#6fc276','#ff5b00','#fdff52',
	'#866f85','#8ffe09','#d6b4fc','#020035','#703be7','#fd3c06',
	'#eecffe','#510ac9','#4f9153','#9f2305','#728639','#de0c62',
	'#916e99','#ffb16d','#3c4d03','#7f7053','#77926f','#010fcc',
	'#ceaefa','#8f99fb','#c6fcff','#5539cc','#544e03','#017a79',
	'#01f9c6','#c9b003','#929901','#0b5509','#960056','#f97306',
	'#a00498','#2000b1','#94568c','#c2be0e','#748b97','#665fd1',
	'#9c6da5','#c44240','#a24857','#825f87','#c9643b','#90b134',
	'#fffd01','#dfc5fe','#b26400','#7f5e00','#de7e5d','#048243',
	'#ffffd4','#3b638c','#b79400','#84597e','#411900','#7b0323',
	'#04d9ff','#667e2c','#fbeeac','#d7fffe','#4e7496','#874c62',
	'#d5ffff','#826d8c','#ffbacd','#d1ffbd','#448ee4','#05472a',
	'#d5869d','#3d0734','#4a0100','#f8481c','#02590f','#89a203',
	'#e03fd8','#d58a94','#7bb274','#526525','#c94cbe','#db4bda',
	'#9e3623','#b5485d','#735c12','#9c6d57','#028f1e','#b1916e',
	'#49759c','#a0450e','#39ad48','#b66a50','#8cffdb','#a4be5c',
	'#7a9703','#ac9362','#01a049','#d9544d','#fa5ff7','#82cafc',
	'#acfffc','#fcb001','#910951','#fe2c54','#c875c4','#cdc50a',
	'#fd411e','#9a0200','#be6400','#030aa7','#fe019a','#f7879a',
	'#887191','#b00149','#12e193','#fe7b7c','#ff9408','#6a6e09',
	'#8b2e16','#696112','#e17701','#0a481e','#343837','#ffb7ce',
	'#6a79f7','#5d06e9','#3d1c02','#82a67d','#029386','#95d0fc',
	'#be0119','#c9ff27','#373e02','#a9561e','#caa0ff','#ca6641',
	'#02d8e9','#88b378','#980002','#cb0162','#5cac2d','#769958',
	'#a2bffe','#10a674','#06b48b','#af884a','#0b8b87','#ffa756',
	'#a2a415','#154406','#856798','#34013f','#632de9','#0a888a',
	'#6f7632','#d46a7e','#1e488f','#bc13fe','#7ef4cc','#76cd26',
	'#74a662','#80013f','#b1d1fc','#0652ff','#045c5a','#5729ce',
	'#069af3','#ff000d','#f10c45','#5170d7','#acbf69','#6c3461',
	'#5e819d','#601ef9','#b0dd16','#cdfd02','#2c6fbb','#c0737a',
	'#fc5a50','#ffffc2','#7f2b0a','#b04e0f','#a03623','#87ae73',
	'#789b73','#98eff9','#658b38','#5a7d9a','#380835','#fffe7a',
	'#5ca904','#d8dcd6','#a5a502','#d648d7','#047495','#b790d4',
	'#5b7c99','#607c8e','#0b4008','#ed0dd9','#8c000f','#ffff84',
	'#bf9005','#d2bd0a','#ff474c','#0485d1','#ffcfdc','#040273',
	'#a83c09','#90e4c1','#516572','#fac205','#d5b60a','#363737',
	'#4b5d16','#6b8ba4','#80f9ad','#a57e52','#a9f971','#c65102',
	'#e2ca76','#b0ff9d','#9ffeb0','#fdaa48','#fe01b1','#c1f80a',
	'#36013f','#341c02','#b9a281','#8eab12','#9aae07','#02ab2e',
	'#7af9ab','#137e6d','#aaa662','#0343df','#15b01a','#7e1e9c',
	'#610023','#014d4e','#8f1402','#4b006e','#580f41','#8fff9f',
	'#dbb40c','#a2cffe','#c0fb2d','#be03fd','#840000','#d0fefe',
	'#3f9b0b','#01153e','#04d8b2','#c04e01','#0cff0c','#0165fc',
	'#cf6275','#ffd1df','#ceb301','#380282','#aaff32','#53fca1',
	'#8e82fe','#cb416b','#677a04','#ffb07c','#c7fdb5','#ad8150',
	'#ff028d','#000000','#cea2fd','#001146','#0504aa','#e6daa6',
	'#ff796c','#6e750e','#650021','#01ff07','#35063e','#ae7181',
	'#06470c','#13eac9','#00ffff','#e50000','#653700','#ff81c0',
	'#d1b26f','#00035b','#c79fef','#06c2ac','#033500','#9a0eea',
	'#bf77f6','#89fe05','#929591','#75bbfd','#ffff14','#c20078'
	]

random.shuffle(colors)

new_prop_cycle = cycler('color', colors)
plt.rc('axes', prop_cycle=new_prop_cycle)

#TODO Look for Hidden Markov model library, looks suitable for IoT apps.

class CNN(nn.Module):
	def __init__(self, input_size=100, output_size=10, num_features=50):
		super(CNN, self).__init__()
		# 1 input image channel, 6 output channels, 3x1 convolution
		# kernel
		self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 2)
		#self.conv2 = nn.Conv1d(in_channels = 6, out_channels = 16, kernel_size = 2)
		# an affine operation: y = Wx + b
		self.fc1 = nn.Linear(4*(num_features - 1), 20) # out_channels * (num_features - num conv layers)
		self.fc2 = nn.Linear(20, 10)
		#self.do  = nn.Dropout(0.5)
		self.fc3 = nn.Linear(10, output_size)

	def forward(self, x):
		#print('1: {}'.format(x.shape))
		#x = F.max_pool1d(F.relu(self.conv1(x)), 2)
		x = F.relu(self.conv1(x))
		#x = self.do(x)
		#print('2: {}'.format(x.shape))
		#x = F.relu(self.conv2(x))
		#print('3: {}'.format(x.shape)) 
		x = x.view((-1, self.num_flat_features(x)))#x.shape[1]*x.shape[2])) 
		#print('4: {}'.format(x.shape))
		x = F.relu(self.fc1(x))
		#print('5: {}'.format(x.shape))
		x = F.relu(self.fc2(x))
		#print('6: {}'.format(x.shape))
		x = self.fc3(x)
		#print('7: {}'.format(x.shape))
		return F.softmax(x, dim=1)

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

class RNN(nn.Module):
	# dimensions mismatch 
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()
		
		self.hidden_size = hidden_size
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x, hidden):
		combined = torch.cat((x, hidden), 1)
		hidden = self.i2h(combined)

		output = self.i2o(combined)
		output = self.softmax(output)
		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)

class TraceFeaturesDataset(Dataset):
	"""Trace features dataset."""

	def __init__(self, features, labels, transform=None):
		"""
		Args:
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.features = features
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.labels)
		
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		inputs = self.features[idx]
		labels = self.labels[idx]

		sample = {'inputs': inputs, 'labels': labels} 

		if self.transform:
			sample = self.transform(sample)

		return sample

class Quantification(object):
	
	"""
	@classmethod
	def classify_convnn(cls, all_features, all_labels, train_features, train_labels, test_features, test_labels): #train_features, train_labels, test_features, test_labels):
		if len(train_features[0]) < 2:
			print('Padded features')
			for i in range(len(train_features)):
				if type(train_features[i]) == list: 
					train_features[i] += [0 for _ in range(2 - len(train_features[i]))]
				else:
					train_features[i] += tuple([0 for _ in range(2 - len(train_features[i]))])
		#if len(test_features[0]) < 2:
		#	# print('padded up to 3 test features')
		#	for i in range(len(test_features)):
		#		if type(test_features[i]) == list:
		#			test_features[i] += [0 for _ in range(2 - len(test_features[i]))]
		#		else:
		#			test_features[i] += tuple([0 for _ in range(2 - len(test_features[i]))])

		inp_size = len(train_features)
		out_size = len(set(train_labels))
		num_f    = len(train_features[0])
		X  = np.array(train_features)
		X  = X.reshape(X.shape[0], 1, X.shape[1]).astype('float32')
		y  = np.array(train_labels).astype('int64')

		skf = StratifiedKFold(n_splits=5)

		for train_index, test_index in skf.split(X,y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			#test_X   = np.array(test_features)
			#test_X   = test_X.reshape(test_X.shape[0], 1, test_X.shape[1]).astype('float32')
			#test_y   = np.array(test_labels).astype('int64')
			#print(len(train_features), len(train_labels), len(set(train_labels)), len(train_features[0]))
			#print(type(train_features), type(train_features[0]), type(train_features[0,0,0]))
			#Cross Validation Parameter Search:

			#X_train = train_features
			#y_train = train_labels
			#X_test  = test_features
			#y_test  = test_labels

			#asd = grid = GridSearchCV(net = NeuralNetClassifier(
			#	CNN,
			#	module__input_size=inp_size, 
			#	module__output_size=out_size, 
			#	module__num_features=num_f,
			#	max_epochs=250,
			#	#lr=0.01, # 0.005 works for some
			#	#momentum=0.9,
			#	# Shuffle training data on each epoch
			#	iterator_train__shuffle=True,
			#	device='cuda' if torch.cuda.is_available() else 'cpu',
			#	verbose=0),
			#	param_grid = {'lr': [0.01, 0.05, 0.005, 0.002]}, n_jobs = -1, #TODO Add layer size options for layer 1 and 2.
			#	cv=StratifiedKFold(5))
			#	#cv=RepeatedKFold(n_splits=min(5, len(xs)), n_repeats=3))
			#	#cv=ShuffleSplit(n_splits=5, test_size=0.20))
			#grid.fit(xs)
			#lr_ = grid.best_params_['lr']
			#print(lr)

			lr_ = 0.0005
#
			#print('Cuda available' if torch.cuda.is_available() else 'Only Cpu available')
			clf = None
#
			#clf = NeuralNetClassifier(
			#	CNN,
			#	module__input_size = inp_size, 
			#	module__output_size = out_size, 
			#	module__num_features = num_f,
			#	max_epochs = 1000,
			#	optimizer = optim.Adam,
			#	lr=lr_, # 0.005 works for some
			#	#optimizer__momentum=0.9,
			#	# Shuffle training data on each epoch
			#	iterator_train__shuffle=True,
			#	device='cuda' if torch.cuda.is_available() else 'cpu',
			#	verbose=0,
			#	callbacks=[
			#		skorch.callbacks.EarlyStopping(patience=50)]
			#)

			clf.fit(X_train, y_train)
			train_acc = clf.score(X_train, y_train)
			#test_acc = clf.score(X_test, y_test)
			#print('\nTraining Accuracy: {:.2f}, Test Accuracy: {:.2f}'.format(train_acc, test_acc))
			results = clf.predict(X_test)
			accuracy = accuracy_score(y_test, results)
			precision = precision_score(y_test, results, average = 'micro')
			recall = recall_score(y_test, results, average = 'micro')
			f1sc = f1_score(y_test, results, average = 'micro')
 
			print('Train Accuracy {:.2f}, Test Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, Overall F-score: {:.2f}'.format(train_acc, accuracy, precision, recall, f1sc))


		#scores = cross_val_score(cls, X, y, cv=5)
		#print('%0.2f accuracy with a standard deviation of %0.2f' % (scores.mean(), scores.std()))

		#net.fit(train_X, train_y)
		#train_acc = net.score(train_X, train_y)*100
		#test_acc = net.score(test_X, test_y)*100
		#print('\nTraining Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'.format(train_acc, test_acc))
		
		return None





		if len(train_features[0]) < 3:
			print('Padded features')
			for i in range(len(train_features)):
				if type(train_features[i]) == list: 
					train_features[i] += [0 for _ in range(3 - len(train_features[i]))]
				else:
					train_features[i] += tuple([0 for _ in range(3 - len(train_features[i]))])
		dataset = TraceFeaturesDataset(train_features, train_labels)
		trainloader = DataLoader(dataset, batch_size=4,
						shuffle=True, num_workers=4)
		net = CNN(input_size=len(train_features), output_size=len(set(train_labels)), num_features=len(train_features[0]))
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

		num_epochs = 10

		for epoch in range(num_epochs):  # loop over the dataset multiple times
			print('Epoch {}/{}'.format(epoch+1, num_epochs))

			correct = 0
			total = 0
			running_loss = 0.0
			start_time = time.time()
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs = data['inputs']
				labels = data['labels']

				inputs = [x.float() for x in inputs]
				inputs = torch.stack(inputs).t()

				inputs = inputs.view((inputs.shape[0], 1, inputs.shape[1]))
				# padding for lda mismatch issue
				#m = nn.ZeroPad2d(2)
				#inputs = m(inputs)
				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs) # Works (requires fixing the dimensions for now, also we have an issue with maxPool1d where third dimension becomes 1.)
				for j in range(len(labels)):
					total += 1
					if torch.argmax(outputs[j]).item() == labels[j].item():
						correct += 1
				#print('output_shape = {}, outputs = {}'.format(outputs.shape, outputs))
				#print('label_shape = {}, labels = {}'.format(labels.shape, labels))
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				if i % 100 == 99:    # print every 00 mini-batches
					#print('Elapsed time: {}'.format(time.time()-start_time))
					#print('[%d, %5d] loss: %.3f' %
						#(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0
					print('Train accuracy: ', round(correct/total, 3))
					correct = 0
					total = 0
		# test network
		# print('begin testing')
		if len(test_features[0]) < 3:
			# print('padded up to 3 test features')
			for i in range(len(test_features)):
				if type(test_features[i]) == list:
					test_features[i] += [0 for _ in range(3 - len(test_features[i]))]
				else:
					test_features[i] += tuple([0 for _ in range(3 - len(test_features[i]))])
	
		testset = TraceFeaturesDataset(test_features, test_labels)
		testloader = DataLoader(testset, batch_size=4,
						shuffle=True, num_workers=4)

		num_classes = len(set(test_labels))
		# print('num classes (secrets): ',num_classes)
		pred_stats = [[0 for l in range(3)] for m in range(num_classes)] #Containing True Positive, False Positive, False Negative Counts for each class.
		# print('pred stats', pred_stats)
		conf_matrix = [[0 for l in range(num_classes)] for m in range(num_classes)]

		correct = 0
		total = 0
		# print('conf_matrix', conf_matrix)
		with torch.no_grad():
			for i, data in enumerate(testloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs = data['inputs']
				labels_t = data['labels']

				inputs = [x.float() for x in inputs]
				inputs = torch.stack(inputs).t()

				inputs = inputs.view((inputs.shape[0], 1, inputs.shape[1]))
				#labels = torch.stack()
			
				# forward + backward + optimize
				outputs = net(inputs) 
				for j in range(len(labels_t)):
					total += 1
					pred = torch.argmax(outputs[j]).item()
					label = labels_t[j].item()
					conf_matrix[label-1][pred-1] +=  1
					if label == pred: 
						pred_stats[label-1][0] += 1
					else: 
						pred_stats[pred-1][1] += 1
						pred_stats[label-1][2] += 1
					if torch.argmax(outputs[j]).item() == labels_t[j].item():
						correct += 1

		total_precision = 0.0
		total_recall    = 0.0
		for i,l in enumerate(pred_stats):
			precision = 100*float(l[0])/(l[0]+l[1]) if (l[0]+l[1]) > 0 else 0.0
			recall = 100*float(l[0])/(l[0]+l[2]) if (l[0]+l[2]) > 0 else 0.0
			f_score = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

			total_precision += precision
			total_recall += recall
			print('Class {} Precision: {:.2f}%, Recall: {:.2f}%, F-score: {:.2f}'.format(i, precision, recall, f_score))
		total_precision = total_precision/len(pred_stats)
		total_recall    = total_recall/len(pred_stats)
		f_score = 2*(total_precision*total_recall)/(total_precision+total_recall) if (total_precision+total_recall) > 0 else 0.0
		print('Overall Precision: {:.2f}%, Recall: {:.2f}%, Overall F-score: {:.2f}'.format(precision, recall, f_score))

		print('Test Accuracy: ', round(correct/total, 3))

	@classmethod
	def classify_rnn(cls, features, labels, test_features, test_labels):
		# incomplete
		dataset = TraceFeaturesDataset(features, labels)

		trainloader = DataLoader(dataset, batch_size=1,
						shuffle=True, num_workers=4)
		rnn = RNN(input_size=len(features[0]), hidden_size=10, output_size=len(set(labels)))
		criterion = nn.NLLLoss()
		optimizer = optim.SGD(rnn.parameters(), lr=0.001, momentum = .9)

		num_epochs = 10

		for epoch in range(num_epochs):  # loop over the dataset multiple times
			print('Epoch {}/{}'.format(epoch+1, num_epochs))

			correct = 0
			total = 0
			running_loss = 0.0
			start_time = time.time()
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs = data['inputs']
				labels = data['labels']
				inputs = [x.float() for x in inputs]
				inputs = torch.stack(inputs).t()

				inputs = inputs.view((inputs.shape[0], 1, inputs.shape[1]))
				#labels = torch.stack()
				# zero the parameter gradients
				optimizer.zero_grad()
				
				# forward + backward + optimize
				hidden = rnn.initHidden()
				for j in range(inputs.size()[0]):
					outputs, hidden = rnn(inputs[j], hidden)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				for k in range(len(labels)):
					total += 1
					if torch.argmax(outputs[k]).item() == labels[k].item():
						correct += 1
					# print('pred','label',torch.argmax(outputs[k]).item(), labels[k].item())
				#print('output_shape = {}, outputs = {}'.format(outputs.shape, outputs))
				#print('label_shape = {}, labels = {}'.format(labels.shape, labels))

				# print statistics
				running_loss += loss.item()
				if i % 100 == 99:    # print every 00 mini-batches
					#print('Elapsed time: {}'.format(time.time()-start_time))
					#print('[%d, %5d] loss: %.3f' %
						#(epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0
					print('Accuracy: ', round(correct/total, 3))
					correct = 0
					total = 0

		# test RNN
		testset = TraceFeaturesDataset(test_features, test_labels)
		trainloader = DataLoader(testset, batch_size=1,
						shuffle=True, num_workers=4)
		
		num_classes = len(set(test_labels))
		pred_stats = [[0 for l in range(3)] for m in range(num_classes)] #Containing True Positive, False Positive, False Negative Counts for each class.
		conf_matrix = [[0 for l in range(num_classes)] for m in range(num_classes)]
		
		correct = 0
		total = 0
		with torch.no_grad():
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs = data['inputs']
				labels_t = data['labels']

				inputs = [x.float() for x in inputs]
				inputs = torch.stack(inputs).t()

				inputs = inputs.view((inputs.shape[0], 1, inputs.shape[1]))
				#labels = torch.stack()
			
				# forward + backward + optimize
				hidden = rnn.initHidden()
				for j in range(inputs.size()[0]):
					outputs, hidden = rnn(inputs[j], hidden)
				for k in range(len(labels_t)):
					total += 1
					pred = torch.argmax(outputs[k]).item()
					label = labels_t[k].item()
					if pred > num_classes:
						# print('too high:', pred, label)
						continue
					conf_matrix[label-1][pred-1] +=  1
					if label == pred: 
						pred_stats[label-1][0] += 1
					else: 
						pred_stats[pred-1][1] += 1
						pred_stats[label-1][2] += 1
					if torch.argmax(outputs[k]).item() == labels_t[k].item():
						correct += 1

		total_precision = 0.0
		total_recall    = 0.0
		for i,l in enumerate(pred_stats):
			precision = 100*float(l[0])/(l[0]+l[1]) if (l[0]+l[1]) > 0 else 0.0
			recall = 100*float(l[0])/(l[0]+l[2]) if (l[0]+l[2]) > 0 else 0.0
			f_score = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

			total_precision += precision
			total_recall += recall
			print('Class {} Precision: {:.2f}%, Recall: {:.2f}%, F-score: {:.2f}'.format(i, precision, recall, f_score))
		total_precision = total_precision/len(pred_stats)
		total_recall    = total_recall/len(pred_stats)
		f_score = 2*(total_precision*total_recall)/(total_precision+total_recall) if (total_precision+total_recall) > 0 else 0.0
		print('Overall Precision: {:.2f}%, Recall: {:.2f}%, Overall F-score: {:.2f}'.format(precision, recall, f_score))

		print('Test Accuracy: ', round(correct/total, 3))
	"""

	@classmethod
	def reduce_dim_lda(cls, features, labels, n_components=1, solver='svd'):
		"""
			Reduces the number of features in our data matrix to fewer number 
			of features using LDA (Linear Discriminant Analysis).
			Returns an array with dimensions (n_samples*n_components) which is
			the data matrix with fewer features.

			features is the list of list of features where each n'th list 
			contains features of n'th interaction.
			labels is a list of integers where each n'th value corresponds to 
			the secret(class for classifiers) of n'th interaction.
			n_components is the number of unique features after feature 
			reduction.
			solver is the solver used for matrix computations.
			'svd' (Singular value decomposition) is recommended for large 
			number of features.
			'lsqr' (Least squares solution) is another efficient method.
			'eigen' (Eigenvalue decomposition) is another choice but not 
			very efficient.
		"""
		assert len(features) == len(labels)

		np_features = np.array(features)
		np_labels = np.array(labels)
		#np_test_features = np.array(test_features)

		lda = LinearDiscriminantAnalysis(n_components=n_components,solver=solver)
		lda = lda.fit(np_features,np_labels)
		
		rd_features = lda.transform(np_features)
		#rd_test_features  = lda.transform(np_test_features)

		return rd_features.tolist()

	@classmethod
	def reduce_dim_lda_new(cls, train_features, train_labels, test_features, 
		n_components=1,solver='svd',padding=False):
		"""
			Reduces the number of features in our data matrix to fewer number 
			of features using LDA (Linear Discriminant Analysis).
			Returns an array with dimensions (n_samples*n_components) which is
			 the data matrix with fewer features.

			features is the list of list of features where each n'th list 
			contains features of n'th interaction.
			labels is a list of integers where each n'th value corresponds
			 to the secret(class for classifiers) of n'th interaction.
			n_components is the number of unique features after feature 
			reduction.
			solver is the solver used for matrix computations.
			'svd' (Singular value decomposition) is recommended for large number of features.
			'lsqr' (Least squares solution) is another efficient method.
			'eigen' (Eigenvalue decomposition) is another choice but not very efficient.
		"""
		assert len(train_features) == len(train_labels)
		max_val = len(set(train_labels)) - 1
		if n_components > max_val:
			n_components = max_val
			print('Maximum reduction: {} features'.format(n_components))
		if n_components > len(train_features[0]) and padding:
			for i in range(len(train_features)):
				train_features[i] += tuple(np.zeros(n_components - len(train_features[i]) + 1))
			for i in range(len(test_features)):
				test_features[i] += tuple(np.zeros(n_components - len(test_features[i]) + 1))

		# if n_components >= len(set(train_labels)):
		# 	for i in range(len(set(train_labels)), n_components + 1):
		# 		train_labels.append(i)
		# 		train_features.append(tuple(np.zeros(len(train_features[0]))))

		np_train_features = np.array(train_features)
		np_train_labels   = np.array(train_labels)
		np_test_features  = np.array(test_features)

		lda = LinearDiscriminantAnalysis(n_components=n_components, solver=solver)
		lda = lda.fit(np_train_features, np_train_labels)
		
		rd_train_features = lda.transform(np_train_features)
		rd_test_features  = lda.transform(np_test_features)
		return (rd_train_features.tolist(), rd_test_features.tolist())

	@classmethod
	def reduce_dim_pca_new(cls, train_features, test_features, n_components=2, 
		padding=False):
		"""
			Reduces the number of features in our data matrix to fewer number of features using PCA (Principal Component Analysis).
			Returns an array with dimensions (n_samples*n_components) which is the data matrix with fewer features.

			features is the list of list of features where each n'th list contains features of n'th interaction.
			n_components is the number of unique features after feature reduction.
		"""
		# print('len features: ', len(train_features))
		# print('features 0: ', train_features[0])
		# print('n_components: ', n_components)
		#if n_components > len(train_features[0]) and padding:
		#	for i in range(len(train_features)):
		#	 	train_features[i] += tuple(np.zeros(n_components - len(train_features[i]) + 1))
		#	for i in range(len(test_features)):
		#		test_features[i] += tuple(np.zeros(n_components - len(test_features[i]) + 1))
		# if n_components > len(set(train_labels)):
		# 	print('yes')

		np_train_features = np.array(train_features)
		np_test_features  = np.array(test_features)

		pca = PCA(svd_solver='auto') #n_components=n_components,
		pca = pca.fit(np_train_features)
		rd_train_features = pca.transform(np_train_features)
		rd_test_features  = pca.transform(np_test_features)

		return (rd_train_features.tolist(), rd_test_features.tolist())
	
	@classmethod
	def reduce_dim_pca(cls, features, n_components=2):
		"""
			Reduces the number of features in our data matrix to fewer number of features using PCA (Principal Component Analysis).
			Returns an array with dimensions (n_samples*n_components) which is the data matrix with fewer features.

			features is the list of list of features where each n'th list contains features of n'th interaction.
			n_components is the number of unique features after feature reduction.
		"""

		np_features = np.array(features)
		
		pca = PCA(n_components=n_components,svd_solver='auto')
		pca = pca.fit(np_features)
		
		rd_features = pca.transform(np_features)
		#rd_test_features  = pca.transform(np_test_features)
		
		return rd_features.tolist()

	@classmethod
	def estimate_entropy_hist(cls, features, labels, tag='', bin_size=1.0, pprint=False, plot=False):
		"""
		This method quantifies the information leakage by modeling the data distribution with a histogram.

		Args:
			features: List of feature values
			labels: List of labels corresponding to each feature
			tag: Name of the feature for plot and printing purposes
			bin_size: Size of the histogram bins for probability estimation
			pprint: This argument sets whether to print the results or not
			plot: This argument sets whether the distributions that are estimated will be plotted.

		Returns:
			The a-priori entropy, a-posteriori entropy and the leakage entropy (the difference)
		"""

		secrets = set(labels)
		priori_entropy = np.log2(len(secrets)) # Uniform distribution

		def myround(x, base=.05, prec=20):
			return round(base * round(float(x)/base),prec)

		#Discretization for time, this should not affect space as
		# this rounds only decimals
		features = [myround(x,bin_size) for x in features]

		###################################################
		#This will be calculating discrete entropy
		posteriori_entropy = 0.0

		data    = sorted(zip(features, labels), key=lambda tup: tup[1])
		counter = collections.Counter(features)
		counter_data = collections.Counter(data)

		N = len(features)
		for (x,n) in counter.items():
			p_x = float(n)/N
			p_yx_total = 0.0
			for y in secrets:
				num_y = counter_data[(x,y)]
				if num_y == 0:
					continue
				else:
					p_yx = float(num_y)/float(n)
					p_yx_total += -p_yx * np.log2(p_yx)
			posteriori_entropy += p_x * p_yx_total

		leakage_entropy = priori_entropy - posteriori_entropy

		#Plotting for the histogram
		if plot:
			secret_point_list = [[] for _ in range(len(secrets))]
			for (i,j) in data:
				secret_point_list[j].append(i)

			min_features = float(min(features))
			max_features = float(max(features))

			bins = np.linspace(min_features, max_features, 100)

			fig, ax = plt.subplots()
			for i in range(len(secrets)):
				ax.hist(secret_point_list[i], bins, alpha=0.5, color=colors[i%len(colors)])

			fig.savefig(tag + '-hist.png', dpi=600)
			plt.close(fig)

		if pprint:
			print('For feature {0}:'.format(tag))
			print('Number of secrets                        : {0}'.format(len(secrets)))
			print('A-priori Entropy                         : {0} bits'.format(priori_entropy))
			print('A-posteriori Entropy                     : {0} bits'.format(posteriori_entropy))
			print('Leakage Entropy (a-priori - a-posteriori): {0} bits'.format(leakage_entropy))
			print('-'*80)

		return (priori_entropy, posteriori_entropy, leakage_entropy)

	@classmethod
	def estimate_entropy_gmm(cls, features, labels, tag='', max_n_comps=10, pprint=False, plot=False):
		"""
		This method quantifies the information leakage by modeling the data distribution with a Gaussian Mixture Model.

		Args:
			features: List of feature values
			labels: List of labels corresponding to each feature
			tag: Name of the feature for plot and printing purposes
			max_n_comps: Maximum number of Gaussian distributions for the mixture model
			pprint: This argument sets whether to print the results or not
			plot: This argument sets whether the distributions that are estimated will be plotted.

		Returns:
			The a-priori entropy, a-posteriori entropy and the leakage entropy (the difference)
		"""
		#Definitions
		min_features = float(min(features))
		max_features = float(max(features))
		tail_percent = 0.05
		epsilon      = np.power(10.0,-20)

		#bdwidth_preset = False

		#if bdwidth is not None:
		#	bdwidth_preset = True
			#print('Bandwidth preset to {}'.format(bdwidth))

		rang = max_features - min_features
		min_lim = np.round(min_features - rang*tail_percent, 5)
		max_lim = np.round(max_features + rang*tail_percent, 5)

		secrets = set(labels)
		priori_entropy = np.log2(len(secrets)) # Assuming uniform distribution over secrets

		data = sorted(zip(features, labels), key=lambda tup: tup[1])

		#Sampling points for probability estimation
		sample_points = np.linspace(min_lim, max_lim, 10000)

		#Points for p(x|y), p(y) is assumed discrete uniform (1/N)
		secret_point_list = [[] for _ in range(len(secrets))]

		for (i,j) in data:
			secret_point_list[j].append(i)

		p_y_lists = [[] for _ in range(len(secrets))]

		for i in range(len(secrets)): # kernel is epanechnikov, a narrower distribution.
			xs = np.array(secret_point_list[i])
			xs = xs.reshape(-1, 1)

			min_n_comps = len(list(set(secret_point_list[i])))

			gmm = BayesianGaussianMixture(n_components=min(min_n_comps,max_n_comps),
				max_iter=5000, n_init=5).fit(xs)
				#, covariance_type='spherical'
				#weight_concentration_prior_type='dirichlet_distribution'
			p_y_lists[i] = gmm.score_samples(np.array(sample_points)[:,np.newaxis])
			p_y_lists[i] = normalize(np.exp(p_y_lists[i])[:,np.newaxis],norm='l1', axis=0)

		#Kernel Density
		posteriori_entropy = 0.0
		p_y = 1.0/len(secrets)
		for (i,_) in enumerate(sample_points):
			sample_ent = 0.0
			p_x = 0.0
			for j in range(len(secrets)):
				p_xy = p_y_lists[j][i][0]
				p_x_y = p_xy*p_y
				p_x += p_x_y
				p_x_y = epsilon if p_x_y < epsilon else p_x_y #To prevent log0 errors
				sample_ent -= p_x_y * np.log2(p_x_y)
			p_x = epsilon if p_x < epsilon else p_x #To prevent log0 errors
			sample_ent += p_x * np.log2(p_x)
			posteriori_entropy += sample_ent

		if plot:
			#Take the tag, plot the points to a figure
			#Write to a file with tag.
			fig, ax = plt.subplots()
			for (i,p) in enumerate(p_y_lists):
				ax.plot(sample_points, p, '-', label='secret = \'{0}\''.format(i), color=colors[i%len(colors)])
				ax.plot(secret_point_list[i], -0.0005 - 0.002 * np.random.random(len(secret_point_list[i])),'+',color=colors[i%len(colors)])
			ax.set_xlim(min_lim, max_lim)
			ax.set_ylim(-0.0025, 0.005)
			#print(tag)
			fig.savefig(tag + '-gmm.png',dpi=600)
			plt.close(fig)
			#with open(tag+'.txt', 'w') as f:
			#	f.write('{}\n{}\n{}\n'.format(secret_list, means_per_secret, variances_per_secret))

		leakage_entropy = priori_entropy - posteriori_entropy

		if pprint:
			print('For feature {0}:'.format(tag))
			print('Number of secrets                        : {0}'.format(len(secrets)))
			print('A-priori Entropy                         : {0} bits'.format(priori_entropy))
			print('A-posteriori Entropy                     : {0} bits'.format(posteriori_entropy))
			print('Leakage Entropy (a-priori - a-posteriori): {0} bits'.format(leakage_entropy))
			print('-'*80)

		return (priori_entropy, posteriori_entropy, leakage_entropy)

	@classmethod
	def estimate_entropy_kde(cls, features, labels, tag='', bdwidth=None, pprint=False, plot=False, option='dynamic', option2='space', DEBUG=False):
		"""
		This method quantifies the information leakage by modeling the data distribution using Kernel Density Estimation.

		Args:
			features: List of feature values.
			labels: List of labels corresponding to each feature.
			tag: Name of the feature for plot and printing purposes.
			bdwidth: The bandwidth value for the kernel, used if option argument is 'fixed'.
			pprint: This argument sets whether to print the results or not.
			plot: This argument sets whether the distributions that are estimated will be plotted.
			option: This argument can be set to 'dynamic', 'stddev', or 'fixed'. These options determine the bandwidth selection method.
				If 'dynamic' is set, this method uses cross validation to search and determine the bandwidth value.
				If 'stddev' is set, the ideal bandwidth assuming the distribution is Gaussian or close to Gausssian is used.
				If 'fixed' is set, the bandwidth value set by the user is used.
			option2: This argument can be set to 'space' or 'time'.
			They are used to select the bandwidth search candidates.
			They are also used in default bandwidth is option is 'fixed' and bdwidth is None.

		Returns:
			The a-priori entropy, a-posteriori entropy and the leakage entropy (the difference).
		"""
		warnings.filterwarnings("ignore")
		if not(option=='dynamic' or option=='stddev' or option=='fixed'):
			print('Option argument needs to be dynamic, stddev, or fixed.')
			return None

		if option=='fixed' and bdwidth is None:
			print('If the kde option is fixed, you need to provide a bandwidth size, bandwidth argument is None.')
			return None

		if not(option2 in ['space', 'time']):
			print('Option2 needs to be set as space or time.')

		#Definitions for Kernel Density Estimation
		min_features = float(min(features))
		max_features = float(max(features))
		tail_percent = 0.05
		epsilon      = np.power(10.0,-10)

		#bdwidth_preset = False

		#if bdwidth is not None:
		#	bdwidth_preset = True
			#print('Bandwidth preset to {}'.format(bdwidth))

		rang = max_features - min_features
		min_lim = np.round(min_features - rang*tail_percent, 8)
		max_lim = np.round(max_features + rang*tail_percent, 8)

		secrets = set(labels)
		#secrets = sorted(secrets)
		priori_entropy = np.log2(len(secrets)) # Assuming uniform distribution over secrets

		data = sorted(zip(features, labels), key=lambda tup: tup[1])

		#Sampling points for probability estimation
		sample_points = np.linspace(min_lim, max_lim, 100000)
		#sample_points = np.array(list(sample_points))# + features) #[:10])

		#Points for p(x|y), p(y) is assumed discrete uniform (1/N)
		secret_point_list = [[] for _ in range(len(secrets))]

		for (i,j) in data:
			secret_point_list[j].append(i)

		p_y_lists = [[] for _ in range(len(secrets))]
		#p_y_lists_features = [[] for _ in range(len(secrets))]

		for i in range(len(secrets)): # kernel is epanechnikov, a narrower distribution.
			xs = np.array(secret_point_list[i])
			xs = xs.reshape(-1, 1)

			max_val = max(secret_point_list[i]) - min(secret_point_list[i])
			#print('Range for feature {} and secret {}: {}'.format(tag, i, max_val))

			if option == 'stddev':
				std_bdwidth = 1.06 * np.std(secret_point_list[i]) * np.power(1.0/float(len(secret_point_list[i])), 0.2)
				bdwidth = std_bdwidth
				#print(std_bdwidth)
				#print('bandwidth: {}'.format(bdwidth))
				#print('StandardDev: {}'.format(np.std(secret_point_list[i])))
				#print('Float coeff: {}'.format(np.float_power(1.0/float(len(secret_point_list[i])), 0.2)))

				if option2 == 'space' and bdwidth <= 0.0:
					bdwidth = 0.1
				if option2 == 'time' and bdwidth <= 0.0:
					bdwidth = 0.00001
			elif option == 'dynamic':
				std_bdwidth = 1.06 * np.std(secret_point_list[i]) * np.power(1.0/float(len(secret_point_list[i])), 0.2)
				if max_val == 0.0 and option2 == 'space':
					max_val = 100.0
				if max_val == 0.0 and option2 == 'time':
					max_val = 0.1
				if option2 == 'space':
					bdwidth = 0.1
					bandwidths = np.linspace(1.0, max_val, 10)
					bandwidths = np.array(list(bandwidths) + [std_bdwidth, 0.1, 1.0, 5.0, 10.0, 20.0])
				else:
					bdwidth = 0.00001
					bandwidths = np.linspace(0.00001, max_val, 10)
					bandwidths = np.array(list(bandwidths) + [std_bdwidth, 0.001, 0.01, 0.1, 1.0, 5.0])

				if len(xs) > 1:
					#print('Running grid search with 3 times repeated 5-fold cross-validation, parallelized')
					grid = GridSearchCV(KernelDensity(kernel='epanechnikov'),
						param_grid={'bandwidth': bandwidths}, n_jobs = -1,
						cv=KFold(n_splits = min(5, len(xs)), shuffle=True))
						#cv=RepeatedKFold(n_splits=min(5, len(xs)), n_repeats=3))
						#cv=ShuffleSplit(n_splits=5, test_size=0.20))
					grid.fit(xs)
					bdwidth = grid.best_params_['bandwidth']
				if bdwidth <= 0.0:
					if option2 == 'space':
						bdwidth = 0.1
					else:
						bdwidth = 0.00001
			elif option == 'fixed':
				bdwidth = bdwidth

			if bdwidth < 0.000000001:
				bdwidth = 0.000000001

			#print('BANDWIDTH:',bdwidth)

			#print('Bdwidth for {}:{} set to {}, options: {} {}'.format(tag, i, bdwidth, option, option2))

			kde = KernelDensity(kernel='epanechnikov', bandwidth=float(bdwidth)).fit(xs)
			#np.array(secret_point_list[i])[:,np.newaxis])
			#kde = grid.best_estimator_
			#sample_points = np.array(list(sample_points) + secret_point_list[i])

			p_y_lists[i] = kde.score_samples(np.array(sample_points)[:,np.newaxis])
			p_y_lists[i] = normalize(np.exp(p_y_lists[i])[:,np.newaxis],norm='l1', axis=0)

			#p_y_lists_features[i] = kde.score_samples(np.array(features)[:, np.newaxis])
			#p_y_lists_features[i] = np.exp(p_y_lists_features[i])[:, np.newaxis]

		#Kernel Density
		posteriori_entropy = 0.0
		p_y = 1.0/len(secrets)
		for (i,_) in enumerate(sample_points):
			sample_ent = 0.0
			p_x = 0.0
			for j in range(len(secrets)):
				p_xy = p_y_lists[j][i][0] # p(x|y)
				p_x_y = p_xy*p_y # p(x,y) = p(x|y)*p(y)
				p_x += p_x_y # p(x) = sum_{y} p(x,y)
				p_x_y = epsilon if p_x_y < epsilon else p_x_y #To prevent log0 errors
				sample_ent -= p_x_y * np.log2(p_x_y) # -sum_y p(x,y) log2(p(x,y)) ??
			p_x = epsilon if p_x < epsilon else p_x #To prevent log0 errors
			sample_ent += p_x * np.log2(p_x) # sum_x p(x) log2(p(x))
			posteriori_entropy += sample_ent

		if plot:
			#Take the tag, plot the points to a figure
			#Write to a file with tag.
			fig, ax = plt.subplots()
			for (i,p) in enumerate(p_y_lists):
				ax.plot(sample_points, p, '-', label='secret = \'{0}\''.format(i), color=colors[i%len(colors)])
				ax.plot(secret_point_list[i], -0.0005 - 0.002 * np.random.random(len(secret_point_list[i])),'+',color=colors[i%len(colors)])
			ax.set_xlim(min_lim, max_lim)
			ax.set_ylim(-0.0025, 0.005)
			#print(tag)
			fig.savefig(tag + '-kde-{}.png'.format(option),dpi=600)
			plt.close(fig)
			#with open(tag+'.txt', 'w') as f:
			#	f.write('{}\n{}\n{}\n'.format(secret_list, means_per_secret, variances_per_secret))

		#y_pred = []
		#for j,f in enumerate(features):
		#	prediction = [p_y_lists_features[i][j][0] for i in range(len(secrets))]
		#	#for i in range(secrets):
		#	#	prediction[i] = p_y_lists_features[i][j][0]
		#	total = sum(prediction)
		#	prediction = [x/total for x in prediction]
		#	y_pred.append(prediction)

		#loss = log_loss(y_true = labels, y_pred = y_pred, labels = secrets)

		leakage_entropy = priori_entropy - posteriori_entropy

		if pprint:
			print('For feature {0}:'.format(tag))
			print('Number of secrets                        : {0}'.format(len(secrets)))
			print('A-priori Entropy                         : {0} bits'.format(priori_entropy))
			print('A-posteriori Entropy                     : {0} bits'.format(posteriori_entropy))
			print('Leakage Entropy (a-priori - a-posteriori): {0} bits'.format(leakage_entropy))
			#print('Loss                                     : {0}'.format(loss))
			print('-'*80)

		return (priori_entropy, posteriori_entropy, leakage_entropy)

	@classmethod
	def estimate_entropy_nn(cls, features, labels, num_pts=10000.0, pprint=False):
		#Definitions for Nearest Neighbor Estimation
		min_features = float(min(features))
		max_features = float(max(features))
		tail_percent = 0.05
		nn_k_value   = 9
		epsilon      = np.power(10.0,-10)

		rang = max_features - min_features
		min_lim = np.round(min_features - rang*tail_percent, 5)
		max_lim = np.round(max_features + rang*tail_percent, 5)

		window_size = (max_lim + 1.0 - min_lim)/num_pts

		secrets = set(labels)
		priori_entropy = np.log2(len(secrets)) # Uniform distribution

		data = sorted(zip(features, labels), key=lambda tup: tup[1])

		#Sampling points for probability estimation
		sample_points = np.arange(min_lim, max_lim+1.0, window_size)

		#Points for p(x|y), p(y) is assumed discrete uniform (1/N)
		secret_point_list = [[] for _ in range(len(secrets))]

		for (i,j) in data:
			#print((i,j))
			secret_point_list[j].append(i)

		def find_lt(a, x):
			'Find rightmost value less than x'
			i = bisect.bisect_left(a, x)
			return i-1

		#Find k-nearest neighbors of p which is greater than l[i]
		#l is list, p is number, i is the point where p>l[i] and p<l[i+1]
		#k is the nearest neighbor window size.
		def nn_ind(l, p, i, k):
			li = max(0, i-k)
			hi = min(len(l),i+k)
			sub_list = l #[li:hi]
			#Calculate & sort the distances.
			dist_list = sorted([np.abs(x-p) for x in sub_list])
			#Take the smallest k distances as sum.
			total = sum(dist_list[0:min(k,len(dist_list))])
			return total

		def nn_list(samples, l):
			ind_samples = 0
			ind_list = 0
			p_list = [0.0] * len(samples)
			for (i,p) in enumerate(samples):
				while ind_list < len(l) and p > l[ind_list]:
					ind_list += 1
				dist = nn_ind(l, p, ind_list, nn_k_value)
				p_list[i] = 1/(dist+epsilon)
			norm_p = sum(p_list)
			p_list =[x/norm_p for x in p_list]
			return p_list

		p_y_lists = [[] for _ in range(len(secrets))]
		for i in range(len(secrets)):
			p_y_lists[i] = nn_list(sample_points, secret_point_list[i])

		posteriori_entropy = 0.0
		p_y = 1.0/len(secrets)
		for (i,_) in enumerate(sample_points):
			sample_ent = 0.0
			p_x = 0.0
			for j in range(len(secrets)):
				p_x_y = p_y_lists[j][i]*p_y
				p_x += p_x_y
				p_x_y = epsilon if p_x_y < epsilon else p_x_y #To prevent log0 errors
				sample_ent -= p_x_y * np.log2(p_x_y)
			p_x = epsilon if p_x < epsilon else p_x #To prevent log0 errors
			sample_ent += p_x * np.log2(p_x)
			posteriori_entropy += sample_ent

		leakage_entropy = priori_entropy - posteriori_entropy

		if pprint:
			tag = ''
			print('For feature {0}:'.format(tag))
			print('Number of secrets                        : {0}'.format(len(secrets)))
			print('A-priori Entropy                         : {0} bits'.format(priori_entropy))
			print('A-posteriori Entropy                     : {0} bits'.format(posteriori_entropy))
			print('Leakage Entropy (a-priori - a-posteriori): {0} bits'.format(leakage_entropy))
			print('-'*80)

		return (priori_entropy, posteriori_entropy, leakage_entropy)

	@classmethod
	def estimate_entropy_norm(cls, features, labels, tag='', min_var=0.001, pprint=False, plot=False):
		#Definitions for Nearest Neighbor Estimation
		min_features = float(min(features))
		max_features = float(max(features))
		tail_percent = 0.05
		epsilon      = np.power(10.0,-50)

		rang = max_features - min_features
		min_lim = np.round(min_features - rang*tail_percent, 5)
		max_lim = np.round(max_features + rang*tail_percent, 5)

		secrets = set(labels)
		priori_entropy = np.log2(len(secrets)) # Uniform distribution

		data = sorted(zip(features, labels), key=lambda tup: tup[1])

		#Sampling points for probability estimation
		sample_points = np.linspace(min_lim, max_lim, 25000) #[:, np.newaxis]

		#Points for p(x|y), p(y) is assumed discrete uniform (1/N)
		secret_point_list = [[] for _ in range(len(secrets))]

		for (i,j) in data:
			secret_point_list[j].append(i)

		secret_list = sorted(list(secrets))
		means_per_secret = []
		variances_per_secret = []

		p_y_lists = [[] for _ in range(len(secrets))]
		for i in range(len(secrets)):
			#Gaussian Fitting
			#print(secret_point_list[i])
			#print(sample_points)
			(loc, scale) = norm.fit(np.array(secret_point_list[i]))

			means_per_secret.append(loc)
			variances_per_secret.append(scale)
			#secret_list.append(secrets[i])

			if scale < min_var:
				scale = min_var
			p_y_lists[i] = norm.pdf(sample_points,loc=loc,scale=scale)
			#print(p_y_lists[i])
			p_y_lists[i] = normalize(p_y_lists[i][:,np.newaxis],norm='l1', axis=0)

		if plot:
			#Take the tag, plot the points to a figure
			#Write to a file with tag.
			fig, ax = plt.subplots()
			for (i,p) in enumerate(p_y_lists):
				ax.plot(sample_points, p, '-', label='secret = \'{0}\''.format(i), color=colors[i%20])
				ax.plot(secret_point_list[i], -0.0005 - 0.002 * np.random.random(len(secret_point_list[i])),'+',color=colors[i%len(colors)])
			ax.set_xlim(min_lim, max_lim)
			ax.set_ylim(-0.0025, 0.005)
			#print(tag)
			fig.savefig(tag + '-norm.png',dpi=600)
			plt.close(fig)
			#with open(tag+'.txt', 'w') as f:
			#	f.write('{}\n{}\n{}\n'.format(secret_list, means_per_secret, variances_per_secret))

		posteriori_entropy = 0.0
		p_y = 1.0/len(secrets)
		for (i,_) in enumerate(sample_points):
			sample_ent = 0.0
			p_x = 0.0
			for j in range(len(secrets)):
				p_xy = p_y_lists[j][i][0] # p(x|y)
				p_x_y = p_xy*p_y # p(x,y) = p(x|y)*p(y)
				p_x += p_x_y # p(x) = sum_{y} p(x,y)
				p_x_y = epsilon if p_x_y < epsilon else p_x_y #To prevent log0 errors
				sample_ent -= p_x_y * np.log2(p_x_y) #Entropy sum per x,y pair, adding -p(x,y)*log2(p(x,y))
			p_x = epsilon if p_x < epsilon else p_x #To prevent log0 errors
			sample_ent += p_x * np.log2(p_x) #Removing p(x)*log(p(x))
			posteriori_entropy += sample_ent

		leakage_entropy = priori_entropy - posteriori_entropy

		if pprint:
			print('For feature {0}:'.format(tag))
			print('Number of secrets                        : {0}'.format(len(secrets)))
			print('A-priori Entropy                         : {0} bits'.format(priori_entropy))
			print('A-posteriori Entropy                     : {0} bits'.format(posteriori_entropy))
			print('Leakage Entropy (a-priori - a-posteriori): {0} bits'.format(leakage_entropy))
			print('-'*80)

		return (priori_entropy, posteriori_entropy, leakage_entropy)

	@classmethod
	def estimate_entropy_kde_multidimensional(cls, features, labels, tag='', bdwidth=None, pprint=True, option='dynamic', option2='space', num_dimensions=2):
		"""
		This method quantifies the information leakage of combined features.
		The method takes a list of data points, reduces the number of features to K using LDA, and quantifies the information leakage.
		One problem is that reducing the number of features could reduce variance and leakage.


		Args:
			features: List of list of feature values.
			labels: List of labels corresponding to each feature.
			tag: Name of the feature for plot and printing purposes.
			bdwidth: The bandwidth value for the kernel, used if option argument is 'fixed'.
			pprint: This argument sets whether to print the results or not.
			plot: This argument sets whether the distributions that are estimated will be plotted.
			option: This argument can be set to 'dynamic', 'stddev', or 'fixed'. These options determine the bandwidth selection method.
				If 'dynamic' is set, this method uses cross validation to search and determine the bandwidth value.
				If 'stddev' is set, the ideal bandwidth assuming the distribution is Gaussian or close to Gausssian is used.
				If 'fixed' is set, the bandwidth value set by the user is used.
			option2: This argument can be set to 'space' or 'time'.
			They are used to select the bandwidth search candidates.
			They are also used in default bandwidth is option is 'fixed' and bdwidth is None.

		Returns:
			The a-priori entropy, a-posteriori entropy and the leakage entropy (the difference).
		"""
		warnings.filterwarnings("ignore")
		if not(option=='dynamic' or option=='stddev' or option=='fixed'):
			print('Option argument needs to be dynamic, stddev, or fixed.')
			return None

		if option=='fixed' and bdwidth is None:
			print('If the kde option is fixed, you need to provide a bandwidth size, bandwidth argument is None.')
			return None

		if not(option2 in ['space', 'time']):
			print('Option2 needs to be set as space or time.')

		print("LenFEATURES", len(features))
		print("LenFEATURES_K", len(features[0]))
		print("LenLABELS", len(labels))

		new_features = cls.reduce_dim_lda(features, labels, n_components=num_dimensions)

		print("LenNEWFEATURES", len(new_features))
		print("LenNEWFEATURES_K", len(new_features[0]))

		#Definitions for Kernel Density Estimation
		feature_range_list = []
		for i in range(num_dimensions):
			feature_list = [el[i] for el in new_features]
			range_tuple = (float(min(feature_list)), float(max(feature_list)))
			feature_range_list.append(range_tuple)
		epsilon     = np.power(10.0,-10)
		num_samples = 100000
		sample_points = [None for _ in range(num_samples)]
		for i in range(num_samples):
			sample_points[i] = [random.uniform(minval-epsilon, maxval+epsilon) for (minval, maxval) in feature_range_list]

		#Computing priori entropy
		secrets = set(labels)
		priori_entropy = np.log2(len(secrets)) # Assuming uniform distribution over secrets
		data = sorted(zip(new_features, labels), key=lambda tup: tup[1])

		#Sampling points for probability estimation
		
		#Points for p(x|y), p(y) is assumed discrete uniform (1/N)
		secret_point_list = [[] for _ in range(len(secrets))]

		for (i,j) in data:
			secret_point_list[j].append(i)

		p_y_lists = [[None for _ in range(num_dimensions)] for _ in range(len(secrets))]
		#p_y_lists_features = [[] for _ in range(len(secrets))]

		for i in range(len(secrets)): # kernel is epanechnikov, a narrower distribution.
			for j in range(num_dimensions): #Finding p(x_j|i) for each i and j
				xs = np.array(secret_point_list[i][j])
				xs = xs.reshape(-1, 1)

				max_val = max(secret_point_list[i][j]) - min(secret_point_list[i][j])
				#print('Range for feature {} and secret {}: {}'.format(tag, i, max_val))

				if option == 'stddev':
					std_bdwidth = 1.06 * np.std(secret_point_list[i][j]) * np.power(1.0/float(len(secret_point_list[i][j])), 0.2)
					bdwidth = std_bdwidth

					if option2 == 'space' and bdwidth <= 0.0:
						bdwidth = 0.1
					if option2 == 'time' and bdwidth <= 0.0:
						bdwidth = 0.00001
				elif option == 'dynamic':
					std_bdwidth = 1.06 * np.std(secret_point_list[i][j]) * np.power(1.0/float(len(secret_point_list[i][j])), 0.2)
					if max_val == 0.0 and option2 == 'space':
						max_val = 100.0
					if max_val == 0.0 and option2 == 'time':
						max_val = 0.1
					if option2 == 'space':
						bdwidth = 0.1
						bandwidths = np.linspace(1.0, max_val, 10)
						bandwidths = np.array(list(bandwidths) + [std_bdwidth, 0.1, 1.0, 5.0, 10.0, 20.0])
					else:
						bdwidth = 0.00001
						bandwidths = np.linspace(0.00001, max_val, 10)
						bandwidths = np.array(list(bandwidths) + [std_bdwidth, 0.001, 0.01, 0.1, 1.0, 5.0])

					if len(xs) > 1:
						#print('Running grid search with 3 times repeated 5-fold cross-validation, parallelized')
						grid = GridSearchCV(KernelDensity(kernel='epanechnikov'),
							param_grid={'bandwidth': bandwidths}, n_jobs = -1,
							cv=KFold(n_splits = min(5, len(xs)), shuffle=True))
							#cv=RepeatedKFold(n_splits=min(5, len(xs)), n_repeats=3))
							#cv=ShuffleSplit(n_splits=5, test_size=0.20))
						grid.fit(xs)
						bdwidth = grid.best_params_['bandwidth']
					if bdwidth <= 0.0:
						if option2 == 'space':
							bdwidth = 0.1
						else:
							bdwidth = 0.00001

				if bdwidth < 0.000000001:
					bdwidth = 0.000000001

				#print('BANDWIDTH:',bdwidth)

				#print('Bdwidth for {}:{} set to {}, options: {} {}'.format(tag, i, bdwidth, option, option2))

				kde = KernelDensity(kernel='epanechnikov', bandwidth=float(bdwidth)).fit(xs)
				#np.array(secret_point_list[i])[:,np.newaxis])
				#kde = grid.best_estimator_
				#sample_points = np.array(list(sample_points) + secret_point_list[i])

				p_y_lists[i][j] = kde.score_samples(np.array(sample_points[j])[:,np.newaxis])
				p_y_lists[i][j] = normalize(np.exp(p_y_lists[i][j])[:,np.newaxis],norm='l1', axis=0)

				#p_y_lists_features[i] = kde.score_samples(np.array(features)[:, np.newaxis])
				#p_y_lists_features[i] = np.exp(p_y_lists_features[i])[:, np.newaxis]

		#Entropy Calculation, seems correct
		posteriori_entropy = 0.0
		p_y = 1.0/len(secrets)
		for (i,_) in enumerate(sample_points):
			sample_ent = 0.0
			p_x = [0.0 for _ in range(num_dimensions)]
			for j in range(len(secrets)):
				p_x_y = p_y
				for k in range(num_dimensions):
					p_xy = p_y_lists[j][i][k] # p(x_k|y=j)
					p_x_y = p_x_y * p_xy # p(x1,x2,y=j) = p(x1|y)*p(x2|y)*p(y)
					p_x_k_y = p_x_y * p_y # p(x_k,y=j) = p(x_k|y=j)p(y=j)

					p_x[k] += p_x_k_y # p(x_k) = sum_{y} p(x_k,y)
					p_x_y = epsilon if p_x_y < epsilon else p_x_y #To prevent log0 errors
				sample_ent -= p_x_y *np.log2(p_x_y) #-sum_y p(x1,x2,y) log2(p(x1,x2,y))
			p_x_joint = 1.0 #p(x1,x2) = p(x1)p(x2) #Assuming independence in this case
			for k in range(num_dimensions):
				p_x[k] = epsilon if p_x[k] < epsilon else p_x[k] #To prevent log0 errors
				p_x_joint = p_x_joint*p_x[k]
			sample_ent += p_x_joint * np.log2(p_x_joint) # sum_x p(x1,x2) log2(p(x1,x2))
			posteriori_entropy += sample_ent

		leakage_entropy = priori_entropy - posteriori_entropy

		#TODO Maybe compute classifier bounds as well????

		if pprint:
			print('For feature {0}:'.format(tag))
			print('Number of secrets                        : {0}'.format(len(secrets)))
			print('A-priori Entropy                         : {0} bits'.format(priori_entropy))
			print('A-posteriori Entropy                     : {0} bits'.format(posteriori_entropy))
			print('Leakage Entropy (a-priori - a-posteriori): {0} bits'.format(leakage_entropy))
			#print('Loss                                     : {0}'.format(loss))
			print('-'*80)

		return (priori_entropy, posteriori_entropy, leakage_entropy)	

	@classmethod
	def phase2interval(cls, interactions):
		#Intervals are list of dictionaries.
		intervals = [None] * len(interactions)

		#orig_markerlist = [Utils.is_phase_marker(x) for x in interactions[0]]
		for (i,intr) in enumerate(interactions):
			markerlist = [Utils.is_phase_marker(x) for x in intr]
			#They contain the number of interaction they point to and a list of lists with 2 elements
			#Establishing interaction_num key value mapping
			interval = {'interaction_num': i}
			interval_list = list()

			#Establishing splitting points according to phase markers.
			begin = 0
			end = len(intr)-1

			for (j,phase_bool) in enumerate(markerlist):
				if phase_bool:
					interval_list.append([begin, j-1])
					begin = j
			interval_list.append([begin,end])

			interval['interval_list'] = interval_list
			#Adding this dictionary to list
			intervals[i] = interval

		return intervals

	@classmethod
	def run_leakiest(cls, features, labels, f_names, attr_name='integer', problem_name='test'):
		"""Takes the feature list (which includes features obtained from each sample), label list (which contains the class of each sample)
			and list of names of feature and creates an .arff file for storing the data.
			Then, it passes the .arff to Leakiest tool using some scripting to obtain leakage and prints it to stdout.
			Returns nothing.
		"""
		e = re.compile(r'\s+')
		problem_name = e.sub('_', problem_name)
		arff_file   = problem_name + '.arff'
		config_file = 'Cfg_' + problem_name + '.txt'
		leakiest_abspath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bin/leakiest-1.4.7.jar')

		features = list(map(list, zip(*features)))

		#Name the problem as a @relation <problem-name>
		#Name the features as @attribute <feature-name> <feature-type> (We're using number for now)
		#Start the data with @data and list the features and classes(as a number from 0 to X) in a list
		#	as comma separated values without space between them.
		print('# of samples (features):' + str(len(features)))
		print('# of samples (labels):'   + str(len(labels)))
		print('# of features:'           + str(len(f_names)))
		print(leakiest_abspath)
		with open(arff_file, 'w') as file:
			file.write('@relation {}\n'.format(problem_name))
			file.write('\n')
			file.write('@attribute class {}\n'.format(attr_name))
			for name in f_names:
				mod_name = e.sub('_', name)
				file.write('@attribute {} number\n'.format(mod_name))
			file.write('\n')
			file.write('@data\n')
			for (f_list,l) in zip(features,labels):
				file.write(str(l))
				for f in f_list:
					file.write(','+str(f))
				file.write('\n')
		#Create a config file that marks the secret column as high and rest as low. (They are indexed from 0 to N-1).
		with open(config_file, 'w') as file:
			cmd_str = '//java -jar {} -cfg {}\n'.format(leakiest_abspath, config_file)
			usage = ['//CFG//\n', '//Usage:\n', cmd_str]
			file.writelines(usage)
			file.writelines('/'*40)
			file.write('\n')

			leakage_usage = ['// The leakage options are -mi (Mutual Information), -cp (Channel Capacity)\n',
					'//-me (Min. Entropy) and -mc (Min. Capacity).\n']

			observation_usage = ['// The observable options are -di (Discrete) or -co (Continuous).\n']
			if attr_name == 'integer':
				parameters = leakage_usage + ['-mi\n','\n'] + observation_usage + ['-di\n','\n', '-a\n', '\t"./{}"'.format(arff_file), '\n']
				file.writelines(parameters)
			else:
				parameters = leakage_usage + ['-mi\n','\n'] + observation_usage + ['-co\n','\n', '-a\n', '\t"./{}"'.format(arff_file), '\n']
				file.writelines(parameters)

			low_hi_params = ['-high 0\n', '\n', '-low @each\n', '\n', '-v 0', '\n']
			file.writelines(low_hi_params)

		#Execute the script file.
		#Print the results to a file and to the stdout.
		cmd_string = ['java', '-jar', leakiest_abspath, '-cfg', config_file, '|', 'tee', 'leak_results_{}.txt'.format(problem_name)]
		print(' '.join(cmd_string))
		subprocess.call(cmd_string)

	@classmethod
	def process_all(cls, ports=[], pcap_filename=None, interactions=None, intervals=None, debug_print=False, calcSpace=True, calcTime=True,
		use_phases=False, plot_folder='./tsa-plots/', dry_mode=False, quant_mode='normal', window_size=None, plot=False, special_tag=None, 
		feature_reduction=None, num_reduced_features=5, alignment=False, new_direction=False, silent=False):
		"""
		This method parses pcaps, gets the alignment information from intervals, extracts features from list of traces,
		quantifies information leakage using various methods and tools.

		Alternatively, instead of giving a pcap filename to parse, it can take interactions (list of traces) as an input and process that.

		Basically, this method does almost everything we need for an experiment.

		Args:
			ports: List of ports for packets that we are interested in.
			pcap_filename: File location string of pcap file. If interactions is not provided, it is populated using parsing the file denoted by this argument.
			interactions: List of packet traces.
			intervals: JSON object describing phases.
			debug_print: This method prints debug information while executing if this argument is True.
			calcSpace: This method quantifies information leakage using space features (packet sizes) if this argument is True.
			calcTime: This method quantifies information leakage using time features (packet timings) if this argument is True.
			use_phases: This method does not use phases if this argument is True.
			plot_folder: Folder location to place the plots this method processes.
			dry_mode: This method does not quantify information leakage if this argument is True. Used for debug and special purposes.
			quant_mode: This argument sets the information leakage quantification or classification method. The options are:
				normal             - Quantifying information by modeling the data distribution with Normal distributions.
				hist               - Quantifying information by modeling the data distribution with Histograms.
				kde                - Quantifying information by modeling the data distribution with Kernel Density Estimation (KDE).
				kde-dynamic        - Quantifying information by modeling the data distribution with KDE with bandwidth selection with Max.Likelihood.
				gmm                - Quantifying information by modeling the data distribution with Bayesian Gaussian Mixture Model.
				leakiest-integer   - Quantifying information using Leakiest tool, integer mode. Used for discrete features (like packet sizes).
				leakiest-real      - Quantifying information using Leakiest tool, real mode. Used for continuous features (like packet timings).
				fbleau             - Quantifying information using F-BLEAU tool.

				rf-classifier      - Classifying information using Random Forest Classifier.
				knn-classifier     - Classifying information using K-Nearest Neighbors Classifier.
				nb-classifier      - Classifying information using Gaussian Naive Bayes.
				fcnn-classifier    - Classifying information using Fully Connected Neural Networks.

				#Future Work
				convnn-classifier - Classifying information using Convolutional Neural Networks.
				rnn-classifier     - Classifying information using Recurrent Neural Networks.

			window_size: This argument sets the window size for KDE and Histogram methods. If it is set to None, the window sizes are set to default values.
			plot: This argument sets whether the distributions that are estimated will be plotted.

		Returns:
			The feature values, feature names, leakage results and trace labels in a tuple.
		"""

		#======================================
		#STEP 1 - CREATING LISTS OF TRACES OUT OF .PCAP FILES
		#======================================

		quant_mode_list = ['normal', 'hist', 'kde', 'kde-dynamic', 'gmm', 'leakiest-integer', 'leakiest-real', 'fbleau'] + \
		['rf-classifier', 'knn-classifier', 'nb-classifier', 'fcnn-classifier'] + ['convnn-classifier', 'rnn-classifier']

		if calcSpace is False and calcTime is False:
			print('Information leakage quantification on both *space and time* are disabled.')
			print('Please turn on either one of them to process your traces.')
			return None

		if quant_mode not in quant_mode_list:
			print('You set quantification mode as {}. The only valid options are {}.'.format(quant_mode, quant_mode_list))
			print('Please set quant_mode argument as one of the valid options.')
			return None

		if feature_reduction is not None and 'classifier' not in quant_mode:
			print('Feature reduction only works on training classifiers and uses the ranking generated by quantification.')
			print('It won\'t have any effect on the results of selected quantification mode.')

		#Feature Reduction we propose uses the quantification's feature rankings for reducing the number of features used for classification.
		quant_mode_classifier = None
		if 'classifier' in quant_mode: #feature_reduction is not None and 
			quant_mode_classifier = quant_mode
		
		if feature_reduction == 'ranking' and 'classifier' in quant_mode:
			quant_mode = 'normal'
			#quant_mode = 'kde-dynamic'

		intrs = None
		split_intrs = None
		intr_list = []
		problem_name = None
		start_time = time.time()

		#Creating folders for plots and graphs
		if plot_folder[-1] != '/':
			plot_folder += '/'

		if not os.path.exists(plot_folder):
			if not silent:
				print('Creating folder for plots and graphs at {}'.format(plot_folder))
			os.makedirs(plot_folder)

		problem_name = 'unknown-pcap' + str(datetime.datetime.now())
		problem_name = problem_name.replace(' ', "_")
		if pcap_filename is not None:
			problem_name = pcap_filename.split('/')[-1]
			if '.' in problem_name:
				x = problem_name.split('.')
				problem_name = '.'.join(x[:-1])
		else:
			if not silent:
				print('Pcap filename not provided, creating default folder with timestamp.')
		pcap_folder = plot_folder + problem_name + '/'

		if plot and not os.path.exists(pcap_folder):
			if not silent:
				print('Creating folder for plots and graphs at {}'.format(pcap_folder))
			os.makedirs(pcap_folder)

		#This branch processes pcaps with the Sniffer class if that's provided
		if (pcap_filename is not None and type(pcap_filename) is str and \
		len(ports) > 0 and interactions is None and '.pcap' in pcap_filename):
			# Starting offline sniffing
			if not silent:
				print('---')
				print('Processing file:')
			print(pcap_filename)
			s = Sniffer(ports, offline=pcap_filename, showpackets=False)
			s.start()
			s.join()
			s.cleanup2interaction()
			#Files to process
			intrs = s.processed_intrs
		#This branch takes the processed interactions file if that's processed beforehand.
		elif (interactions is not None and type(interactions) is list):
			intrs = interactions
		else:
			assert False, 'Either ports and pcap_filename or interactions file needs to be provided.'

		if len(intrs) == 0:
			print('Cannot read the pcap file, no traces found.')
			return None

		if alignment:
			use_phases = True
			secrets = Transform.rd_secrets(intrs)
			intr_sizes = [[p.size for p in t] for t in intrs]
			subsetSize = min(1000, int(len(intrs)/5))
			alignment = Alignment(filepath='', interactions=intr_sizes, secrets=secrets, subsetSize=subsetSize)
			intervals = alignment.align()

		parse_time = time.time()

		#=====================================
		#ADDITION - THE INTERVALS & DIRECTIONS
		#=====================================
		#Creating intervals out of phases
		if use_phases and (intervals is None):
			intervals = cls.phase2interval(intrs)

		#Clean intervals if interval algorithm gave empty interval list.
		if use_phases:
			intervals = [x for x in intervals if len(x['interval_list']) != 0]

			#Print number of intervals found by the objects. If they are inconsistent, we can apply a fix later.
			lengths = list(set([len(x['interval_list']) for x in intervals]))
			lengths.sort()
			if len(lengths) != 1:
				if not silent:
					print('The number of intervals/phases in the json object vary between these numbers: {}'.format(lengths))

				# Counting the number of phases for each trace.
				len_dict = {}
				for x in intervals:
					if len(x['interval_list']) not in len_dict:
						len_dict[len(x['interval_list'])] = 1
					else:
						len_dict[len(x['interval_list'])] += 1

				# Determining the majority among number of phases and only keeping the traces with that much phases.
				
				majority_phase_num = max([(k, len_dict[k]) for k in len_dict.keys()], key=lambda x: x[1])[0]
				intervals = [x for x in intervals if len(x['interval_list']) == majority_phase_num]
				if not silent:
					print('List for number of phases and number of traces with that much phases: {}'.format(len_dict))
					print('Number of traces after the culling: {}'.format(len(intervals)))
			else:
				if not silent:
					print('The number of intervals/phases in the json object: {}'.format(lengths[0]))

		if use_phases:
			#Applying intervals to get smaller interactions
			subintrs = Transform.tf_split_intervals(intrs, intervals)
			#split_intrs = map(list,zip(*subintrs))
			split_intrs = [list(x) for x in zip(*subintrs)]
			#print('Number of Phases: {}'.format(split_intrs))
			if len(split_intrs) == 1: use_phases = False

		#Pruning 0-sized packets for easier processing
		intrs = Transform.tf_prune_size(intrs)

		#Removing traces based on alignment json file
		if use_phases and intervals is not None:
			intrs = Transform.tf_remove_intrs_intervals(intrs, intervals)

		#Pruning 0-sized packets for easier processing
		intrs = Transform.tf_prune_size(intrs)

		#Applying directions for full trace
		(dirs, dir_intrs) = Transform.tf_split_directions(intrs) #SLOW
		intr_list = [intrs]
		intr_tags = ['full trace, both directions']
		if len(dirs) > 0:
			intr_list = intr_list + dir_intrs
			intr_tags = intr_tags + ['full trace, in direction {}->{}'.format(src,dst) for (src,dst) in dirs]
		elif new_direction:
			(addrs, dir_intrs) = Transform.tf_split_directions_for_single_address(intrs)
			if len(addrs) > 0:
				intr_list = intr_list + dir_intrs
				temp_tags = [['full trace, in direction {}->ALL'.format(addr), 'full trace, in direction ALL->{}'.format(addr)] for addr in addrs]
				temp_tags = functools.reduce(lambda x,y: x+y, temp_tags, [])
				intr_tags = intr_tags + temp_tags

		all_space_features = []
		all_time_features  = []
		all_space_tags     = []
		all_time_tags      = []
		all_space_leakage  = []
		all_time_leakage   = []

		if use_phases:
			for (i,new_intrs) in enumerate(split_intrs):
				#Pruning 0-sized packets for easier processing
				new_intrs = Transform.tf_prune_size(new_intrs)

				#Direction splitting for intervals
				(dirs,dir_intrs) = Transform.tf_split_directions(new_intrs) #SLOW

				#Collecting all interactions and naming them.
				intr_list.append(new_intrs)
				intr_tags.append('interval {}, both directions'.format(i+1))

				if len(dirs) > 0:
					intr_list = intr_list + dir_intrs
					intr_tags += ['interval {}, in direction {}->{}'.format(i+1,src,dst) for (src,dst) in dirs]
				elif new_direction:
					(addrs, dir_intrs) = Transform.tf_split_directions_for_single_address(intrs)
					if len(addrs) > 0:
						intr_list = intr_list + dir_intrs
						temp_tags = [['interval {}, in direction {}->ALL'.format(i+1, addr), 'full trace, in direction ALL->{}'.format(addr)] for addr in addrs]
						temp_tags = functools.reduce(lambda x,y: x+y, temp_tags, [])
						intr_tags = intr_tags + temp_tags

		#==========================================
		#STEP 2 - USING PACKETS TO EXTRACT FEATURES
		#==========================================

		#Secrets and conversion to label format
		secrets = Transform.rd_secrets(intrs)
		(labels, s2l_dict) = Utils.secrets2labels(secrets)
		full_entropy = np.log2(len(set(labels))) # Entropy for uniformly distributed secret value.

		#Extracting features for all interactions/subinteractions
		for (intr,tag) in zip(intr_list,intr_tags):
			if not silent:
				print('Processing {0}'.format(tag))

			#Extracting features
			(space_features, time_features, space_tags, time_tags) = Transform.rd_extract_features(intr, tag, calcSpace, calcTime, padding=False)

			all_space_features += space_features
			all_time_features  += time_features
			all_space_tags     += space_tags
			all_time_tags      += time_tags

			if not silent:
				print("Number of features extracted: {}".format(len(space_features)))

			#====================================================
			#STEP 3 - USING FEATURES TO CALCULATE LEAKAGE ENTROPY
			#====================================================

			space_leakage_list = []
			time_leakage_list  = []

			#Calculating leakage for space features
			if calcSpace and not dry_mode:
				if window_size is None and (quant_mode == 'hist'):
					window_size = 0.1
				if window_size is None and (quant_mode == 'normal'):
					window_size = 0.0001

				if window_size is None and (quant_mode == 'kde'):
					if not silent:
						print('Running KDE Quantification with window size equal to standard deviation.')
					option = 'stddev'
				elif window_size is not None and (quant_mode == 'kde'):
					if not silent:
						print('Running KDE Quantification with fixed window size of {}.'.format(window_size))
					option = 'fixed'
				elif quant_mode == 'hist':
					if not silent:
						print('Running Histogram Q. with bin size = {}'.format(window_size))

				for (i,features) in enumerate(space_features):
					#Features have the same value, no leakage, no need to analyze.
					tag = space_tags[i]
					if not silent:
						print("Testing tag: {}".format(tag))
					if special_tag is not None and special_tag not in tag:
						continue

					if(len(set(features)) <= 1) and quant_mode in ['normal', 'kde', 'hist', 'kde-dynamic', 'gmm']:
						space_leakage_list.append((0.0,tag))
						continue

					filename = pcap_folder + 'SPACE-' + tag.replace(' ','_')
					leakage_entropy = 0.0

					if quant_mode == 'normal':
						(_, _, leakage_entropy) = cls.estimate_entropy_norm(features, labels, filename, min_var=window_size, pprint=debug_print, plot=plot)
					elif quant_mode == 'kde':
						(_, _, leakage_entropy) = cls.estimate_entropy_kde(features, labels, filename, bdwidth=window_size, pprint=debug_print, plot=plot, option=option, option2='space')
					elif quant_mode == 'kde-dynamic':
						(_, _, leakage_entropy) = cls.estimate_entropy_kde(features, labels, filename, bdwidth=window_size, pprint=debug_print, plot=plot,
							option='dynamic', option2='space')
					elif quant_mode == 'gmm':
						(_, _, leakage_entropy) = cls.estimate_entropy_gmm(features, labels, filename, max_n_comps=window_size, pprint=debug_print, plot=plot)
					elif quant_mode == 'hist':
						(_, _, leakage_entropy) = cls.estimate_entropy_hist(features, labels, filename, bin_size=window_size, pprint=debug_print, plot=plot)

					if quant_mode in ['normal', 'kde', 'hist', 'kde-dynamic', 'gmm']:
						space_leakage_list.append((leakage_entropy,tag))

				all_space_leakage += space_leakage_list

			#Calculating leakage for time features
			if calcTime and not dry_mode:
				if window_size is None and (quant_mode == 'hist'):
					window_size = 0.00001
				if window_size is None and (quant_mode == 'normal'):
					window_size = 0.00001

				if window_size is None and (quant_mode == 'kde'):
					if not silent:
						print('Running KDE Quantification with window size equal to standard deviation.')
					option = 'stddev'
				elif window_size is not None and (quant_mode == 'kde'):
					if not silent:
						print('Running KDE Quantification with fixed window size of {}.'.format(window_size))
					option = 'fixed'
				elif quant_mode == 'hist':
					if not silent:
						print('Running Histogram Q. with bin size = {}'.format(window_size))

				for (i,features) in enumerate(time_features):
					tag = time_tags[i]
					if not silent:
						print("Testing tag: {}".format(tag))

					if special_tag is not None and special_tag not in tag:
						continue

					if(len(set(features)) <= 1) and quant_mode in ['normal', 'kde', 'hist', 'kde-dynamic', 'gmm']:
						time_leakage_list.append((0.0, tag))
						continue

					filename = pcap_folder + 'TIME-' + tag.replace(' ','_')
					leakage_entropy = 0.0

					if quant_mode == 'normal':
						(_, _, leakage_entropy) = cls.estimate_entropy_norm(features, labels, filename, min_var=window_size,  pprint=debug_print, plot=plot)
					elif quant_mode == 'kde':
						(_, _, leakage_entropy) = cls.estimate_entropy_kde( features, labels, filename, bdwidth=window_size,  pprint=debug_print, plot=plot,
							option=option, option2='time')
					elif quant_mode == 'kde-dynamic':
						(_, _, leakage_entropy) = cls.estimate_entropy_kde(features, labels, filename, bdwidth=window_size, pprint=debug_print, plot=plot,
							option='dynamic', option2='time')
					elif quant_mode == 'gmm':
						(_, _, leakage_entropy) = cls.estimate_entropy_gmm(features, labels, filename, max_n_comps=window_size, pprint=debug_print, plot=plot)
					elif quant_mode == 'hist':
						(_, _, leakage_entropy) = cls.estimate_entropy_hist(features, labels, filename, bin_size=window_size, pprint=debug_print, plot=plot)

					if quant_mode in ['normal', 'kde', 'hist', 'kde-dynamic', 'gmm']:
						time_leakage_list.append((leakage_entropy,tag))

				all_time_leakage += time_leakage_list

			#PRINTING TIME!
			if debug_print:
				print('Number of secrets: {0}'.format(len(set(labels))))
				print('Full Entropy     : {:0.3f} bits'.format(full_entropy))

				print('='*80)
				if calcSpace:
					print('Number of features: {}'.format(len(space_features)))
					space_leakage_list.sort(key=itemgetter(0), reverse=True)
					for (leakage,tag) in space_leakage_list:
						print('{:0.3f}/{:0.3f} bits leaked for feature: {}'.format(leakage,full_entropy,tag))
					print('='*80)

				if calcTime:
					print(type(time_features))
					print(type(all_time_features))
					print('Number of features: {}'.format(len(time_features)))
					time_leakage_list.sort(key=itemgetter(0), reverse=True)
					for (leakage,tag) in time_leakage_list:
						print('{:0.3f}/{:0.3f} bits leaked for feature: {}'.format(leakage,full_entropy,tag))

		magic_lim = 20
		if not silent:
			print('~'*80)
			print('~'*80)
			print('Number of secrets: {0}'.format(len(set(labels))))
			print('Full Entropy     : {:0.3f} bits'.format(full_entropy))
			print('~'*80)
		if calcSpace and not dry_mode and not silent:
			limit = min(magic_lim,len(all_space_leakage))
			print('Printing {} most leaking space features:'.format(limit))
			print('='*80)

			prefix = problem_name + ' SPACE '

			all_space_leakage.sort(key=itemgetter(0), reverse=True)
			for (leakage,tag) in all_space_leakage[:limit]:
				percentage = int(np.round((leakage/full_entropy) * 100))
				print_str = prefix + '{}% {:0.3f}/{:0.3f} bits leaked for feature: {}'.format(percentage,leakage,full_entropy,tag)
				print(print_str)

		if calcTime and not dry_mode and not silent:
			limit = min(magic_lim,len(all_time_leakage))
			print('+'*80)
			print('Printing {} most leaking time features:'.format(limit))
			print('='*80)

			prefix = problem_name + ' TIME '

			all_time_leakage.sort(key=itemgetter(0), reverse=True)
			for (leakage,tag) in all_time_leakage[:limit]:
				percentage = int(np.round((leakage/full_entropy) * 100))
				print_str = prefix + '{}% {:0.3f}/{:0.3f} bits leaked for feature: {}'.format(percentage,leakage,full_entropy,tag)
				print(print_str)

		sys.stdout.flush()

		if quant_mode == 'leakiest-real':
			if calcSpace:
				print('Running Leakiest continuous mode for Space features:')
				sys.stdout.flush()
				cls.run_leakiest(all_space_features, labels, all_space_tags, 'real', problem_name + '-Space') # Original ndss paper was using 'integer' for space instead of real.
			if calcTime:
				print('Running Leakiest continuous mode for Time features:')
				sys.stdout.flush()
				cls.run_leakiest(all_time_features, labels, all_time_tags, 'real', problem_name + '-Time')
		elif quant_mode == 'leakiest-integer':
			if calcSpace:
				print('Running Leakiest discrete mode for Space features:')
				sys.stdout.flush()
				cls.run_leakiest(all_space_features, labels, all_space_tags, 'integer', problem_name + '-Space')
			if calcTime:
				print('Running Leakiest discrete mode for Time features:')
				sys.stdout.flush()
				cls.run_leakiest(all_time_features, labels, all_time_tags, 'integer', problem_name + '-Time')
		elif quant_mode == 'fbleau':
			#TODO Test if fbleau is installed in the machine.
			#If not, just exit gracefully.
			if calcSpace:
				print('Running F-BLEAU for Space features:')
				sys.stdout.flush()
				cls.run_fbleau(all_space_features, labels, all_space_tags, 'log', problem_name + '-Space') # Original ndss paper was using 'integer' for space instead of real.
			if calcTime:
				print('Running F-BLEAU for Time features:')
				sys.stdout.flush()
				cls.run_fbleau(all_time_features, labels, all_time_tags, 'log', problem_name + '-Time')

		sys.stdout.flush()
		if calcSpace and calcTime:
			features = all_space_features + all_time_features
			tags = all_space_tags + all_time_tags
			leakage = all_space_leakage + all_time_leakage
			leakage.sort(key=itemgetter(0), reverse=True)
		elif calcSpace:
			features = all_space_features
			tags = all_space_tags
			leakage = all_space_leakage
		elif calcTime:
			features = all_time_features
			tags = all_time_tags
			leakage = all_time_leakage
		
		quant_time = time.time()

		#Reducing number of features with respect to ranking
		if feature_reduction is not None:
			if not silent:
				print('Reducing number of features to {} using {}.'.format(num_reduced_features, feature_reduction))

		if feature_reduction=='ranking' and quant_mode_classifier is not None:
			if not silent:
				print('Applying feature selection using mutual information.')
			new_features = []
			new_tags = []
			for (l, tag) in leakage[:num_reduced_features]:
				if not silent:
					print('Selecting feature {} with leakage {}'.format(tag, l))
				for (f, t) in zip(features, tags):
					if tag == t:
						new_features.append(f)
						new_tags.append(t)
						break
			features = new_features
			tags = new_tags

		fr_time = time.time()

		if quant_mode_classifier is not None:
			#For classification methods
			quant_mode = quant_mode_classifier
		else:
			#For quantification methods
			#if not silent:
			#	print("Returning Leakage: {}".format(leakage))
			importance_tags_list = copy.deepcopy(leakage)
			if not (quant_mode == 'fbleau' or 'leakiest' in quant_mode):
				leakage = leakage[0][0]
			return (labels, features, tags, leakage, importance_tags_list)

		X = []
		y = []

		if 'classifier' in quant_mode:
			features_max = [max(f) for f in features]
			features_norm = [[float(x)/max_f if max_f>0 else float(x) for x in f] for (f, max_f) in zip(features, features_max)]
			features_t = list(zip(*features_norm))
			X = features_t
			y = labels
			pairs = list(zip(features_t, labels))
			if not silent:
				print('Number of data points:{}, Number of corresponding labels:{}'.format(len(features_t), len(labels)))
				print('Number of features:{}'.format(len(features_t[0])))
			##Balanced Shuffle:
			secs = list(set(labels))
			pairs_partitioned = [list() for _ in range(len(secs))]
			for pair in pairs:
				for i, l in enumerate(secs):
					if pair[1] == l:
						pairs_partitioned[i].append(pair)
			
			ratio = 0.8

			X_train = []
			X_test  = []
			y_train = []
			y_test  = []

			for pairs_list in pairs_partitioned:
				random.shuffle(pairs_list)
				X_train += [x[0] for x in pairs_list[:round(len(pairs_list)*ratio)]]
				X_test  += [x[0] for x in pairs_list[round(len(pairs_list)*ratio):]]
				y_train += [x[1] for x in pairs_list[:round(len(pairs_list)*ratio)]]
				y_test  += [x[1] for x in pairs_list[round(len(pairs_list)*ratio):]]
		
		pad = True
		if quant_mode == 'rnn-classifier' or quant_mode == 'convnn-classifier':
			pad = True

		#Doing k-fold cross validation for robust analysis
		n_splits = 5
		avg_accuracy = 0.0
		avg_precision = 0.0
		avg_recall = 0.0
		avg_f1 = 0.0
		avg_importance = None

		skf = StratifiedKFold(n_splits=n_splits)

		nn_once_run = False
		
		for train_index, test_index in skf.split(X,y):
			X_train = [X[ind] for ind in train_index]
			X_test  = [X[ind] for ind in test_index]
			y_train = [y[ind] for ind in train_index]
			y_test  = [y[ind] for ind in test_index]

			if feature_reduction=='pca' and 'classifier' in quant_mode and quant_mode_classifier is not None:
				#print('Applying dimensionality reduction using PCA (Principal Component Analysis).')
				X_train, X_test = cls.reduce_dim_pca_new(X_train, X_test, n_components=num_reduced_features, padding = pad)

			if feature_reduction=='lda' and 'classifier' in quant_mode and quant_mode_classifier is not None:
				#print('Applying dimensionality reduction using LDA (Linear Discriminant Analysis).')
				X_train, X_test = cls.reduce_dim_lda_new(X_train, y_train, X_test, n_components=num_reduced_features, padding = pad)
				
			if quant_mode == 'rf-classifier':
				ensemble = importlib.import_module('sklearn.ensemble') #from sklearn.ensemble import RandomForestClassifier

				clf = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
				if not silent:
					print('Using Random Forest Classifier')
			elif quant_mode == 'knn-classifier':
				neighbors = importlib.import_module('sklearn.neighbors') #from sklearn.neighbors import KNeighborsClassifier # K-NN

				clf = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
				if not silent:
					print('Using K-Nearest Neighbors Classifier')
			elif quant_mode == 'nb-classifier':
				naive_bayes = importlib.import_module('sklearn.naive_bayes')
				#from sklearn.naive_bayes import GaussianNB # Gaussian Naive-Bayes

				clf = naive_bayes.GaussianNB()
				if not silent:
					print('Using Gaussian Naive Bayes Classifier')
			elif quant_mode == 'fcnn-classifier':
				neural_network = importlib.import_module('sklearn.neural_network')
				#from sklearn.neural_network import MLPClassifier # Fully Connected Neural Networks

				#nn_once_run = True
				grid = GridSearchCV(neural_network.MLPClassifier(batch_size=16, max_iter=2500),
					param_grid = {'hidden_layer_sizes': [(20,10), (10,10), (5,5)]}, n_jobs = -1,
					cv=StratifiedKFold(5))
				grid.fit(X_train, y_train)
				hidden_layer_sizes = grid.best_params_['hidden_layer_sizes']
				#print(lr)
				#clf = grid.best_estimator_

				clf = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, batch_size=16, max_iter=2500)

				if not silent:
					print('Using Fully Connected Neural Network for classification')
			elif quant_mode == 'convnn-classifier' and not nn_once_run:
				nn_once_run = True
				cls.classify_convnn(X, y, X_train, y_train, X_test, y_test) #Check if we need to retrain the network n times, usually neural network CV is done differently
			elif quant_mode == 'rnn-classifier':
				cls.classify_rnn(X_train, y_train, X_test, y_test)

			if quant_mode in ['rf-classifier', 'knn-classifier', 'nb-classifier', 'fcnn-classifier']:
				#if not nn_once_run:
				clf.fit(X_train, y_train)
				train_acc = clf.score(X_train, y_train)
				#test_acc = clf.score(X_test, y_test)
				#print('\nTraining Accuracy: {:.2f}, Test Accuracy: {:.2f}'.format(train_acc, test_acc))
				results = clf.predict(X_test)
				accuracy = accuracy_score(y_test, results)
				precision, recall, f1sc, support = precision_recall_fscore_support(y_test, results, average = 'macro')
				#precision = precision_score(y_test, results, average = 'macro')
				#recall = recall_score(y_test, results, average = 'macro')
				#f1sc = f1_score(y_test, results, average = 'macro')
				importance = clf.feature_importances_

				avg_accuracy += accuracy
				avg_precision += precision
				avg_recall += recall
				avg_f1 += f1sc
				if avg_importance is None:
					avg_importance = importance
				else:
					avg_importance = [el1+el2 for (el1, el2) in zip(avg_importance, importance)]

		avg_accuracy /= n_splits
		avg_precision /= n_splits
		avg_recall /= n_splits
		avg_f1 /= n_splits
		avg_importance = [el/n_splits for el in avg_importance]

		importance_tags_list = list(zip(avg_importance, tags))
		importance_tags_list.sort(key=lambda x: -x[0])

		print('Average Test Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, Overall F-score: {:.2f}'.format(avg_accuracy, avg_precision, avg_recall, avg_f1))
		if not silent:
			for importance, tag in importance_tags_list:
				print('***', 'Feature Importance for {} : {:.5f}'.format(tag, importance))
		
		leakage = avg_accuracy
		training_time = time.time()

		if not silent:
			print('Trace parsing/processing time: {:.2f} s'.format(parse_time - start_time))
			print('Quantification time:           {:.2f} s'.format(quant_time - parse_time))
			print('Feature reduction time:        {:.2f} s'.format(fr_time - quant_time))
			print('Classification time:           {:.2f} s'.format(training_time - fr_time))
			print('Total time:                    {:.2f} s'.format(training_time - start_time))

		return (labels, features, tags, leakage, importance_tags_list)

	@classmethod
	def inject_noise(cls, traces, target_tags, tags, features, labels, option='gaussian', label_distinguishing=True, option_val=100, pkt_injection = False, time_injection = False, train_traces = None):
		#Options from Related Work: Uniform, Uniform 1-255, Mice/Elephants, Linear, Exp, MTU (Pad fully), Pad to longest, Pinheiro 100, 500, 700, 900 
		#Options from our work: Gaussian dist?, Targeted padding with uniform and gaussian dist, Time delays, dummy packet injections, etc. (???)
		options_related = ['heartbeat', 'uniform', 'uniform255', 'mice-elephants', 'linear', 'exp', 'mtu', 'mtu-20ms', 'pad-highest', 'p100', 'p500', 'p700', 'p900', 'similarity', 'targeted']
		options_proposed = ['gaussian', 'fixed', 'proportional']
		options = options_related + options_proposed
		MTU = 1500

		if train_traces is None:
			train_traces = copy.deepcopy(traces)

		if option not in options and 'targeted' not in option:
			print('Given option: {} is not in list of options: {}'.format(option, options))
			return None

		if pkt_injection > 0:
			pkt_injection_param = pkt_injection
			pkt_injection = True
		else:
			pkt_injection_param = 0
			pkt_injection = False

		if time_injection > 0:
			time_injection_param = time_injection
			time_injection = True
		else:
			time_injection = False
			time_injection_param = 0.0

		#Packet injection code
		if pkt_injection:
			avg_pkts_injected = 0.0
			pkt_count = 0
			#Random packet injection
			directions = set()
			for tr in traces:
				trace_directions = set()
				for p in tr:
					direction = (p.src,p.dst)
					if direction not in trace_directions:
						trace_directions.add(direction)
				if len(directions) == 0:
					directions = trace_directions.union(directions)
				else:
					directions = trace_directions.intersection(directions)
			
			average_tr_len = np.average([len(tr) for tr in traces])
			for tr in traces:
				if len(tr) == 1 or len(tr) == 0:
					continue
				trace_diff = int(np.round(len(traces)-average_tr_len)*2)
				if trace_diff <= 0:
					trace_diff = 0
				if trace_diff > pkt_injection_param:
					trace_diff = pkt_injection_param
				randomval = random.randint(1,trace_diff)
				avg_pkts_injected += randomval
				pkt_count += len(tr)
				direction = (None, None)
				for _ in range(randomval):
					wrong_choice = True
					while wrong_choice:
						insert_ind = random.randint(1,len(tr)-1)
						direction = (None, None)
						coin = bool(random.getrandbits(1))
						timing = 0.0
						sport = None
						dport = None
						size = None
						load = None
						flags = None
						if len(directions) != 0:
							direction = random.sample(directions, 1)[0]
							
							#if insert_ind != len(tr) - 1:
							#	timing = random.uniform(tr[insert_ind].time, tr[insert_ind+1].time)
							#else:
							#	timing = tr[insert_ind].time
							#
							#tr.insert(insert_ind, Packet(direction[0], direction[1], tr[insert_ind].sport, tr[insert_ind].dport, tr[insert_ind].load, tr[insert_ind].size, tr[insert_ind].time, tr[insert_ind].flags))
							if insert_ind == len(tr):
								sport = tr[insert_ind-1].sport
								dport = tr[insert_ind-1].dport
								if tr[insert_ind-1].size > 1:
									size = random.randint(1, tr[insert_ind].size)
								else:
									size = tr[insert_ind-1].size
								load  = tr[insert_ind-1].load
								flags = tr[insert_ind-1].flags
							elif insert_ind == len(tr)-1 or coin:
								#direction = (tr[insert_ind].src, tr[insert_ind].dst)
								sport = tr[insert_ind].sport
								dport = tr[insert_ind].dport
								if tr[insert_ind].size > 1:
									size = random.randint(1, tr[insert_ind].size)
								else:
									size = tr[insert_ind].size
								load  = tr[insert_ind].load
								flags = tr[insert_ind].flags
							else: 
								#direction = (tr[insert_ind+1].src, tr[insert_ind+1].dst)
								sport = tr[insert_ind+1].sport
								dport = tr[insert_ind+1].dport
								if tr[insert_ind+1].size > 1:
									size = random.randint(1, tr[insert_ind+1].size)
								else:
									size = tr[insert_ind+1].size
								load  = tr[insert_ind+1].load
								flags = tr[insert_ind+1].flags
							if 'M' in flags:
								wrong_choice = True
								continue
							else:
								wrong_choice = False
							
							if insert_ind != len(tr) - 1:
								timing = random.uniform(tr[insert_ind].time, tr[insert_ind+1].time)
							else:
								timing = tr[insert_ind].time
							tr.insert(insert_ind, Packet(direction[0], direction[1], sport, dport, load, size, timing, flags))
			print('Average number of pkts injected: {:.2f}'.format(float(avg_pkts_injected)/pkt_count))

		#Make traces similar to traces with other labels
		if option == 'similarity':
			for (tr, l) in zip(traces, labels):
				#Find another trace with a different label
				random_counter = 0
				saved_trace = None
				while True:
					random_counter += 1
					rand_index = random.randint(0,len(train_traces)-1)
					tr2 = train_traces[rand_index]
					l2  = train_traces[rand_index]
					if l != l2 and len(tr) <= len(tr2):
						saved_trace = tr2
						break
					elif l != l2:
						saved_trace = tr2

					if random_counter > 20 and saved_trace is not None:
						break
				
				tr2 = saved_trace
				if len(tr) < len(tr2) and len(tr) > 0:
					p = tr[-1]
					tr = tr + [Packet(p.src, p.dst, p.sport, p.dport, p.load, 0, p.time, p.flags) for x in range(len(tr2)-len(tr))]
				
				for ind, p in enumerate(tr):
					if ind < len(tr2) and p.size < tr2[ind].size:
						p.size = tr2[ind].size

		#This needs to use the training data as the basis, I need to add some details so that it does not modify test data based on test data
		if 'targeted' in option:
			if 'targeted5' == option:
				new_target_tags = target_tags[:5]
			elif 'targeted10' == option:
				new_target_tags = target_tags[:10]
			elif 'targeted15' == option:
				new_target_tags = target_tags[:15]
			elif 'targeted20' == option:
				new_target_tags = target_tags[:20]
			else:
				new_target_tags = target_tags[:25]
			
			mean_std_padding = False
			mean_std_noising = False
			for tag in new_target_tags:
				max_padding = option_val
				#print('Targeting tag: {}'.format(tag))
				
				if 'Size of packet' in tag: #Size of packet INDEX in LOCATION
					pkt_ind = int(tag.split()[3]) - 1 #Extracting target packet index
					location = ' '.join(tag.split()[5:]) #Extracting packet context (direction, phase, etc.)
					#print('TAG: {}, PKT_IND: {}, LOCATION: {}'.format(tag, pkt_ind, location))
					if 'full trace, both directions' in location:
						#Padding every packet in that index to the max possible size among all traces
						max_size = max([tr[pkt_ind].size if pkt_ind < len(tr) else 0 for tr in train_traces])
						for tr in traces:
							if pkt_ind < len(tr) and tr[pkt_ind].size < max_size:
								tr[pkt_ind].size = max_size
					elif 'full trace, in direction' in location:
						directions = location.split()[4].split('->')
						src = directions[0]
						dst = directions[1]
						#Finding max size among all traces
						max_size = 0
						print(src, dst)
						for tr in train_traces:
							dir_counter = 0
							for ind, p in enumerate(tr):
								if p.src != src or p.dst != dst:
									continue
								elif p.src == src and p.dst == dst and dir_counter == pkt_ind:
									if p.size > max_size:
										max_size = p.size
									break
								elif p.src == src and p.dst == dst and dir_counter != pkt_ind:
									dir_counter += 1

						for tr in traces:
							dir_counter = 0
							for ind, p in enumerate(tr):
								if p.src != src or p.dst != dst:
									continue
								elif p.src == src and p.dst == dst and dir_counter == pkt_ind:
									if p.size < max_size:
										p.size = max_size
									break
								elif p.src == src and p.dst == dst and dir_counter != pkt_ind:
									dir_counter += 1
					else:
						print('Something is wrong, this direction is not recognized;', tag, '--', location)
					
				elif 'Number of packets with size' in tag: #Number of packets with size SIZE in LOCATION
					size = int(tag.split()[5]) #Extracting target packet size
					location = ' '.join(tag.split()[7:]) #Extracting packet context (direction, phase)
					#print('TAG: {}, SIZE: {}, LOCATION: {}'.format(tag, size, location))
					if 'full trace, both directions' in location:
						for ind1, tr in enumerate(traces):
							for ind, p in enumerate(tr):
								if p.size == size and p.size < max_padding:
									p.size = max_padding
					elif 'full trace, in direction' in location:
						directions = location.split()[4].split('->')
						src = directions[0]
						dst = directions[1]
						for ind1, tr in enumerate(traces):
							for ind, p in enumerate(tr):
								if p.size == size and p.src == src and p.dst == dst and p.size < max_padding:
									p.size = max_padding
					else:
						print('Something is wrong, this direction is not recognized;', tag, '--', location)
					
					#Counting number of packets with that size among traces
					#count_list = [0 for tr in traces]
					#for ind1, tr in enumerate(traces):
					#	count_list[ind1] = 0
					#	for ind, p in enumerate(tr):
					#		if p.size == size:
					#			count_list[ind1] += 1
					#max_count = max(count_list)
					#
					#for ind1, tr in enumerate(traces):
					#	random_counter = 0
					#	while count_list[ind1] < max_count:
					#		random_counter += 1
					#		rand_index = random.randint(0,len(tr)-1)
					#		if tr[rand_index] < size:
					#			tr[rand_index] = size
					#			count_list[ind1] += 1
					#
					#		if random_counter > 20:
					#			for _ in range(max_count - count_list[ind1]):
					#				rand_index = random.randint(0,len(tr)-1)
					#				tr.insert(rand_index, Packet(tr[rand_index].src, tr[rand_index].dst, tr[rand_index].sport, tr[rand_index].dport, tr[rand_index].load, tr[rand_index].size, tr[rand_index].time, tr[rand_index].flags))
					#			break
				elif 'Number of packets in' in tag: #Number of packets in LOCATION
					location = ' '.join(tag.split()[4:]) #Extracting packet context (direction, phase)
					#TODO Inject packets to have equal number of packets
					continue
				#elif 'Sum of sizes in' in tag:
				#	location = ' '.join(tag.split()[4:])  #Extracting packet context (direction, phase)
				#	#TODO Pad packets such that sizes are same/random in the same range
				#	#May screw up the packet based mitigation
				#	continue
				elif 'Minimum of sizes in' in tag:
					location = ' '.join(tag.split()[4:])  #Extracting packet context (direction, phase)
					if 'full trace, both directions' in location:
						min_size = max([min([p.size for p in tr]) for tr in train_traces])
						for tr in traces:
							for p in tr:
								if p.size < min_size:
									p.size = min_size
					elif 'full trace, in direction' in location:
						directions = location.split()[4].split('->')
						src = directions[0]
						dst = directions[1]
						min_size = 0
						for tr in train_traces:
							subtrace = [p.size for p in tr if p.src == src and p.dst == dst]
							if len(subtrace) > 0 and min(subtrace) > min_size:
								min_size = min(subtrace)
						for tr in traces:
							for p in tr:
								if p.src == src and p.dst == dst and p.size < min_size:
									p.size = min_size
					else:
						print('Something is wrong, this direction is not recognized;', tag, '--', location)
				elif 'Maximum of sizes in' in tag:
					continue
					#location = ' '.join(tag.split()[4:])  #Extracting packet context (direction, phase)
					#if 'full trace, both directions' in location:
					#	#Counting number of packets with that size among traces
					#	max_size = max([max([p.size for p in tr]) for tr in train_traces])
					#	min_size = min([max([p.size for p in tr]) for tr in train_traces])
					#	for tr in traces:
					#		for p in tr:
					#			if min_size <= p.size and p.size <= max_size:
					#				p.size = max_size
					#elif 'full trace, in direction' in location:
					#	#TODO Placeholder
					#	max_size = max([max([p.size for p in tr]) for tr in train_traces])
					#	min_size = min([max([p.size for p in tr]) for tr in train_traces])
					#	for tr in traces:
					#		for p in tr:
					#			if min_size <= p.size and p.size <= max_size:
					#				p.size = max_size
				elif ('Sum of sizes in' in tag or 'Mean of sizes in' in tag or 'Std.dev. of sizes in' in tag):			
					location = ' '.join(tag.split()[4:]) #Extracting packet context (direction, phase)
					if 'full trace, both directions' in location and not mean_std_padding:
						mean_std_padding = True
						#Padding with uniform 0-20
						for tr in traces:
							for p in tr:
								p.size += int(np.round(np.random.uniform(0, 20)))
								if p.size > 1448:
									p.size = 1448
					elif 'full trace, in direction' in location:
						directions = location.split()[4].split('->')
						src = directions[0]
						dst = directions[1]
						for tr in traces:
							for p in tr:
								if p.src == src and p.dst == dst:
									p.size += int(np.round(np.random.uniform(0, 20)))
								if p.size > 1448:
									p.size = 1448
					else:
						print('Something is wrong, this direction is not recognized;', tag, '--', location)
				elif 'Minimum of deltas in' in tag and time_injection: #Minimum of deltas in LOCATION
					location = ' '.join(tag.split()[4:])
					pass
				elif 'Maximum of deltas in' in tag and time_injection: #Maximum of deltas in LOCATION
					location = ' '.join(tag.split()[4:])
					pass
				elif 'Timing delta between first and last packet in' in tag and time_injection: #Timing delta between first and last packet in LOCATION
					location = ' '.join(tag.split()[8:])
					if 'full trace, both directions' in location:
						#Padding with uniform 0-10
						for tr in traces:
							delay = np.abs(np.random.uniform(0.0,0.100))
							tr[-1].time += delay
					elif 'full trace, in direction' in location:
						#Placeholder, add direction-based padding
						directions = location.split()[4].split('->')
						src = directions[0]
						dst = directions[1]
						for tr in traces:
							delay = np.abs(np.random.uniform(0.0,0.100))
							tr[-1].time += delay
						#for tr in traces:
						#	delay = np.abs(np.random.uniform(0.0,0.100))
						#	pkt_counter = 0
						#	for idx, p in enumerate(tr):
						#		if p.src == src and p.dst == dst:
						#			pkt_counter += 1
						#		if pkt_counter > idx1:
						#			p.time += delay
					else:
						print('Something is wrong, this direction is not recognized;', tag, '--', location)
				
				elif 'Timing delta between packets' in tag and time_injection: # Timing delta between packets 1,2 in LOCATION
					idx1 = int(tag.split()[4].split(',')[0])
					idx2 = int(tag.split()[4].split(',')[1])
					location = ' '.join(tag.split()[6:]) #Extracting packet context (direction, phase)
					if 'full trace, both directions' in location:
						#Padding with uniform 0-10
						for tr in traces:
							delay = np.abs(np.random.uniform(0.0,0.100))
							for idx, p in enumerate(tr):
								if idx > idx1:
									p.time += delay
					elif 'full trace, in direction' in location:
						#Placeholder, add direction-based padding
						directions = location.split()[4].split('->')
						src = directions[0]
						dst = directions[1]
						for tr in traces:
							delay = np.abs(np.random.uniform(0.0,0.050))
							pkt_counter = 0
							for idx, p in enumerate(tr):
								if p.src == src and p.dst == dst:
									pkt_counter += 1
								if pkt_counter > idx1:
									p.time += delay
					else:
						print('Something is wrong, this direction is not recognized;', tag, '--', location)
				elif 'Avg of deltas in' in tag or 'Std.dev. of deltas in' in tag and not mean_std_noising and time_injection:
					location = ' '.join(tag.split()[4:]) #Extracting packet context (direction, phase)
					mean_std_noising = True
					if 'full trace, both directions' in location or 'full trace, in direction' in location:
						avg_time_delay = 0.0
						time_delay_count = 0
						for tr in traces:
							average_delay = 0.0
							count = 0
							for p_ind, p in enumerate(tr[:-1]):
								delay = tr[p_ind+1].time - tr[p_ind].time
								average_delay += delay
								count += 1
							if count == 0:
								continue
							average_delay = average_delay/float(count)
							average_delay = average_delay/2.0

							if average_delay > time_injection_param: #0.500
								average_delay = time_injection_param #0.500

							for p_ind, p in enumerate(tr):
								delay = 0
								if p_ind == len(tr) - 1:
									delay = np.abs(np.random.uniform(0.0,average_delay))
									p.time += delay
								else:
									next_p = tr[p_ind]
									if p.src == next_p.dst and p.dst == next_p.src:
										delay = np.abs(np.random.uniform(0.0,average_delay))
										for pkt in tr[p_ind:]:
											pkt.time += delay
										#Keep the delta, add cumulative noise shifting all packets
									else:
										delay = np.random.uniform(0.0, next_p.time-p.time)
										p.time += delay #np.random.uniform(p.time, next_p.time)
								avg_time_delay += delay
								time_delay_count += 1
						print('Average delay per packet: {}'.format(avg_time_delay/time_delay_count))
					elif 'full trace, in direction' in location:
						pass
					else:
						print('Something is wrong, this direction is not recognized;', tag, '--', location)
				elif mean_std_padding:
					print('Already padded all packets for this tag or something similar;', tag)
				elif mean_std_noising:
					print('Already injected noise to all packets for this tag or something similar;', tag)
				else:
					print('Something is wrong, this tag is not recognized;', tag)

		#TIMING NOISE + PKT INJECTION
		if option in options_related or 'similarity' in option:
			#Time delay injection
			if time_injection:
				avg_time_delay = 0.0
				time_delay_count = 0
				for tr in traces:
					average_delay = 0.0
					count = 0
					for p_ind, p in enumerate(tr[:-1]):
						delay = tr[p_ind+1].time - tr[p_ind].time
						average_delay += delay
						count += 1
					average_delay = average_delay/float(count)
					average_delay = average_delay/2.0

					if average_delay > time_injection_param: #0.400:
						average_delay = time_injection_param #0.400

					for p_ind, p in enumerate(tr):
						delay = 0
						if p_ind == len(tr) - 1:
							delay = np.abs(np.random.uniform(0.0,average_delay))
							p.time += delay
						else:
							next_p = tr[p_ind]
							if p.src == next_p.dst and p.dst == next_p.src:
								delay = np.abs(np.random.uniform(0.0,average_delay))
								for x, pkt in enumerate(tr[p_ind:]):
									pkt.time += delay + x*np.abs(np.random.uniform(0.0,0.010))
								#Keep the delta, add cumulative noise shifting all packets
							else:
								delay = np.random.uniform(0.0, next_p.time-p.time)
								p.time += delay #np.random.uniform(p.time, next_p.time)
						avg_time_delay += delay
						time_delay_count += 1
				print('Average delay per packet: {}'.format(avg_time_delay/time_delay_count))
			
		#PACKET MODIFICATION FOR RELATED WORK
		if option in options_related and 'targeted' not in option and option != 'similarity':
			for tr in traces:
				for ind, p in enumerate(tr):
					if option == 'uniform':
						if p.size >= MTU:
							p.size = MTU
						else:
							p.size = random.randint(p.size+1, MTU)
					if option == 'uniform255':
						p.size += random.randint(1, 255)
						if p.size > MTU: p.size = MTU
					if option == 'mice-elephants':
						if p.size <= 100: p.size = 100
						else: p.size = MTU
					if option == 'linear':
						div_val = int(p.size/128)
						p.size = int(min((div_val+1)*128, MTU))
					if option == 'exp':
						if p.size <= 1:
							p.size = 1
						elif p.size == 2:
							p.size = 2
						elif p.size <= 4:
							p.size = 4
						else:
							div_val = np.log2(float(p.size))
							div_val = np.ceil(div_val)
							p.size = int(min(np.power(2, div_val), MTU))
					if option == 'mtu' or option == 'heartbeat':
						p.size = MTU
					if option == 'mtu-20ms':
						p.size = MTU
						rand_delay = np.random.uniform(0,0.020)
						p.time += rand_delay
						if ind != len(tr)-1:
							for px in tr[ind+1:]:
								px.time += rand_delay
					if option == 'p100':
						if p.size <= 100: p.size = 100
						elif p.size <= 200: p.size = 200
						elif p.size <= 300: p.size = 300
						elif p.size < 999: p.size = random.randint(p.size+1, 1000)
						elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
						else: p.size = MTU
					if option == 'p500':
						if p.size <= 500: p.size = 500
						elif p.size < 999: p.size = random.randint(p.size+1, 1000)
						elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
						else: p.size = MTU
					if option == 'p700':
						if p.size <= 700: p.size = 700
						elif p.size < 999: p.size = random.randint(p.size+1, 1000)
						elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
						else: p.size = MTU
					if option == 'p900':
						if p.size <= 900: p.size = 900
						elif p.size < 999: p.size = random.randint(p.size+1, 1000)
						elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
						else: p.size = MTU
					if option == 'pad-highest' :
						p.size = max([max([p.size for p in tr]) for tr in traces])
					
			#HEARTBEAT MODIFICATIONS FOR PACKET INJECTION AND TIMING
			if option == 'heartbeat':
				directions = set()
				
				for tr in traces:
					trace_directions = set()
					for p in tr:
						direction = (p.src,p.dst)
						if direction not in trace_directions:
							trace_directions.add(direction)
					if len(directions) == 0:
						directions = trace_directions.union(directions)
					else:
						directions - trace_directions.intersection(directions)
				print('Common directions in the traces:{}'.format(directions))

				if len(directions) != 0:
					new_traces = []
					for tr in traces:
						new_tr = tr[:]
						overhead = 0
						for ind, p in enumerate(tr):
							if len(directions) == 1:
								sample_direction = random.sample(directions, k=1) #Quick fix
							else:
								sample_direction = random.sample(directions, k=2) #Quick fix
							for direction in sample_direction:#Taking too long
								if p.src != direction[0] or p.dst != direction[1]:
									new_tr.insert(ind+overhead, Packet(direction[0], direction[1], p.sport, p.dport, p.load, p.size, p.time, p.flags))
									overhead += 1
						new_traces.append(new_tr)
					traces = new_traces
				max_packet_count = int(max([len(tr) for tr in traces]))
				print('Max trace length: {}'.format(max_packet_count))
				print('Trace lengths: {}'.format(set([len(tr) for tr in traces])))
				
				for tr in traces:
					if max_packet_count > len(tr):
						num_pkts_injected = max_packet_count - len(tr) #random.randint(0, (max_packet_count-len(tr))/2))
						if num_pkts_injected > 100:
							num_pkts_injected = 100
						for ind in range(num_pkts_injected):
							random_ind = random.sample(list(range(1,len(tr))), 1)[0]
							if len(directions) != 0:
								directions = random.sample(directions, 1)[0]
								tr.insert(random_ind, Packet(direction[0], direction[1], tr[random_ind].sport, tr[random_ind].dport, tr[random_ind].load, tr[random_ind].size, tr[random_ind].time, tr[random_ind].flags))
							else:
								direction = (tr[random_ind].src, tr[random_ind].dst)
								tr.insert(random_ind, Packet(direction[0], direction[1], tr[random_ind].sport, tr[random_ind].dport, tr[random_ind].load, tr[random_ind].size, tr[random_ind].time, tr[random_ind].flags))
				
				print('Unique trace lengths afterwards: {}'.format(set([len(tr) for tr in traces])))

				total_time_overhead = 0.0
				total_time_passed = sum([abs(tr[-1].time - tr[0].time) for tr in traces])
				max_duration = max([abs(tr[-1].time - tr[0].time) for tr in traces])
				max_num_packets = max([len(tr) for tr in traces])
				interval_list = []
				for tr in traces:
					for p1, p2 in zip(tr[0:-1], tr[1:]):
						if abs(p2.time-p1.time) > 0.01:
							interval_list.append(abs(p2.time-p1.time))
						#if abs(p2.time-p1.time) < min_interval and abs(p2.time-p1.time) > 0.01:
						#	min_interval = abs(p2.time-p1.time)
				avg_interval = np.average(interval_list)
				print('Max trace duration is:', max_duration)
				print('Min packet time interval duration is:', avg_interval)
				
				#Randomized version
				for tr in traces:
					for (c,p) in enumerate(tr):
						prob = random.uniform(0.0, 1.0)
						if prob < 0.25:
							time_padding = np.abs(np.random.uniform(0,avg_interval*5))
							for px in tr[c:]:
								px.time += time_padding
							
						#if c != len(tr) -1 and p.time > tr[c+1].time:
						#	p.time = tr[c+1].time
					#if max_num_packets > len(tr):
					#	for i in range(max_num_packets-len(tr)):
					#		c = len(tr) + i + 1
					#		if tr[-1].flags != 'M':
					#			tr += [Packet(tr[-1].src, tr[-1].dst, tr[-1].sport, tr[-1].dport, tr[-1].load, tr[-1].size, tr[0].time + c*min_interval, 'X')]
					#		else:
					#			tr += [Packet(tr[-2].src, tr[-2].dst, tr[-2].sport, tr[-2].dport, tr[-2].load, tr[-2].size, tr[0].time + c*min_interval, 'X')]
				#Padding the gaps with new packets
				#for tr in traces:
				#	new_tr = [(p.time,p) for p in tr]
				#	for i in range(len(new_tr)-1):
				#		time_diff = new_tr[i+1][0] - new_tr[i][0]
				#		pkts_to_inject = int(np.floor(time_diff/min_interval))
				#		if pkts_to_inject == 0:
				#			continue
				#		else: 
				#			tr = tr[:i+1] + [Packet(tr[i].src, tr[i].dst, tr[i].sport, tr[i].dport, tr[i].load, tr[i].size, tr[i].time, 'X') for _ in range(pkts_to_inject)] + tr[i+1:]
				if tr[-1].flags == 'M':
					tr = tr[:-1] + [Packet(tr[-2].src, tr[-2].dst, tr[-2].sport, tr[-2].dport, tr[-2].load, tr[-2].size, max_duration, 'X'), tr[-1]]
					total_time_overhead += max_duration - tr[-2].time
				else:
					tr = tr + [Packet(tr[-1].src, tr[-1].dst, tr[-1].sport, tr[-1].dport, tr[-1].load, tr[-1].size, max_duration, 'X')]
					total_time_overhead += max_duration - tr[-1].time
				print('Total time overhead = {:.2f}'.format(total_time_overhead/total_time_passed))

		if option in options_related or 'targeted' in option:
			return traces

		#OUR NOISE INJECTION ALGORITHM

		if 'gaussian' in option:
			variance = option_val
			#Packet size injection
			mean, sigma = 0, variance
			for tr in traces:
				for p in tr:
					p.size += int(np.round(np.random.uniform(mean, sigma*2)))
					#p.size += int(np.round(np.abs(np.random.normal(mean,sigma))))
					if p.size > 1448:
						p.size = 1448

			#Gaussian time delay
			if time_injection:
				avg_time_delay = 0.0
				time_delay_count = 0
				for tr in traces:

					average_delay = 0.0
					count = 0
					for p_ind, p in enumerate(tr[:-1]):
						delay = tr[p_ind+1].time - tr[p_ind].time
						average_delay += delay
						count += 1
					average_delay = average_delay/float(count)
					average_delay = average_delay/2.0

					if average_delay > 0.500:
						average_delay = 0.500

					for p_ind, p in enumerate(tr):
						delay = 0
						if p_ind == len(tr) - 1:
							delay = np.abs(np.random.uniform(0.0,average_delay))
							p.time += delay
						else:
							next_p = tr[p_ind]
							if p.src == next_p.dst and p.dst == next_p.src:
								delay = np.abs(np.random.uniform(0.0,average_delay))
								for pkt in tr[p_ind:]:
									pkt.time += delay
								#Keep the delta, add cumulative noise shifting all packets
							else:
								delay = np.random.uniform(0.0, next_p.time-p.time)
								p.time += delay #np.random.uniform(p.time, next_p.time)
						avg_time_delay += delay
						time_delay_count += 1
				print('Average delay per packet: {}'.format(avg_time_delay/time_delay_count))

			#Packet injection code
			if pkt_injection:
				avg_pkts_injected = 0.0
				pkt_count = 0
				#Random packet injection
				directions = set()
				for tr in traces:
					trace_directions = set()
					for p in tr:
						direction = (p.src,p.dst)
						if direction not in trace_directions:
							trace_directions.add(direction)
					if len(directions) == 0:
						directions = trace_directions.union(directions)
					else:
						directions - trace_directions.intersection(directions)
				
				average_tr_len = np.average([len(tr) for tr in traces])
				for tr in traces:
					trace_diff = int(np.round(len(traces)-average_tr_len)*2)
					if trace_diff <= 0:
						trace_diff = 2
					if trace_diff > 10:
						trace_diff = 10
					randomval = random.randint(0,trace_diff)
					avg_pkts_injected += randomval
					pkt_count += len(tr)
					for _ in range(randomval):
						wrong_choice = True
						while wrong_choice:
							insert_ind = random.randint(1,len(tr)-1)
							direction = None
							coin = bool(random.getrandbits(1))
							timing = 0.0
							sport = None
							dport = None
							size = None
							load = None
							flags = None
							if len(directions) != 0:
								direction = random.sample(directions, 1)[0]
								
								#if insert_ind != len(tr) - 1:
								#	timing = random.uniform(tr[insert_ind].time, tr[insert_ind+1].time)
								#else:
								#	timing = tr[insert_ind].time
								#
								#tr.insert(insert_ind, Packet(direction[0], direction[1], tr[insert_ind].sport, tr[insert_ind].dport, tr[insert_ind].load, tr[insert_ind].size, tr[insert_ind].time, tr[insert_ind].flags))
							#if True:
							if insert_ind == len(tr)-1 or coin:
								#direction = (tr[insert_ind].src, tr[insert_ind].dst)
								sport = tr[insert_ind].sport
								dport = tr[insert_ind].dport
								if tr[insert_ind].size > 1:
									size = random.randint(1, tr[insert_ind].size)
								else:
									size = tr[insert_ind].size
								load  = tr[insert_ind].load
								flags = tr[insert_ind].flags
							else: 
								#direction = (tr[insert_ind+1].src, tr[insert_ind+1].dst)
								sport = tr[insert_ind+1].sport
								dport = tr[insert_ind+1].dport
								if tr[insert_ind+1].size > 1:
									size = random.randint(1, tr[insert_ind+1].size)
								else:
									size = tr[insert_ind+1].size
								load  = tr[insert_ind+1].load
								flags = tr[insert_ind+1].flags
							if 'M' in flags:
								wrong_choice = True
								continue
							else:
								wrong_choice = False
							
							if insert_ind != len(tr) - 1:
								timing = random.uniform(tr[insert_ind].time, tr[insert_ind+1].time)
							else:
								timing = tr[insert_ind].time
							tr.insert(insert_ind, Packet(direction[0], direction[1], sport, dport, load, size, timing, flags))

				
				print('Average number of pkts injected: {:.2f}'.format(float(avg_pkts_injected)/pkt_count))



		return traces

	@classmethod
	def calculate_bayes_error(cls, labels, features, tags, target_tags):
		print('Calculating Accuracy Bound based on Bayes Error')
		#print('Number of features:{}, Number of corresponding labels:{}'.format(len(features), len(labels)))
		#print('Number of data points:{}'.format(len(features[0])))
		#print('Number of tags:{}'.format(len(tags)))
		#print('Number of targeted features: {}'.format(len(target_tags)))
		target_tags = target_tags[:25]
		epsilon = np.power(10.0,-10)
		min_var = 0.001
		num_points = 10000

		option = 'kde'#'norm'
		labels_list = sorted( list(set(labels)) )
		print(labels_list)
		p_x_given_secret = [[] for _ in labels_list]
		testing_score = [[] for _ in labels_list]
		accuracy_bound_list = []

		bound_correctness = 0
		bound_incorrectness = 0
		

		#features_max = [max(f) for f in features]
		#features_norm = [[float(x)/max_f if max_f>0 else float(x) for x in f] for (f, max_f) in zip(features, features_max)]
		#features_t = list(zip(*features))
		#X = features_t
		y = labels
		#pairs = list(zip(features_t, labels))
		#
		###Balanced Shuffle:
		#pairs_partitioned = [list() for _ in range(len(labels_list))]
		#for pair in pairs:
		#	for i, l in enumerate(labels_list):
		#		if pair[1] == l:
		#			pairs_partitioned[i].append(pair)
		#
		#ratio = 0.8
		#
		#X_train = []
		#X_test  = []
		#y_train = []
		#y_test  = []
		#
		#for pairs_list in pairs_partitioned:
		#	random.shuffle(pairs_list)
		#	X_train += [x[0] for x in pairs_list[:round(len(pairs_list)*ratio)]]
		#	X_test  += [x[0] for x in pairs_list[round(len(pairs_list)*ratio):]]
		#	y_train += [x[1] for x in pairs_list[:round(len(pairs_list)*ratio)]]
		#	y_test  += [x[1] for x in pairs_list[round(len(pairs_list)*ratio):]]
		#
		#features_train = list(zip(*X_train))
		#features_test = list(zip(*X_test))

		#Divide to train/test
		#Classify for each tag
		#two_dim_computation_points = []

		for target_tag in target_tags:
			for i, tag in enumerate(tags):
				if target_tag == tag:
					target_features = features[i] #features_train[i]
					test_features   = features[i]   #features_test[i]
					#print(len(target_features))
					#print(len(test_features))
					min_features = min(target_features)
					max_features = max(target_features)
					min_max_range = max_features - min_features
					sample_points = np.linspace(min_features-0.05*min_max_range, max_features+0.05*min_max_range, num_points)
					features_per_class = [[] for _ in labels_list]
					for f, l in zip(target_features, y):
						features_per_class[l].append(f)

					for l in labels_list:
						if option == 'kde':
							xs = np.array(features_per_class[l])
							xs = xs.reshape(-1, 1)

							if len(features_per_class[l]) == 0:
								std_bdwidth = min_var
							else:
								std_bdwidth = 1.06 * np.std(features_per_class[l]) / np.power(float(len(features_per_class[l])), 0.2)
							if std_bdwidth < min_var:
								std_bdwidth = min_var

							
							bandwidths = np.array([std_bdwidth, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])

							if len(xs) > 1:
								#print('Running grid search with 3 times repeated 5-fold cross-validation, parallelized')
								grid = GridSearchCV(KernelDensity(kernel='epanechnikov'),
									param_grid={'bandwidth': bandwidths}, n_jobs = -1,
									cv=KFold(n_splits = min(5, len(xs)), shuffle=True))
									#cv=RepeatedKFold(n_splits=min(5, len(xs)), n_repeats=3))
									#cv=ShuffleSplit(n_splits=5, test_size=0.20))
								grid.fit(xs)
								bdwidth = grid.best_params_['bandwidth']

							kde = KernelDensity(kernel='epanechnikov', bandwidth=bdwidth).fit(xs)

							p_x_given_secret[l] = kde.score_samples(np.array(sample_points)[:,np.newaxis])
							p_x_given_secret[l] = normalize(np.exp(p_x_given_secret[l])[:,np.newaxis],norm='l1', axis=0)
							testing_score[l] = kde.score_samples(np.array(test_features)[:,np.newaxis])
							#testing_score[l] = normalize(np.exp(p_x_given_secret[l])[:,np.newaxis],norm='l1', axis=0)

						elif option == 'norm':
							(loc, scale) = norm.fit(np.array(features_per_class[l]))
							if scale < min_var:
								scale = min_var

							p_x_given_secret[l] = norm.pdf(sample_points,loc=loc,scale=scale)
							p_x_given_secret[l] = normalize(p_x_given_secret[l][:,np.newaxis],norm='l1', axis=0)
							testing_score[l] = norm.pdf(np.array(test_features),loc=loc,scale=scale)

							#testing_score[l] = normalize(np.exp(p_x_given_secret[l])[:,np.newaxis],norm='l1', axis=0)
						elif option == 'kde-dynamic':
							continue
						else:
							continue
					#two_dim_computation_points.append(p_x_given_secret)
					
					accuracy_bound = 0.0
					#Add the p(x|s) for the max p(x|s), add them all up, multiply with p(s)
					for j, _ in enumerate(sample_points):
						accuracy_bound += max([p_x_given_secret[l][j][0] for l in labels_list]) 
					accuracy_bound = accuracy_bound/float(len(labels_list)) #Multiply with p(s) assuming equal distribution

					accuracy_bound_list.append(accuracy_bound)

					real_accuracy = 0.0
					for j, (val, label) in enumerate(zip(test_features,y)):
						classified_class = np.argmax([testing_score[l][j] for l in labels_list])
						#print(correct_class, label)
						if classified_class == label:
							real_accuracy += 1.0
					real_accuracy = real_accuracy/float(len(y))
					printed_str = 'Accuracy bound and real accuracy for feature {}: {:.2f}% - {:.2f}%'.format(target_tag, accuracy_bound*100.0, real_accuracy*100.0)
					if real_accuracy > accuracy_bound:
						bound_incorrectness += 1
						printed_str = '!!! ' + printed_str
					else:
						bound_correctness += 1
					print(printed_str)

		print('Number of features where accuracy upper bound holds : {}/{}'.format(bound_correctness, bound_correctness+bound_incorrectness))

		#if False:
		#	#Two dimensional accuracy bound computation
		#	accuracy_bound = 0.0
		#	#Add the p(x|s) for the max p(x|s), add them all up, multiply with p(s)
		#	f1_p_x_given_secret = two_dim_computation_points[0]
		#	f2_p_x_given_secret = two_dim_computation_points[0]
		#	for j1, _ in enumerate(sample_points):
		#		for j2, _ in enumerate(sample_points):
		#			accuracy_bound += max([f1_p_x_given_secret[l][j1][0]*f2_p_x_given_secret[l][j2][0] for l in labels_list])
		#	accuracy_bound = accuracy_bound/float(len(labels_list))
		#	accuracy_bound_list.append(accuracy_bound)
		#	print('Accuracy bound for two features: {:.2f}%'.format(accuracy_bound*100.0))
		#	#p(x1,x2,s) = p(x1|x2,s)*p(x2|s)*p(s) = p(x1|s)*p(x2|s)*p(s) #Assuming independence of features
		#	#p(x1,x2,s) = p(x1,x2|s)*p(s) => p(x1,x2|s) = p(x1|s)*p(x2|s)
		
		accuracy_bound = max(accuracy_bound_list)

		return accuracy_bound

	@classmethod
	def extract_location(cls, location_str):
		src_dst_str = location_str.split()[-1].split('->')
		src_ip = src_dst_str[0]
		dst_ip = src_dst_str[1]
		return (src_ip, dst_ip)

	@classmethod
	def modify(cls, traces, feature_tag, param = 0.020):
		MAX_VAL = 1500
		MTU = MAX_VAL
		tag = feature_tag
		if 'Size of packet' in tag: #Size of packet INDEX in LOCATION
			pkt_ind = int(tag.split()[3]) - 1
			location = ' '.join(tag.split()[5:])
			if 'full trace, both directions' in location or 'interval' in location:
				target_val = max([t[pkt_ind].size for t in traces if len(t) > pkt_ind])
				for t in traces:
					if len(t) > pkt_ind:
						t[pkt_ind].size = target_val if (t[pkt_ind].size <= target_val) else random.randint(t[pkt_ind].size, MAX_VAL)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				target_val = 0
				#Finding Max size of Pkt[i] in location
				for t in traces:
					pkt_num_counter = 0
					for p in t:
						if p.src == src_ip and p.dst == dst_ip:
							pkt_num_counter += 1
							if (pkt_num_counter-1) == pkt_ind and p.size > target_val:
								target_val = p.size
								break
							elif (pkt_num_counter-1) == pkt_ind:
								break

				#Applying the modification to all traces
				for t in traces:
					pkt_num_counter = 0
					for p in t:
						if p.src == src_ip and p.dst == dst_ip:
							pkt_num_counter += 1
							if (pkt_num_counter-1) == pkt_ind and p.size < target_val:
								p.size = target_val
								break
		elif 'Number of packets with size' in tag: #Number of packets with size SIZE in LOCATION
			#TODO Change this, it's wrong, we need to inject packets, not pad them.
			size = int(tag.split()[5])
			location = ' '.join(tag.split()[7:])
			if 'full trace, both directions' in location or 'interval' in location:
				max_num_pkts = max([len([p for p in t if p.size == size]) for t in traces])
				min_num_pkts = min([len([p for p in t if p.size == size]) for t in traces])
				pkt_diff = int(2*(max_num_pkts - min_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_loc = random.randint(1, len(t)-1)
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						pkt.size = size
						#pkt.time = t[rand_loc-1].time
						traces[ind].insert(rand_loc, pkt)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				max_num_pkts = max([len([p for p in t if p.size == size and p.src == src_ip and p.dst == dst_ip]) for t in traces])
				min_num_pkts = min([len([p for p in t if p.size == size and p.src == src_ip and p.dst == dst_ip]) for t in traces])
				pkt_diff = int(2*(max_num_pkts - min_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_loc = random.randint(1, len(t)-1)
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						pkt.size = size
						pkt.src = src_ip
						pkt.dst = dst_ip
						#pkt.time = t[rand_loc-1].time
						traces[ind].insert(rand_loc, pkt)
		elif 'Number of packets in' in tag: #Number of packets in LOCATION
			location = ' '.join(tag.split()[4:])
			#Find the max number of packets, find the average per trace
			#Inject packets randomly equal to the difference to equalize the leakage
			if 'full trace, both directions' in location or 'interval' in location:
				max_num_pkts = max([len(t) for t in traces])
				avg_num_pkts = np.mean([len(t) for t in traces])
				pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_loc = random.randint(1, len(t)-1)
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						if pkt.size > 1:
							pkt.size = random.randint(1, pkt.size)
						#pkt.time = t[rand_loc-1].time
						traces[ind].insert(rand_pkt, pkt)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location) #Inject packets with src->dst combination
				max_num_pkts = max([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
				avg_num_pkts = np.mean([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
				pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
				#new_traces = copy.deepcopy(mod_traces)
				for ind,t in enumerate(traces):
					rand_val = random.randint(0, pkt_diff)
					for _ in range(rand_val):
						rand_pkt = random.randint(1, len(t)-1)
						pkt = copy.deepcopy(t[rand_pkt])
						if pkt.size > 1:
							pkt.size = random.randint(1, pkt.size)
						pkt.src = src_ip
						pkt.dst = dst_ip
						traces[ind].insert(rand_pkt, pkt)
		elif 'Minimum of sizes in' in tag: #Minimum of sizes in LOCATION
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location or 'interval' in location:
				max_min_size = max([min([p.size for p in t]) for t in traces])
				for t in traces:
					for p in t:
						if p.size < max_min_size:
							p.size = max_min_size
						if p.size > 1500:
							p.size = 1500
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location) #Inject packets with src->dst combination
				new_traces = [[p.size for p in t if p.src==src_ip and p.dst==dst_ip] for t in traces]
				min_fn = lambda x: min(x) if len(x) > 0 else 0
				max_min_size = max([min_fn(t) for t in new_traces])
				for t in traces:
					for p in t:
						if p.size < max_min_size and p.src == src_ip and p.dst == dst_ip:
							p.size = max_min_size
						if p.size > 1500:
							p.size = 1500
		elif 'Maximum of sizes in' in tag: #Maximum of sizes in LOCATION
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location or 'interval' in location:
				max_size = max([max([p.size for p in t]) for t in traces])
				for ind,t in enumerate(traces):
					rand_loc = random.randint(1, len(t)-1)
					pkt = copy.deepcopy(t[rand_loc])
					pkt.size = max_size
					#pkt.time = t[rand_loc-1].time
					traces[ind].insert(rand_loc, pkt)
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location) #Inject packets with src->dst combination
				new_traces = [[p.size for p in t if p.src==src_ip and p.dst==dst_ip] for t in traces]
				max_fn = lambda x: max(x) if len(x) > 0 else 0
				max_size = max([max_fn(t) for t in new_traces])
				for ind,t in enumerate(traces):
					rand_loc = random.randint(1, len(t)-1)
					pkt = copy.deepcopy(t[rand_loc])
					pkt.size = max_size
					pkt.src = src_ip
					pkt.dst = dst_ip
					traces[ind].insert(rand_loc, pkt)
		elif 'Sum of sizes in' in tag: #Sum of sizes in LOCATION #AGGREGATE
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location:
				pkt_size_padding = None
				if param is None:
					list_sizes = [sum([p.size for p in t]) for t in traces]
					size_difference = np.mean(list_sizes) - min(list_sizes)
					list_num_pkts = [len(t) for t in traces]
					avg_num_pkts = np.mean(list_num_pkts)
					pkt_size_padding = int(size_difference/(2*avg_num_pkts))
					print("Size difference per packet: {}".format(pkt_size_padding))
				elif param == 'exp':
					for t in traces:
						for p in t:
							if p.size <= 1:
								p.size = 1
							elif p.size == 2:
								p.size = 2
							elif p.size <= 4:
								p.size = 4
							else:
								div_val = np.log2(float(p.size))
								div_val = np.ceil(div_val)
								p.size = int(min(np.power(2, div_val), 1500))
				elif param == 'inject':
					max_num_pkts = max([len(t) for t in traces])
					avg_num_pkts = np.mean([len(t) for t in traces])
					pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
					#new_traces = copy.deepcopy(mod_traces)
					for ind,t in enumerate(traces):
						rand_val = random.randint(0, pkt_diff)
						for _ in range(rand_val):
							rand_loc = random.randint(1, len(t)-1)
							rand_pkt = random.randint(1, len(t)-1)
							pkt = copy.deepcopy(t[rand_pkt])
							pkt.size = random.randint(1, MAX_VAL)
							#pkt.time = t[rand_loc-1].time
							traces[ind].insert(rand_loc, pkt)
				elif param == 'uniform':
					for t in traces:
						for p in t:
							if p.size >= MTU:
								p.size = MTU
							else:
								p.size = random.randint(p.size+1, MTU)
				elif param == 'uniform255':
					for t in traces:
						for p in t:
							p.size += random.randint(1, 255)
							if p.size > MTU: p.size = MTU
				elif param == 'mice-elephants':
					for t in traces:
						for p in t:
							if p.size <= 100: p.size = 100
							else: p.size = MTU
				elif param == 'linear':
					for t in traces:
						for p in t:
							div_val = int(p.size/128)
							p.size = int(min((div_val+1)*128, MTU))
				elif param == 'mtu':
					for t in traces:
						for p in t:
							p.size = MTU
				elif param == 'p100':
					for t in traces:
						for p in t:
							if p.size <= 100: p.size = 100
							elif p.size <= 200: p.size = 200
							elif p.size <= 300: p.size = 300
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
				elif param == 'p500':
					for t in traces:
						for p in t:
							if p.size <= 500: p.size = 500
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
				elif param == 'p700':
					for t in traces:
						for p in t:
							if p.size <= 700: p.size = 700
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
				elif param == 'p900':
					for t in traces:
						for p in t:
							if p.size <= 900: p.size = 900
							elif p.size < 999: p.size = random.randint(p.size+1, 1000)
							elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
							else: p.size = MTU
				else:
					pkt_size_padding = int(param)
					if pkt_size_padding > 0:
						for t in traces:
							for p in t:
								p.size = p.size + random.randint(0,pkt_size_padding)
								if p.size > 1500:
									p.size = 1500
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				pkt_size_padding = None
				if param is None:
					list_sizes = [sum([p.size for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces]
					size_difference = np.mean(list_sizes) - min(list_sizes)
					list_num_pkts = [len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces]
					avg_num_pkts = np.mean(list_num_pkts)
					pkt_size_padding = int(size_difference/(2*avg_num_pkts))
					print("Size difference per packet: {}".format(pkt_size_padding))
				elif param == 'exp':
					for t in traces:
						for p in t:
							if p.src != src_ip or p.dst != dst_ip:
								continue
							if p.size <= 1:
								p.size = 1
							elif p.size == 2:
								p.size = 2
							elif p.size <= 4:
								p.size = 4
							else:
								div_val = np.log2(float(p.size))
								div_val = np.ceil(div_val)
								p.size = int(min(np.power(2, div_val), 1500))
				elif param == 'inject':
					max_num_pkts = max([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
					avg_num_pkts = np.mean([len([p for p in t if p.src == src_ip and p.dst == dst_ip]) for t in traces])
					pkt_diff = int(2*(max_num_pkts - avg_num_pkts))
					#new_traces = copy.deepcopy(mod_traces)
					for ind,t in enumerate(traces):
						rand_val = random.randint(0, pkt_diff)
						for _ in range(rand_val):
							rand_loc = random.randint(1, len(t)-1)
							rand_pkt = random.randint(1, len(t)-1)
							pkt = copy.deepcopy(t[rand_pkt])
							pkt.size = random.randint(1, MAX_VAL)
							pkt.src = src_ip
							pkt.dst = dst_ip
							#pkt.time = t[rand_loc-1].time
							traces[ind].insert(rand_loc, pkt)
				elif param == 'uniform':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size >= MTU:
									p.size = MTU
								else:
									p.size = random.randint(p.size+1, MTU)
				elif param == 'uniform255':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								p.size += random.randint(1, 255)
								if p.size > MTU: p.size = MTU
				elif param == 'mice-elephants':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 100: p.size = 100
								else: p.size = MTU
				elif param == 'linear':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								div_val = int(p.size/128)
								p.size = int(min((div_val+1)*128, MTU))
				elif param == 'mtu':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								p.size = MTU
				elif param == 'p100':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 100: p.size = 100
								elif p.size <= 200: p.size = 200
								elif p.size <= 300: p.size = 300
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				elif param == 'p500':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 500: p.size = 500
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				elif param == 'p700':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 700: p.size = 700
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				elif param == 'p900':
					for t in traces:
						for p in t:
							if p.src == src_ip and p.dst == dst_ip:
								if p.size <= 900: p.size = 900
								elif p.size < 999: p.size = random.randint(p.size+1, 1000)
								elif p.size <= 1399: p.size = random.randint(p.size+1, 1400)
								else: p.size = MTU
				else:
					pkt_size_padding = int(param)
					if pkt_size_padding > 0:
						for t in traces:#TODO Add search for modifying total size
							for p in t:
								if p.src == src_ip and p.dst == dst_ip:
									p.size = p.size + random.randint(0,pkt_size_padding)
									if p.size > 1500:
										p.size = 1500
		elif 'Mean of sizes in' in tag: #Mean of sizes in LOCATION #AGGREGATE
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location:
				pass
			elif 'full trace, in direction'  in location:
				pass
		elif 'Std.dev. of sizes in' in tag: #Std.dev. of sizes in LOCATION #AGGREGATE
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location:
				pass
			elif 'full trace, in direction'  in location:
				pass
		###TIMING SIDE-CHANNELS
		elif 'Timing delta between first and last packet in' in tag or 'Avg of deltas in' in tag or 'Std.dev. of deltas in' in tag or 'Maximum of deltas in' in tag or 'Minimum of deltas in' in tag: #Timing delta between first and last packet in LOCATION
			#location = ' '.join(tag.split()[8:]) #'full trace, both directions' in location or 'full trace, in direction' in location:
			if param == '20ms':
				for t in traces:
					len_t = len(t)
					for ind, p in enumerate(t):
						rand_delay = np.random.uniform(0,0.020)
						p.time += rand_delay
						if ind != len_t-1:
							for px in t[ind+1:]:
								px.time += rand_delay
			elif param == '10ms':
				for t in traces:
					len_t = len(t)
					for ind, p in enumerate(t):
						rand_delay = np.random.uniform(0,0.010)
						p.time += rand_delay
						if ind != len_t-1:
							for px in t[ind+1:]:
								px.time += rand_delay
			else:
				avg_time_delay = 0.0
				time_delay_count = 0
				for tr in traces:
					average_delay = 0.0
					count = 0
					for p_ind, p in enumerate(tr[:-1]):
						delay = tr[p_ind+1].time - tr[p_ind].time
						average_delay += delay
						count += 1
					average_delay = average_delay/float(count)
					average_delay = average_delay/2.0

					if average_delay > param: #0.400:
						average_delay = param #0.400

					for p_ind, p in enumerate(tr):
						delay = 0
						if p_ind == len(tr) - 1:
							delay = np.abs(np.random.uniform(0.0,average_delay))
							p.time += delay
						else:
							next_p = tr[p_ind]
							if p.src == next_p.dst and p.dst == next_p.src:
								delay = np.abs(np.random.uniform(0.0,average_delay))
								for x, pkt in enumerate(tr[p_ind:]):
									pkt.time += delay + x*np.abs(np.random.uniform(0.0,0.010))
								#Keep the delta, add cumulative noise shifting all packets
							else:
								delay = np.random.uniform(0.0, next_p.time-p.time)
								p.time += delay #np.random.uniform(p.time, next_p.time)
						avg_time_delay += delay
						time_delay_count += 1
				#print('Average delay per packet: {}'.format(avg_time_delay/time_delay_count))
			if False:
				src_ip, dst_ip = cls.extract_location(location)
				pass
		elif 'Timing delta between packets' in tag: #Timing delta between packets 1,2 in LOCATION
			location = ' '.join(tag.split()[8:])
			#Similar to size of pkt i, equalize this measure by padding them
			indices = tag.split()[4]
			ind1 = int(indices.split(",")[0])
			ind2 = int(indices.split(",")[1])
			if 'full trace, both directions' in location or 'full trace, in direction' in location:
				for t in traces:
					delay_val = random.uniform(0, 0.100)
					for i, p in enumerate(t):
						if i >= ind2:
							p.time += delay_val
			if False:
				src_ip, dst_ip = cls.extract_location(location)
				pass
		elif 'Minimum of deltas in' in tag: #Minimum of deltas in LOCATION
			location = ' '.join(tag.split()[4:])
			#TODO Similar to minimum sizes, delay packets to equalize this
			if 'full trace, both directions' in location:
				pass
			elif 'full trace, in direction'  in location:
				src_ip, dst_ip = cls.extract_location(location)
				pass
		elif 'Maximum of deltas in' in tag: #Maximum of deltas in LOCATION
			location = ' '.join(tag.split()[4:])
			#TODO Similar to maximum sizes, delay packets to equalize this
			if 'full trace, both directions' in location:
				pass
			elif 'full trace, in direction'  in location:
				src_ip, dst_ip = cls.extract_location(location)
				pass
		elif 'Avg of deltas in' in tag: #Avg of deltas in LOCATION
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location:
				pass
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				pass
		elif 'Std.dev. of deltas in' in tag: #Std.dev. of deltas in LOCATION
			location = ' '.join(tag.split()[4:])
			if 'full trace, both directions' in location:
				pass
			elif 'full trace, in direction' in location:
				src_ip, dst_ip = cls.extract_location(location)
				pass
	
		return traces

	@classmethod
	def targeted_defense(cls, traces, trace_filename, weights, calculate_bounds = False):
		#SETTING THE INITIAL PARAMETERS
		calcSpace = True
		calcTime = True
		pre_time = time.time()
		rep_count = 1
		alignment = False
		silence = True
		options_leakage = []
		feature_reduction = None
		#classifier = 'kde-dynamic'
		classifier = 'rf-classifier'
		print('Feature Reduction Method: {}'.format(feature_reduction))

		w_leakage = weights[0] #1.0
		w_overhead = weights[1] #0.1
		w_toverhead = weights[2] #0.0
		min_objective_fn = 1000

		#Pruning packets with size > 1500, don't know how that's possible. #Change this so that either we merge all packets or split all packets, it's inconsistent right now.
		full_traces = []
		for t in traces:
			new_t = []
			for p in t:
				if p.size <= 1500:
					new_t.append(p)
				elif p.size > 1500:
					old_size = p.size
					while old_size > 1500:
						new_p = copy.deepcopy(p)
						old_size = old_size - 1500
						new_p.size = 1500
						new_t.append(new_p)
					new_p = copy.deepcopy(p)
					new_p.size = old_size
					new_t.append(new_p)
			if len(new_t) > 0:
				full_traces.append(new_t)

		print(len(traces))
		print(len(full_traces))
		print("Trace length distribution: {}".format(set([len(t) for t in traces])))
		print('Number of packets with size > 1500 per trace:', sum([ len([p for p in tr if p.size > 1500]) for tr in traces])/float(len(traces)))
		print('Number of packets with size > 1500 per trace in pruned traces:', sum([ len([p for p in tr if p.size > 1500]) for tr in full_traces])/float(len(full_traces)))
		print('Number of traces with 1 packets per trace:', sum([ 1 for t in traces if len(t) <= 1]))

		test_traces = []
		train_traces = []

		labels = Transform.rd_secrets(full_traces)
		labels_list = list(set(labels))
		numbers_list = list(range(len(labels_list)))
		labels_to_numbers = {k: v for k, v in zip(labels_list, numbers_list)}
		traces_per_label = [[] for _ in labels_list]
		
		print('Set of labels: {}'.format(labels_list))

		for l, tr in zip(labels, full_traces):
			ind = labels_to_numbers[l]
			traces_per_label[ind].append(tr)
		for l in labels_list:
			ind = labels_to_numbers[l]
			list_length = len(traces_per_label[ind])
			train_traces += traces_per_label[ind][:int(list_length/2)]
			test_traces  += traces_per_label[ind][int(list_length/2):]
		print('Number of full traces: {}'.format(len(full_traces)))
		print('Number of train/test traces: {}, {}'.format(len(train_traces), len(test_traces)))
		#Divide traces to labels, send equal to both parts

		import warnings
		warnings.filterwarnings("ignore")

		#INITIAL RUN FOR NO-PADDING
		quant_time = time.time()
		print('Classifier: {}'.format(classifier))
		classifier1 = 'rf-classifier' #'kde-dynamic'
		(labels, features, tags, orig_leakage, feature_importance) = cls.process_all(interactions=train_traces, pcap_filename=None, calcSpace=calcSpace, calcTime=calcTime, quant_mode=classifier1, window_size=None, 
		feature_reduction=feature_reduction, num_reduced_features=10, alignment=alignment, new_direction=False, silent=False)

		print("INITIAL QUANT TIME: {:.2f} seconds".format(time.time()-quant_time))
		options_leakage.append(('No-mitigation', orig_leakage, 0.0, 0.0))
		target_tags = [x[1] for x in feature_importance]

		print("ALL FEATURES: {}".format(target_tags))

		if calculate_bounds:
			accuracy_bound = cls.calculate_bayes_error(labels, features, tags, target_tags)
			print('Accuracy Bound for Option {}: {:.2f}'.format('No-mitigation', accuracy_bound))

		print('Leakage for Option {}: {:.2f}'.format('No-mitigation', orig_leakage))
		print('Overhead for Option {}: {:.2f}'.format('No-mitigation', 0.0))
		print('Time Overhead for Option {}: {:.2f}'.format('No-mitigation', 0.0))
		print('RANDOM GUESS ACCURACY for {} classes: {:.2f}'.format(len(labels_list), 1.0/len(labels_list)))
		print('='*40)
		print('%'*40)

		min_objective_fn = w_leakage*1.0 + w_overhead*0.0 + w_toverhead*0.0

		total_size_orig = 0
		total_time_orig = 0.0
		for t in train_traces:
			total_time_orig += abs(t[-1].time - t[0].time)
			for p in t:
				total_size_orig += p.size

		#Loop for Mitigation Strategy Synthesis
		non_improvement_count = 0
		non_improvement_limit = 20
		old_target_tags = set()
		t_traces = copy.deepcopy(train_traces) #T

		for i in range(len(target_tags)):
			#Step 1: Target top feature, use distribution to find the padding style, distribute it to the packets if aggregate
			#Select top feature
			tag = None
			if w_toverhead < 10:
				for t in target_tags:
					if t not in old_target_tags: #"Mean" not in t and "Avg" not in t and "Std" not in t and "delta" not in t and 
						tag = t
						break
			else:
				for t in target_tags:
					if t not in old_target_tags and "delta" not in t: #"Mean" not in t and "Avg" not in t and "Std" not in t and "delta" not in t and 
						tag = t
						break
			if tag is None:
				print("Went over all the tags, early termination!")
				return None
			old_target_tags.add(tag)
			
			#Modify Trace
			new_traces_list = []
			if 'Sum of sizes in' in tag:
				for param in [None, 50, 100, 150, 200, 250, 'inject', 'exp', 'linear', 'uniform', 'uniform255', 'mice-elephants', 'mtu', 'p100', 'p500', 'p700', 'p900']:
					print("Targeting tag: {}, parameter: {}".format(tag, param))
					new_traces = cls.modify(copy.deepcopy(t_traces), tag, param) # T' = modify(T, feature)
					new_traces_list.append(new_traces)
			elif 'Timing delta between first and last packet in' in tag or 'Avg of deltas in' in tag or 'Std.dev. of deltas in' in tag or 'Maximum of deltas in' in tag or 'Minimum of deltas in' in tag:
				for param in ['10ms', '20ms', 0.010, 0.020, 0.050, 0.100, 0.200, 0.300]:
					print("Targeting tag: {}, parameter: {}".format(tag, param))
					new_traces = cls.modify(copy.deepcopy(t_traces), tag, param) # T' = modify(T, feature)
					new_traces_list.append(new_traces)
			else:
				print("Targeting tag: {}".format(tag))
				new_traces = cls.modify(copy.deepcopy(t_traces), tag, None) # T' = modify(T, feature)
				new_traces_list.append(new_traces)

			for new_traces in new_traces_list:
				#Step 2: Quantify and Check if the modification improves the privacy
				avg_leakage = 0.0
				for rep in range(rep_count):
					(labels, features, tags, leakage, feature_importance) = cls.process_all(interactions=new_traces, pcap_filename=None, calcSpace=calcSpace, calcTime=calcTime, quant_mode=classifier, window_size=None, 
					feature_reduction=feature_reduction, num_reduced_features=10, alignment=alignment, new_direction=False, silent=silence)
					avg_leakage += float(leakage)/rep_count


				#target_tags = [x[1] for x in feature_importance]
				#Calculating the Average Overhead against the original traces
				total_size_mod = 0
				total_time_mod = 0.0
				for t in new_traces:
					total_time_mod += abs(t[-1].time - t[0].time)
					for p in t:
						total_size_mod += p.size

				overhead_mod = float(total_size_mod-total_size_orig)/float(total_size_orig)
				t_overhead_mod = float(total_time_mod-total_time_orig)/float(total_time_orig)
				#abs_overhead_mod += float(total_size_mod-total_size_orig)/(len(mod_traces)*float(rep_count))

				if overhead_mod < 0:
					print("OVERHEAD less than 0!")
					overhead_mod = 0.0
				if t_overhead_mod < 0:
					print("TIME OVERHEAD less than 0!")
					t_overhead_mod = 0.0
				objective_fn = w_leakage*avg_leakage + w_overhead*overhead_mod + w_toverhead*t_overhead_mod

				#Save the results and print to screen
				method_name = 'Targeted_Mitigation_Step_{}'.format(i)
				options_leakage.append((method_name, avg_leakage, overhead_mod, t_overhead_mod))
				#target_tags = [x[1] for x in feature_importance]

				#Bound calculation???
				#if calculate_bounds:
				#	accuracy_bound = cls.calculate_bayes_error(labels, features, tags, target_tags)
				#	print('Accuracy Bound for Option {}: {:.2f}'.format(method_name, accuracy_bound))

				print('Total size: {}, Original size: {}'.format(total_size_mod, total_size_orig))
				print('Total time: {}, Original time: {}'.format(total_time_mod, total_time_orig))

				print('Leakage for Option {}: {:.2f}'.format(method_name, avg_leakage))
				print('Overhead for Option {}: {:.2f}'.format(method_name, overhead_mod))
				print('Time Overhead for Option {}: {:.2f}'.format(method_name, t_overhead_mod))
				print('ObjectiveFN for Option {}: {:.2f}'.format(method_name, objective_fn))
				
				if objective_fn < min_objective_fn:
					t_traces = copy.deepcopy(new_traces) #T <- T'
					min_objective_fn = objective_fn 
					non_improvement_count = 0
					print("Improving minimization of the Objective Function {}*leakage + {}*space + {}*time".format(w_leakage, w_overhead, w_toverhead))
				else:
					non_improvement_count += 1
					print("Not improving minimization, count: {}".format(non_improvement_count))
			
			if non_improvement_count >= non_improvement_limit:
				print("Early Termination!")
				return None
			print('='*40)
			
	
###################################################################################


	@classmethod
	def generate_defense(cls, traces, filename, weights_list, run_related = True, run_gs = True, run_sa = False, time_injection=[False], calculate_bounds = False):#l_weight = 1.0, oh_weight = 0.1):

		#TODO 
		#Modify targeted mitigation such that after each application, the random forest generates new rankings
		#Modify targeted mitigation such that it uses Gaussian/KDE quantification, might want to compare them.

		#TODO Quantification: Change this so that ranking and results use quantification, NOT CLASSIFICATION
		#TODO Have classification at the end with Precision, Recall and F-Score metrics as well.

		#SETTING THE INITIAL PARAMETERS
		pre_time = time.time()
		options_leakage = []
		rep_count = 3
		feature_reduction = None
		classifier = 'rf-classifier'
		print('Feature Reduction Method: {}'.format(feature_reduction))
		time_injection_list = time_injection
		pkt_injection_list = [False] #[True, False]

		#Pruning packets with size > 1500, don't know how that's possible.
		full_traces = [ [p for p in tr if p.size <= 1500] for tr in traces]
		print(len(traces))
		print(len(full_traces))
		print('Number of packets with size > 1500 per trace:', sum([ len([p for p in tr if p.size > 1500]) for tr in traces])/float(len(traces)))

		test_traces = []
		train_traces = []

		labels = Transform.rd_secrets(full_traces)
		labels_list = list(set(labels))
		numbers_list = list(range(len(labels_list)))
		labels_to_numbers = {k: v for k, v in zip(labels_list, numbers_list)}
		traces_per_label = [[] for l in labels_list]
		
		print('Set of labels: {}'.format(labels_list))

		#print('Labels: {}'.format(labels))
		for l, tr in zip(labels, full_traces):
			ind = labels_to_numbers[l]
			traces_per_label[ind].append(tr)
		for l in labels_list:
			ind = labels_to_numbers[l]
			list_length = len(traces_per_label[ind])
			train_traces += traces_per_label[ind][:int(list_length/2)]
			test_traces  += traces_per_label[ind][int(list_length/2):]
		print('Number of full traces: {}'.format(len(full_traces)))
		print('Number of train/test traces: {}, {}'.format(len(train_traces), len(test_traces)))
		#Divide traces to labels, send equal to both parts
		
		#INITIAL RUN FOR NO-PADDING
		print('Classifier: {}'.format(classifier))
		(labels, features, tags, orig_leakage, feature_importance) = cls.process_all(interactions=test_traces, calcSpace=True, calcTime=True, quant_mode=classifier, window_size=None, 
		feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=False)

		options_leakage.append(('No-mitigation', orig_leakage, 0.0, 0.0))
		target_tags = [x[1] for x in feature_importance]

		if calculate_bounds:
			accuracy_bound = cls.calculate_bayes_error(labels, features, tags, target_tags)
			print('Accuracy Bound for Option {}: {:.2f}'.format('No-mitigation', accuracy_bound))

		print('Leakage for Option {}: {:.2f}'.format('No-mitigation', orig_leakage))
		print('Overhead for Option {}: {:.2f}'.format('No-mitigation', 0.0))
		print('Time Overhead for Option {}: {:.2f}'.format('No-mitigation', 0.0))
		print('RANDOM GUESS ACCURACY for {} classes: {:.2f}'.format(len(labels_list), 1.0/len(labels_list)))
		print('='*40)
		print('%'*40)

		#Finding max value for the features targeting specific packet sizes.
		max_padding = 0
		for tag in target_tags:
			if 'Number of packets with size' in tag:
				pkt_size = int(tag.split()[5])
				if pkt_size > max_padding:
					max_padding = pkt_size

		#RELATED WORK TESTS
		start_time = time.time()
		mcr_option = ['' for _ in weights_list]
		mcr_leakage = [10000.0 for _ in weights_list] 
		mcr_overhead = [10000.0 for _ in weights_list]
		mcr_t_overhead = [10000.0 for _ in weights_list]
		mcr_option_both = ['' for _ in weights_list]
		mcr_leakage_both = [10000.0 for _ in weights_list] 
		mcr_overhead_both = [10000.0 for _ in weights_list]
		mcr_t_overhead_both = [10000.0 for _ in weights_list]
		mcr_option_time = ['' for _ in weights_list]
		mcr_leakage_time = [10000.0 for _ in weights_list] 
		mcr_overhead_time = [10000.0 for _ in weights_list]
		mcr_t_overhead_time = [10000.0 for _ in weights_list] 
		mcr_option_inj = ['' for _ in weights_list]
		mcr_leakage_inj = [10000.0 for _ in weights_list] 
		mcr_overhead_inj = [10000.0 for _ in weights_list]
		mcr_t_overhead_inj = [10000.0 for _ in weights_list]
		mcr_option_base = ['' for _ in weights_list]
		mcr_leakage_base = [10000.0 for _ in weights_list] 
		mcr_overhead_base = [10000.0 for _ in weights_list]
		mcr_t_overhead_base = [10000.0 for _ in weights_list]

		#options_related = ['heartbeat', 'mtu-20ms']
		options_related = ['exp', 'linear', 'uniform', 'uniform255', 'mice-elephants', 'mtu', 'mtu-20ms', 'p100', 'p500', 'p700', 'p900']# 'similarity', 

		if run_related:
			#Size/Time Calculation for Test Set
			total_size_orig_train = 0
			total_time_orig_train = 0.0
			total_size_orig_test  = 0
			total_time_orig_test  = 0.0
			for t in train_traces:
				total_time_orig_train += abs(t[-1].time - t[0].time)
				for p in t:
					total_size_orig_train += p.size
			for t in test_traces:
				total_time_orig_test += abs(t[-1].time - t[0].time)
				for p in t:
					total_size_orig_test += p.size
			
			if total_time_orig_train < 0.0001:
				total_time_orig_train = 1.0
			if total_time_orig_test < 0.0001:
				total_time_orig_test = 1.0

			for train_mode in [False]:
				for option in options_related:	
					for pkt_injection in pkt_injection_list:
						for time_injection in time_injection_list:
							t_overhead = 0.0
							overhead = 0.0
							leakage = 0.0
							abs_overhead = 0.0
							accuracy_bound = 0.0
							
							for rep_counter in range(rep_count):
								if train_mode:
									print('Using option: {} on Training Set, PktInjection = {}, TimeInjection = {}, run #{}'.format(option, pkt_injection, time_injection, rep_counter))
								else:
									print('Using option: {} on Test Set, PktInjection = {}, TimeInjection = {}, run #{}'.format(option, pkt_injection, time_injection, rep_counter))
								#Inject noise into traces and get new set of traces
								if train_mode:
									new_traces1 = copy.deepcopy(train_traces)
								else:
									new_traces1 = copy.deepcopy(test_traces)
								if 'targeted' in option:
									new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option=option, pkt_injection=pkt_injection, option_val=max_padding, time_injection=time_injection, train_traces=train_traces)
									(new_labels, new_features, new_tags, leakage_sample, new_feature_importance) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
										feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)
								else:
									new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option=option, pkt_injection=pkt_injection, time_injection=time_injection, train_traces=train_traces)
									(new_labels, new_features, new_tags, leakage_sample, new_feature_importance) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
										feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)

								total_size_1 = 0
								total_time_1 = 0.0
								for t in new_traces1:
									total_time_1 += abs(t[-1].time - t[0].time)
									for p in t:
										total_size_1 += p.size
								
								if train_mode:
									overhead_sample = float(total_size_1-total_size_orig_train)/float(total_size_orig_train)
									t_overhead_sample = float(total_time_1-total_time_orig_train)/float(total_time_orig_train)
									abs_overhead += float(total_size_1-total_size_orig_train)/(len(new_traces1)*float(rep_count))
								else:
									overhead_sample = float(total_size_1-total_size_orig_test)/float(total_size_orig_test)
									t_overhead_sample = float(total_time_1-total_time_orig_test)/float(total_time_orig_test)
									abs_overhead += float(total_size_1-total_size_orig_test)/(len(new_traces1)*float(rep_count))

								if calculate_bounds:
									new_target_tags = [x[1] for x in new_feature_importance[:int(len(new_feature_importance)/4)]]
									accuracy_bound_sample = cls.calculate_bayes_error(new_labels, new_features, new_tags, new_target_tags)					
									accuracy_bound += accuracy_bound_sample/float(rep_count)

								leakage    += leakage_sample/float(rep_count)
								overhead   += overhead_sample/float(rep_count)
								t_overhead += t_overhead_sample/float(rep_count)

							print_option = '{}-{}-{}'.format(option, pkt_injection, time_injection)

							if calculate_bounds:
								print('Accuracy Bound for Option {}: {:.2f}'.format(print_option, accuracy_bound))
							print('Leakage for Option {}: {:.2f}'.format(print_option, leakage))
							print('Overhead for Option {}: {:.2f}'.format(print_option, overhead))
							print('Time Overhead for Option {}: {:.2f}'.format(print_option, t_overhead))
							if train_mode:
								print('Absolute Byte Overhead vs. Original Trace Size per Trace for Option {}: {:.2f}/{:.2f} bytes'.format(print_option, abs_overhead, float(total_size_orig_train)/len(new_traces1)))
							else:
								print('Absolute Byte Overhead vs. Original Trace Size per Trace for Option {}: {:.2f}/{:.2f} bytes'.format(print_option, abs_overhead, float(total_size_orig_test)/len(new_traces1)))

							if train_mode:
								options_leakage.append((print_option+'-TRAIN', leakage, overhead, t_overhead))
							else:
								options_leakage.append((print_option+'-TEST', leakage, overhead, t_overhead))

							if '20ms' in option:
								continue
							if train_mode:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if (l_weight*leakage + oh_weight*overhead) < (l_weight*mcr_leakage[ind]+oh_weight*mcr_overhead[ind]):
										mcr_option[ind] = print_option
										mcr_leakage[ind] = leakage
										mcr_overhead[ind] = overhead
										mcr_t_overhead[ind] = t_overhead
							if not train_mode:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if mcr_option[ind] == print_option:
										print('OVERALL-Best option for {}*L+{}*OH: {}, {}'.format(l_weight, oh_weight, mcr_option[ind], print_option))
										mcr_leakage[ind] = leakage
										mcr_overhead[ind] = overhead
										mcr_t_overhead[ind] = t_overhead

							#TIME ON EXAMPLES
							if train_mode and time_injection and not pkt_injection:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if (l_weight*leakage + oh_weight*overhead) < (l_weight*mcr_leakage_time[ind]+oh_weight*mcr_overhead_time[ind]):
										mcr_option_time[ind] = print_option
										mcr_leakage_time[ind] = leakage
										mcr_overhead_time[ind] = overhead
										mcr_t_overhead_time[ind] = t_overhead
							if not train_mode and time_injection and not pkt_injection:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if mcr_option_time[ind] == print_option:
										print('TIME-Best option for {}*L+{}*OH: {}, {}'.format(l_weight, oh_weight, mcr_option_time[ind], print_option))
										mcr_leakage_time[ind] = leakage
										mcr_overhead_time[ind] = overhead
										mcr_t_overhead_time[ind] = t_overhead
							
							#PKT_INJ_ON EXAMPLES
							if train_mode and pkt_injection and not time_injection: 
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if (l_weight*leakage + oh_weight*overhead) < (l_weight*mcr_leakage_inj[ind]+oh_weight*mcr_overhead_inj[ind]):
										mcr_option_inj[ind] = print_option
										mcr_leakage_inj[ind] = leakage
										mcr_overhead_inj[ind] = overhead
										mcr_t_overhead_inj[ind] = t_overhead
							if not train_mode and pkt_injection and not time_injection:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if mcr_option_inj[ind] == print_option:
										print('PKTINJECTION-Best option for {}*L+{}*OH: {}, {}'.format(l_weight, oh_weight, mcr_option_inj[ind], print_option))
										mcr_leakage_inj[ind] = leakage
										mcr_overhead_inj[ind] = overhead
										mcr_t_overhead_inj[ind] = t_overhead

							if train_mode and time_injection and pkt_injection:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if (l_weight*leakage + oh_weight*overhead) < (l_weight*mcr_leakage_both[ind]+oh_weight*mcr_overhead_both[ind]):
										mcr_option_both[ind] = print_option
										mcr_leakage_both[ind] = leakage
										mcr_overhead_both[ind] = overhead
										mcr_t_overhead_both[ind] = t_overhead
							if not train_mode and time_injection and pkt_injection:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if mcr_option_both[ind] == print_option:
										print('TIME+PKTINJ-Best option for {}*L+{}*OH: {}, {}'.format(l_weight, oh_weight, mcr_option_both[ind], print_option))
										mcr_leakage_both[ind] = leakage
										mcr_overhead_both[ind] = overhead
										mcr_t_overhead_both[ind] = t_overhead

							if train_mode and not time_injection and not pkt_injection:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if (l_weight*leakage + oh_weight*overhead) < (l_weight*mcr_leakage_base[ind]+oh_weight*mcr_overhead_base[ind]):
										mcr_option_base[ind] = print_option
										mcr_leakage_base[ind] = leakage
										mcr_overhead_base[ind] = overhead
										mcr_t_overhead_base[ind] = t_overhead
							if not train_mode and not time_injection and not pkt_injection:
								for ind, (l_weight, oh_weight) in enumerate(weights_list):
									if mcr_option_base[ind] == print_option:
										print('BASELINE-Best option for {}*L+{}*OH: {}, {}'.format(l_weight, oh_weight, mcr_option_base[ind], print_option))
										mcr_leakage_base[ind] = leakage
										mcr_overhead_base[ind] = overhead
										mcr_t_overhead_base[ind] = t_overhead

					print('='*40)
					print('%'*40)

		
		print('=========')
		print('TARGETED METHODS:')

		#GRID SEARCH
		mid_time = time.time()
		mcc_option = ['' for _ in weights_list]
		mcc_leakage = [10000.0 for _ in weights_list] 
		mcc_overhead = [10000.0 for _ in weights_list]
		mcc_t_overhead = [10000.0 for _ in weights_list]

		if run_gs:
			#Size/Time Calculation for Train Set
			total_size_orig = 0
			total_time_orig = 0.0
			total_size_1 = 0
			total_time_1 = 0.0
			for t in traces:
				total_time_orig += abs(t[-1].time - t[0].time)
				for p in t:
					total_size_orig += p.size
			if total_time_orig < 0.0001:
				total_time_orig = 1.0

			for variance in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 250, 500]:
				for pkt_injection in [True, False]:
					print('Testing Gaussian with variance = {}, pkt_injection = {}, time_injection={}'.format(variance, pkt_injection, time_injection))
					t_overhead = 0.0
					overhead = 0.0
					leakage = 0.0
					option = (variance, pkt_injection)
					abs_overhead = 0.0
					accuracy_bound = 0.0

					for rep_counter in range(rep_count):
						print('Run #{}'.format(rep_counter))
						new_traces1 = copy.deepcopy(traces)
						#Run inject noise, get new traces.
						new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option='gaussian', label_distinguishing=False, option_val=variance, pkt_injection=pkt_injection, time_injection=time_injection)
						
						(new_labels, new_features, new_tags, leakage_sample, new_feature_importance) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
						feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)

						if calculate_bounds:
							new_target_tags = [x[1] for x in new_feature_importance[:int(len(new_feature_importance)/4)]]
							accuracy_bound_sample = cls.calculate_bayes_error(new_labels, new_features, new_tags, new_target_tags)
							accuracy_bound += accuracy_bound_sample/float(rep_count)

						#Calculating the overhead
						total_size_1 = 0
						total_time_1 = 0.0
						for t in new_traces1:
							total_time_1 += abs(t[-1].time - t[0].time)
							for p in t:
								total_size_1 += p.size
						if total_time_1 < 0.0001:
							total_time_1 = 1.0
						
						overhead_sample = float(total_size_1-total_size_orig)/float(total_size_orig)
						t_overhead_sample = float(total_time_1-total_time_orig)/float(total_time_orig)
						
						abs_overhead += float(total_size_1-total_size_orig)/(len(new_traces1)*float(rep_count))
						leakage    += leakage_sample/float(rep_count)
						overhead   += overhead_sample/float(rep_count)
						t_overhead += t_overhead_sample/float(rep_count)
						
					
					#leakage = leakage
					#overhead = overhead
					#t_overhead = t_overhead
					if calculate_bounds:
						print('Accuracy Bound for Gaussian padding, variance: {:.2f}, pkt_injection: {} - {:.2f}'.format(variance, pkt_injection, accuracy_bound))
					print('Leakage for Gaussian padding, variance: {:.2f}, pkt_injection: {} - {:.2f}'.format(variance, pkt_injection, leakage))
					print('Overhead for Gaussian padding, variance: {:.2f}, pkt_injection: {} - {:.2f}'.format(variance, pkt_injection, overhead))
					print('Time Overhead for Gaussian padding, variance: {:.2f}, pkt_injection: {} - {:.2f}'.format(variance, pkt_injection, t_overhead))
					print('Absolute Byte Overhead vs. Original Trace Size per Trace for Gaussian padding, for option {}: {:.2f}/{:.2f} bytes'.format(option, abs_overhead, float(total_size_orig)/len(new_traces1)))

					for ind, (l_weight, oh_weight) in enumerate(weights_list):
						if (l_weight*leakage + oh_weight*overhead) < (l_weight*mcc_leakage[ind] + oh_weight*mcc_overhead[ind]):
							mcc_option[ind] = option
							mcc_leakage[ind] = leakage
							mcc_overhead[ind] = overhead
							mcc_t_overhead[ind] = t_overhead

					print('='*40)
					print('%'*40)

			#TESTING THE BEST SELECTED ONES
			#Size/Time Calculation for Test Set
			total_size_orig = 0
			total_time_orig = 0.0
			total_size_1 = 0
			total_time_1 = 0.0
			for t in test_traces:
				total_time_orig += abs(t[-1].time - t[0].time)
				for p in t:
					total_size_orig += p.size
			if total_time_orig < 0.0001:
				total_time_orig = 1.0
		
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				var, pkt_inject = mcc_option[ind]
				new_leakage, new_overhead, new_t_overhead = 0.0, 0.0, 0.0

				for _ in range(rep_count):
					new_traces1 = copy.deepcopy(test_traces)
					#Run inject noise, get new traces.
					new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option='gaussian', label_distinguishing=False, option_val=var, pkt_injection=pkt_inject, time_injection=time_injection)
					
					#print('KDE:')
					##Run second leg of quantification, get the reduced leakage.
					#(new_labels, new_features, new_tags, new_leakage) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='kde', window_size=None, 
					#feature_reduction=None, num_reduced_features=5, alignment=False, new_direction=False)
					print('Random Forest:')
					#Run second leg of quantification, get the reduced leakage.
					(new_labels, new_features, new_tags, leakage_sample, _) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
					feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)

					total_size_1 = 0
					total_time_1 = 0.0
					for t in new_traces1:
						total_time_1 += abs(t[-1].time - t[0].time)
						for p in t:
							total_size_1 += p.size
					if total_time_1 < 0.0001:
						total_time_1 = 1.0
					
					overhead_sample = float(total_size_1-total_size_orig)/float(total_size_orig)
					t_overhead_sample = float(total_time_1-total_time_orig)/float(total_time_orig)
					
					new_leakage    += leakage_sample/float(rep_count)
					new_overhead   += overhead_sample/float(rep_count)
					new_t_overhead += t_overhead_sample/float(rep_count)
				#mcc_option[ind] = option
				print('Grid Search option/parameter with lowest {}*L+{}*OH TRAINING: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcc_option[ind], mcc_leakage[ind], mcc_overhead[ind], mcc_t_overhead[ind], (oh_weight*mcc_overhead[ind]+l_weight*mcc_leakage[ind])))
				mcc_leakage[ind] = new_leakage
				mcc_overhead[ind] = new_overhead
				mcc_t_overhead[ind] = new_t_overhead
				print('Grid Search option/parameter with lowest {}*L+{}*OH TESTING: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcc_option[ind], mcc_leakage[ind], mcc_overhead[ind], mcc_t_overhead[ind], (oh_weight*mcc_overhead[ind]+l_weight*mcc_leakage[ind])))

		#SIMULATED ANNEALING SEARCH
		mid2_time = time.time()
		if run_sa:
			for (alpha, beta) in [(1.0, 1.0), (0.1, 1.0), (1.0, 0.1)]:
				print('TESTING WITH Alpha={}, Beta={}'.format(alpha, beta))
				train_mode = True
				#alpha, beta = 1.0, 0.1 #Leakage/Overhead weights for SA run

				total_size_orig_train = 0
				total_time_orig_train = 0.0
				total_size_orig_test  = 0
				total_time_orig_test  = 0.0
				for t in train_traces:
					total_time_orig_train += abs(t[-1].time - t[0].time)
					for p in t:
						total_size_orig_train += p.size
				for t in test_traces:
					total_time_orig_test += abs(t[-1].time - t[0].time)
					for p in t:
						total_size_orig_test += p.size
				
				if total_time_orig_train < 0.0001:
					total_time_orig_train = 1.0
				if total_time_orig_test < 0.0001:
					total_time_orig_test = 1.0

				pkt_injection = False
				best_sa_option = ['' for _ in weights_list]
				best_sa_leakage = [10000.0 for _ in weights_list] 
				best_sa_overhead = [10000.0 for _ in weights_list]
				best_sa_t_overhead = [10000.0 for _ in weights_list]
				best_sa_eval = [10000.0 for _ in weights_list]

				curr_eval = 10000.0
				print('Starting Simulated Annealing Approach')
				
				#Simulated Annealing approach
				pkt_injection_limit = random.randint(1,10)
				time_injection_limit = random.uniform(0.0, 0.250)
				option = random.sample(options_related, k=1)[0]
				
				print('Iteration -1, option: {}, pkt_injection_param: {}, time_delay_param: {}'.format(option, pkt_injection_limit, time_injection_limit))

				init_temperature = 10.0
				
				t_overhead = 0.0
				overhead = 0.0
				leakage = 0.0
				for rep_counter in range(rep_count):
					new_traces1 = copy.deepcopy(traces)
					#Run inject noise, get new traces.
					#if train_mode:
					#	print('Using option: {} on Training Set, PktInjection = {}, TimeInjection = {}, run #{}'.format(option, pkt_injection_, time_injection, rep_counter))
					#else:
					#	print('Using option: {} on Test Set, PktInjection = {}, TimeInjection = {}, run #{}'.format(option, pkt_injection, time_injection, rep_counter))
					#Inject noise into traces and get new set of traces
					if train_mode:
						new_traces1 = copy.deepcopy(train_traces)
					else:
						new_traces1 = copy.deepcopy(test_traces)
					if 'targeted' in option:
						new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option=option, pkt_injection=pkt_injection_limit, option_val=max_padding, time_injection=time_injection_limit, train_traces=train_traces)
						(new_labels, new_features, new_tags, leakage_sample, new_feature_importance) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
							feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)
					else:
						new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option=option, pkt_injection=pkt_injection_limit, time_injection=time_injection_limit, train_traces=train_traces)
						(new_labels, new_features, new_tags, leakage_sample, new_feature_importance) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
							feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)
				
					total_size_1 = 0
					total_time_1 = 0.0
					for t in new_traces1:
						total_time_1 += abs(t[-1].time - t[0].time)
						for p in t:
							total_size_1 += p.size

					if train_mode:
						overhead_sample = float(total_size_1-total_size_orig_train)/float(total_size_orig_train)
						t_overhead_sample = float(total_time_1-total_time_orig_train)/float(total_time_orig_train)
						#abs_overhead += float(total_size_1-total_size_orig_train)/(len(new_traces1)*float(rep_count))
					else:
						overhead_sample = float(total_size_1-total_size_orig_test)/float(total_size_orig_test)
						t_overhead_sample = float(total_time_1-total_time_orig_test)/float(total_time_orig_test)
						#abs_overhead += float(total_size_1-total_size_orig_test)/(len(new_traces1)*float(rep_count))

					leakage    += leakage_sample/float(rep_count)
					overhead   += overhead_sample/float(rep_count)
					t_overhead += t_overhead_sample/float(rep_count)
					
				if (alpha*leakage + beta*overhead) < curr_eval:
					curr_eval = (alpha*leakage + beta*overhead)

				for ind, (l_weight, oh_weight) in enumerate(weights_list):
					if (l_weight*leakage + oh_weight*overhead) < (l_weight*best_sa_leakage[ind]+oh_weight*best_sa_overhead[ind]):
						best_sa_option[ind]     = (option, time_injection_limit, pkt_injection_limit)
						best_sa_leakage[ind]    = leakage
						best_sa_overhead[ind]   = overhead
						best_sa_t_overhead[ind] = t_overhead
						best_sa_eval[ind]       = l_weight*leakage + oh_weight*overhead
				
				#Repeat for N steps
				for iterx in range(50):
					#Fast temperature update
					temp = init_temperature-float(iterx + 1)*float(init_temperature)/50

					#Generate new candidate
					#cand = -1.0
					#cand = curr + temp * random.uniform(-1,1)
					lower_pkt_limit = max(int(pkt_injection_limit-temp), 0)
					upper_pkt_limit = min(int(pkt_injection_limit+temp), 10)
					new_pkt_injection_limit = random.randint(lower_pkt_limit, upper_pkt_limit)
					lower_time_limit = max((time_injection_limit - 0.05*temp), 0.0)
					upper_time_limit = min((time_injection_limit + 0.05*temp), 250.0)
					new_time_injection_limit = random.uniform(lower_time_limit, upper_time_limit)
					new_option = random.sample(options_related, k=1)[0]

					print('Iteration {}, option: {}, pkt_injection_param: {}, time_delay_param: {}'.format(iterx, new_option, new_pkt_injection_limit, new_time_injection_limit))

					cand_eval = 10000.0

					t_overhead = 0.0
					overhead = 0.0
					leakage = 0.0
					for _ in range(rep_count):
						if train_mode:
							new_traces1 = copy.deepcopy(train_traces)
						else:
							new_traces1 = copy.deepcopy(test_traces)
						if 'targeted' in option:
							new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option=new_option, pkt_injection=new_pkt_injection_limit, option_val=max_padding, time_injection=new_time_injection_limit, train_traces=train_traces)
							(new_labels, new_features, new_tags, leakage_sample, new_feature_importance) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
								feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)
						else:
							new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option=new_option, pkt_injection=new_pkt_injection_limit, time_injection=new_time_injection_limit, train_traces=train_traces)
							(new_labels, new_features, new_tags, leakage_sample, new_feature_importance) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
								feature_reduction=feature_reduction, num_reduced_features=10, alignment=False, new_direction=False, silent=True)

						if train_mode:
							overhead_sample = float(total_size_1-total_size_orig_train)/float(total_size_orig_train)
							t_overhead_sample = float(total_time_1-total_time_orig_train)/float(total_time_orig_train)
							#abs_overhead += float(total_size_1-total_size_orig_train)/(len(new_traces1)*float(rep_count))
						else:
							overhead_sample = float(total_size_1-total_size_orig_test)/float(total_size_orig_test)
							t_overhead_sample = float(total_time_1-total_time_orig_test)/float(total_time_orig_test)
							#abs_overhead += float(total_size_1-total_size_orig_test)/(len(new_traces1)*float(rep_count))

						leakage    += leakage_sample/float(rep_count)
						overhead   += overhead_sample/float(rep_count)
						t_overhead += t_overhead_sample/float(rep_count)
						
					#print('Overhead for Gaussian padding, variance:{:.2f}, {} - '.format(variance, pkt_injection), overhead)
					#print('Time Overhead for Gaussian padding, variance:{:.2f}, {} - '.format(variance, pkt_injection), t_overhead)

					#if (alpha*leakage + beta*overhead) < cand_eval:
					cand_eval = (alpha*leakage + beta*overhead)

					for ind, (l_weight, oh_weight) in enumerate(weights_list):
						if (l_weight*leakage + oh_weight*overhead) < best_sa_eval[ind]:
							best_sa_option[ind]     = (new_option, new_time_injection_limit, new_pkt_injection_limit)
							best_sa_leakage[ind]    = leakage
							best_sa_overhead[ind]   = overhead
							best_sa_t_overhead[ind] = t_overhead
							best_sa_eval[ind]       = (l_weight*leakage + oh_weight*overhead)
							# best, best_eval, best_leak, best_overhead = cand, cand_eval, leakage, overhead
						#print('>{}, Var:{:.2f}, Leak/Overhead/Obj:{:.2f},{:.2f},{:.2f}'.format(iterx, best, best_leak, best_overhead, best_eval))

					diff = cand_eval - curr_eval

					metropolis = math.exp(-diff/temp)
					print('Objective difference (Diff = CandObj - CurrObj): {} = {} - {}'.format(diff, cand_eval, curr_eval))
					print('Temperature: {}'.format(temp))
					print('Acceptance probability: {:.4f}'.format(metropolis))

					if diff < 0 or random.uniform(0,2) < metropolis:
						#Move the current point if the solution is more efficient or with a random probability
						curr_eval = cand_eval
						option = new_option
						time_injection_limit = new_time_injection_limit
				
						pkt_injection_limit = new_pkt_injection_limit
				
				print('For Alpha: {} and Beta: {} RESULTS')
				for ind, (l_weight, oh_weight) in enumerate(weights_list):
					print('Simulated Annealing option/parameter with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, best_sa_option[ind], best_sa_leakage[ind], best_sa_overhead[ind], best_sa_t_overhead[ind], (oh_weight*best_sa_overhead[ind]+l_weight*best_sa_leakage[ind])))

			

		end_time = time.time()

		print('='*50)

		print('*'*50)
		print('Analyzed {}, number of traces: {}'.format(filename, len(traces)))
		print('Option - Leakage - Overhead - Timing Overhead - Objective Function Result')
		if run_related:
			#print('The related work option with lowest overhead  : {} - {:.2f} - {:.2f} - {:.2f}'.format(mor_option, mor_leakage, mor_overhead, mor_t_overhead, mor_overhead))
			#print('The related work option with lowest leakage   : {} - {:.2f} - {:.2f} - {:.2f}'.format(mlr_option, mlr_leakage, mlr_overhead, mlr_t_overhead, mlr_leakage))
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				print('OVERALL-The related work option with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcr_option[ind], mcr_leakage[ind], mcr_overhead[ind], mcr_t_overhead[ind], (oh_weight*mcr_overhead[ind]+l_weight*mcr_leakage[ind])))
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				print('TIME-The related work option with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcr_option_time[ind], mcr_leakage_time[ind], mcr_overhead_time[ind], mcr_t_overhead_time[ind], (oh_weight*mcr_overhead_time[ind]+l_weight*mcr_leakage_time[ind])))
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				print('PKTINJECTION-The related work option with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcr_option_inj[ind], mcr_leakage_inj[ind], mcr_overhead_inj[ind], mcr_t_overhead_inj[ind], (oh_weight*mcr_overhead_inj[ind]+l_weight*mcr_leakage_inj[ind])))
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				print('TIME+PKTINJ-The related work option with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcr_option_both[ind], mcr_leakage_both[ind], mcr_overhead_both[ind], mcr_t_overhead_both[ind], (oh_weight*mcr_overhead_both[ind]+l_weight*mcr_leakage_both[ind])))			
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				print('BASELINE-The related work option with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcr_option_base[ind], mcr_leakage_base[ind], mcr_overhead_base[ind], mcr_t_overhead_base[ind], (oh_weight*mcr_overhead_base[ind]+l_weight*mcr_leakage_base[ind])))

			print('\nALL RELATED WORK LEAKAGE\n')
			for option,leakage,overhead,t_overhead in options_leakage:
				print('Related Work: {} - {:.2f} - {:.2f} - {:.2f}'.format(option, leakage, overhead, t_overhead))
		
		if run_gs:
			#print('Grid Search option/parameter with lowest overhead  : {} - {:.2f} - {:.2f} - {:.2f}'.format(moc_option, moc_leakage, moc_overhead, moc_t_overhead, moc_overhead))
			#print('Grid Search option/parameter with lowest leakage   : {} - {:.2f} - {:.2f} - {:.2f}'.format(mlc_option, mlc_leakage, mlc_overhead, mlc_t_overhead, mlc_leakage))
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				print('Grid Search option/parameter with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, mcc_option[ind], mcc_leakage[ind], mcc_overhead[ind], mcc_t_overhead[ind], (oh_weight*mcc_overhead[ind]+l_weight*mcc_leakage[ind])))

		if run_sa:
			for ind, (l_weight, oh_weight) in enumerate(weights_list):
				print('Simulated Annealing option/parameter with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(l_weight, oh_weight, best_sa_option[ind], best_sa_leakage[ind], best_sa_overhead[ind], best_sa_t_overhead[ind], (oh_weight*best_sa_overhead[ind]+l_weight*best_sa_leakage[ind])))
		#print('SA option/parameter with lowest {}*L+{}*OH: {} - {:.2f} - {:.2f} - {:.2f}'.format(best, best_leak, best_overhead, best_eval))

		print('Preprocessing runtime: {:.3f} seconds'.format(start_time - pre_time))
		if run_related:
			print('Related Work runtime : {:.3f} seconds'.format(mid_time   - start_time))
		if run_gs:
			print('Grid Search runtime  : {:.3f} seconds'.format(mid2_time  - mid_time))
		if run_sa:
			print('SA runtime           : {:.3f} seconds'.format(end_time   - mid2_time))
		print('Total runtime        : {:.3f} seconds'.format(end_time   - pre_time))

		print('*'*50)
		print('='*50)

		#for padding in [100, 500, 1000]:
		#	print('Testing Fixed padding with padding = {}'.format(padding))
		#	new_traces1 = copy.deepcopy(traces)
		#	#Run inject noise, get new traces.
		#	new_traces1 = cls.inject_noise(new_traces1, target_tags, tags, features, labels, option='fixed', label_distinguishing=False, option_val=padding)
		#	
		#	print('KDE:')
		#	#Run second leg of quantification, get the reduced leakage.
		#	(new_labels, new_features, new_tags, new_leakage) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='kde', window_size=None, 
		#	feature_reduction=None, num_reduced_features=5, alignment=False, new_direction=False)
		#	print('Random Forest:')
		#	#Run second leg of quantification, get the reduced leakage.
		#	(new_labels, new_features, new_tags, new_leakage) = cls.process_all(interactions=new_traces1, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
		#	feature_reduction=None, num_reduced_features=5, alignment=False, new_direction=False)
		#
		#	total_size_1 = 0
		#	for t in new_traces1:
		#		for p in t:
		#			total_size_1 += p.size
		#	print('Overhead for Fixed padding, value:{} - '.format(padding), float(total_size_1-total_size_orig)/float(total_size_orig))
		#	
		#	print('='*40)
		#	print('%'*40)
		
		
		#new_traces2 = copy.deepcopy(traces)
		##Run inject noise with class merging class 0 to 1, 2 to 3
		#new_traces2 = cls.inject_noise(new_traces2, target_tags, tags, features, labels, label_distinguishing=True)
		#
		#print('KDE:')
		##Run second leg of quantification, get the reduced leakage.
		#(new_labels, new_features, new_tags, new_leakage) = cls.process_all(interactions=new_traces2, calcSpace=True, calcTime=True, quant_mode='kde', window_size=None, 
		#feature_reduction=None, num_reduced_features=5, alignment=False, new_direction=False)
		#print('Random Forest:')
		##Run second leg of quantification, get the reduced leakage.
		#(new_labels, new_features, new_tags, new_leakage) = cls.process_all(interactions=new_traces2, calcSpace=True, calcTime=True, quant_mode='rf-classifier', window_size=None, 
		#feature_reduction=None, num_reduced_features=5, alignment=False, new_direction=False)
		#
		#total_size_2 = 0
		#for t in new_traces2:
		#	for p in t:
		#		total_size_2 += p.size
		#print('Overhead for 2nd method:', float(total_size_2-total_size_orig)/float(total_size_orig))

		#TODO: Summarize all the results, with a Pareto curve
		#Implement the simulated annealing
		#Add more parameters/feature targeting
		#How to guide the search, maybe do kind of like binary search, 
		#	increase the variance/feature targeting when the leakage is high, reduce it when the overhead is high. 

		return None

	@classmethod
	def run_fbleau(cls, features, labels, tags, quant_option='log', filename='trial'):
		""""""
		trainfn = 'train-{}.csv'.format(filename)
		testfn  = 'test-{}.csv'.format(filename)

		transposed_f = map(list, zip(*features))
		#print(len(labels))
		#print(len(features))
		label_feature_pairs = list(zip(labels, transposed_f))

		#Train/test is divided 80-20%
		label_feature_pairs.sort(key=lambda el: el[0])
		train_file = open(trainfn, 'w')
		test_file  = open(testfn, 'w')

		for i, (l, f_list) in enumerate(label_feature_pairs):
			line = '{}, {}'.format(l, ', '.join([str(x) for x in f_list]))
			#print(line)
			#print('')
			#Compose string
			if i%5 == 0:
				test_file.write(line + '\n')
				#Write to test
			else:
				train_file.write(line + '\n')
				#Write to train
		train_file.close()
		test_file.close()

		#cmd_string = ['fbleau', quant_option, trainfn, testfn] #, '|', 'tee', 'fbleau_leak_results_{}.txt'.format(filename)]
		cmd_string = 'fbleau {} {} {} | tee {}'.format(quant_option, trainfn, testfn, 'fbleau_leak_results_{}.txt'.format(filename))
		#print(cmd_string)
		#subprocess.call(['ls', '-l'])
		subprocess.call(cmd_string, shell=True)

		return None