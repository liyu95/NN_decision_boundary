#!/usr/bin/env python
from sklearn.datasets import make_classification, make_blobs
from sklearn.datasets import make_circles, make_moons
import sklearn
import numpy 
import pylab
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
from itertools import cycle
import numpy as np


class batch_object(object):
        """docstring for batch_object"""
        def __init__(self, data_list, batch_size):
                super(batch_object, self).__init__()
                self.data_list = data_list
                self.batch_size = batch_size
                self.pool = cycle(data_list)
                
        def next_batch(self):
                data_batch = list()
                for i in xrange(self.batch_size):
                        data_batch.append(next(self.pool))
                data_batch = np.array(data_batch)
                return data_batch

def to_categorical(label_array,total_classes):
    from sklearn.preprocessing import OneHotEncoder
    enc=OneHotEncoder()
    label_list=[]
    for i in range(len(label_array)):
        label_list.append([label_array[i]])
    return enc.fit_transform(label_list).toarray()

def get_svm(feature, label):
	clf = SVC(kernel='linear')
	clf.fit(feature, label)
	return clf


# This binary classification should be linear separable
def play_iris():
	data = load_iris()
	feature = data.data
	label = data.target
	label[label==2]=1
	# clf = LinearSVC(random_state=0)
	clf = SVC(kernel='linear')
	clf.fit(feature, label)
	scores = cross_val_score(clf, feature, label, cv=5)


def generate_lin_sep_blobs(n_samples, random_state):
	samples = make_blobs(n_samples=n_samples, n_features=2,
		centers=[(0.2, 0.2), (0.8, 0.8)], cluster_std=0.1,
		random_state=random_state)
	return samples[0], samples[1]

def draw_from_moons(n_samples, random_state):
	samples = make_moons(n_samples=n_samples, 
		random_state=random_state)
	return samples[0], samples[1]

def draw_from_sectors(size, inner, outer, random_state):
	np.random.seed(random_state)
	x1 = np.random.rand(size)*outer
	x2 = np.random.rand(size)*outer
	feature =  np.vstack([x1,x2]).transpose()
	z = np.linalg.norm(feature, ord=2, axis=1)
	feature = feature[(z<=outer) & (z>=inner)]
	return feature

def generate_difficult_sectors(n_samples, random_state):
	feature0 = draw_from_sectors(n_samples, 0.1, 0.2, random_state)
	feature1 = draw_from_sectors(n_samples*3, 0.283, 0.4, random_state)
	label0 = np.zeros(len(feature0))
	label1 = np.ones(len(feature1))
	feature = np.concatenate((feature0, feature1), axis=0)
	label = np.concatenate((label0, label1), axis=0)
	return feature, label

def plot_difficult_sectors(feature, label, name=None):
	pylab.figure()
	red = feature[label == 0]
	blue = feature[label == 1]
	pylab.plot(red[:, 0], red[:, 1], 'r.')
	pylab.plot(blue[:, 0], blue[:, 1], 'b.')
	pylab.plot([0,0.283],[0.283,0])
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)

def plot_blobs(feature, label, name=None):
	pylab.figure()
	red = feature[label == 0]
	blue = feature[label == 1]
	pylab.plot(red[:, 0], red[:, 1], 'r.')
	pylab.plot(blue[:, 0], blue[:, 1], 'b.')
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)

def lin_sep_with_ground_truth(size, margin, random_state):
	np.random.seed(random_state)
	x1 = np.random.rand(size)
	x2 = np.random.rand(size)
	feature =  np.vstack([x1,x2]).transpose()
	z = np.sum(feature, 1)
	feature = feature[(z<=1.0-margin) | (z>=1.0+margin)]
	z = z[(z<=1.0-margin) | (z>=1.0+margin)]
	ind_1 = np.where(z>=1.0+margin)[0]
	ind_2 = np.where(z<=1.0-margin)[0]
	label = np.zeros(len(feature))
	label[ind_1] = 0
	label[ind_2] = 1
	return feature, label


def non_lin_sep(size, margin, random_state):
	np.random.seed(random_state)
	x1 = np.random.rand(size)
	x2 = np.random.rand(size)
	feature =  np.vstack([x1,x2]).transpose()
	z = np.linalg.norm(feature-[0.5,0.5], ord=2, axis=1)
	feature = feature[(z<=0.2-margin) | (z>=0.2+margin)]
	z = z[(z<=0.2-margin) | (z>=0.2+margin)]
	ind_1 = np.where(z>=0.2+margin)[0]
	ind_2 = np.where(z<=0.2-margin)[0]
	label = np.zeros(len(feature))
	label[ind_1] = 0
	label[ind_2] = 1
	return feature, label



def visual_separable_binary(feature,label,margin, name=None):
	pylab.figure()
	pylab.scatter(feature[label==0,0],feature[label==0,1])
	pylab.scatter(feature[label==1,0],feature[label==1,1])
	pylab.plot([0, 1.0+margin], [1.0+margin, 0])
	pylab.plot([0, 1.0-margin], [1.0-margin, 0])
	pylab.xlim(0,1)
	pylab.ylim(0,1)
	if name==None:
		pylab.show()
	else:
		pylab.savefig(name)

def random_points(start, end, size, random_state):
	np.random.seed(random_state)
	x1 = np.random.uniform(start, end, size)
	x2 = np.random.uniform(start, end, size)
	feature =  np.vstack([x1,x2]).transpose()
	label = np.random.choice(2,size)
	return feature, label
	