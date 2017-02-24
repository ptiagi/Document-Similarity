import sys
import os
import math
from math import*
from decimal import Decimal
from sklearn.decomposition import PCA
import numpy as np

f = open(os.path.abspath(sys.argv[1]))

document = []
vocabulary = []
uniqueWords = []
vector = []
tf = []
dict_idf = {}
idf = []
d = []
manhattan = []
euclidean = []
supremum = []
cos_sim =[]
pca_euclidean_dist = []

for line in f:
	document.append(line)
for i in document:
	vocabulary.append(i.rstrip().split(" "))

for words in vocabulary:
	for w in words:
		if w not in uniqueWords:
			uniqueWords.append(w)
print(len(uniqueWords))

for words in vocabulary:
	dict_words = {}
	for w in words:
		dict_words[w] = dict_words.get(w, 0) + 1
	vector.append(dict_words.copy())


doc_num = 0
for count in vector:
	d_tf = {}
	for wrd in count:
		d_tf[wrd] = float(count.get(wrd))/float(len(vocabulary[doc_num]))
	tf.append(d_tf.copy())
	doc_num += 1


for wrd_idf in uniqueWords:
	c = 0
	for doc in vocabulary:
		if wrd_idf in doc:
			c += 1 
	dict_idf[wrd_idf] = math.log(len(vocabulary)/c)


for t in tf:
	temp = []	
	for wrd in dict_idf.keys():
		temp.append(dict_idf.get(wrd)*t.get(wrd,0))
	d.append(temp)


### Minkowski Distance ###
def root(number, h):
	root_value = 1/float(h)
	return round((number ** root_value), 4)

def minkowski_distance(x, y, h):
	sum = 0
	for a, b in zip(x,y):
		sum += pow(abs(a-b),h)
	return float(root(sum,h))

#(a). Manhattan distance, h =1
query = d[len(d)-1]
def manhattan_distance(d):
	count = 0
	for i in d:
		count += 1
		manhattan.append((count, minkowski_distance(i, query, 1)))	
	return(sorted(manhattan, key=lambda x: x[1]))

result = [x[0] for x in manhattan_distance(d)[0:5]]
print(' '.join(map(str, result)))

def euclidean_distance(d):
	count = 0
	for i in d:
		count += 1
		euclidean.append((count, minkowski_distance(i, query, 2)))	
	return(sorted(euclidean, key=lambda x: x[1]))

result = [x[0] for x in euclidean_distance(d)[0:5]]
print(' '.join(map(str, result)))

max_dist = 0

def supremum_distance(d):
	doc_num = 0
	for i in d:
		max_dist = 0
		doc_num += 1
		for count in range(len(i)):
			max_dist = round(max(max_dist, abs(i[count] - query[count])),4)
		supremum.append((doc_num, max_dist))	
	return(sorted(supremum, key=lambda x: x[1]))

result = [x[0] for x in supremum_distance(d)[0:5]]
print(' '.join(map(str, result)))


def cosine_similarity(d):
	doc_num = 0
	for i in d:
		n = []
		docmnt = []
		q = []
		doc_num += 1
		for k in range(len(i)):
			n.append(i[k]*query[k])
			docmnt.append(i[k]*i[k])
			q.append(query[k]*query[k])
		numerator = sum(n)
		denominator = root(sum(docmnt),2)*root(sum(q),2)
		c_sim = round(float(numerator/denominator),4)
		cos_sim.append((doc_num, c_sim))
	return(sorted(cos_sim,key = lambda x: x[1],reverse=True))

result = [x[0] for x in cosine_similarity(d)[0:5]]
print(' '.join(map(str, result)))


### PCA ###
pca = PCA(n_components = 2)
principal_components = pca.fit_transform(d)

#### Euclidean Distance of Two projected Vectors ###

def pca_euclidean_distance(principal_components):
	count = 0
	for i in principal_components:
		count += 1
		pca_euclidean_dist.append((count, minkowski_distance(i, principal_components[len(principal_components)-1], 2)))	
	return(sorted(pca_euclidean_dist, key=lambda x: x[1]))

result = [x[0] for x in pca_euclidean_distance(principal_components)[0:5]]
print(' '.join(map(str, result)))


