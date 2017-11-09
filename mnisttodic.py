import cPickle, gzip
import numpy as np
import pickle as pkl

f = gzip.open('./mnist.pkl.gz', 'rb');
train_set, valid_set, test_set = cPickle.load(f)
f.close()

digits={'train_image':train_set[0],'train_label':train_set[1],'test_image':test_set[0],'test_label':test_set[1]}

pkl.dump(digits, open("./packeddata/mnist.pkl","wb" ))
