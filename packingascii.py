import cv2
import numpy as np
from glob import glob
import pickle as pkl
from sklearn.utils import shuffle

data_dir = "./NIST//"

dir = [folder.split('//')[-1] + "/*.png" for folder in glob(data_dir + '*')]
X = []
y = []

for j,f in enumerate(dir):
    progress=0
    filenames = glob(f)
    i=0
    while progress<10000:
        if i>=len(filenames):
            i=0
        img_frame=filenames[i]
        data = cv2.imread(img_frame)
        data_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        X.append(data_gray)
        y.append(int(f[len(data_dir)-1:-8]))
        print f[len(data_dir)-1:-5] + str(progress) + ' out of 10000 '
        progress+=1
        i+=1

symbol_dir = "./symbols//"

dir2 = [folder.split('//')[-1] + "/*.jpg" for folder in glob(symbol_dir + '//*')]

ascll = [40,45,42,43,41]

for j,f in enumerate(dir2):
    print f
    filenames = glob(f)
    progress=0
    i=0
    while progress<10000:
        if i>=len(filenames):
            i=0
        img_frame=filenames[i]
        data = cv2.imread(img_frame)
        data_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        X.append(data_gray)
        y.append(ascll[j])
        print chr(ascll[j])+'/' + str(progress) + ' out of 10000 '
        progress += 1
        i += 1

X=np.array(X)
y=np.array(y)
X=X[...,np.newaxis]
print X.shape
print y.shape
print len(dir)+len(dir2)

X,y=shuffle(X,y,random_state=0)

samples=300000
X=X[:samples]
y=y[:samples]


dic={'pictures':X,'labels':y,'category':len(dir)+len(dir2)}
print "start dumping"
pkl.dump(dic, open("./packeddata/full.pkl","wb" ))