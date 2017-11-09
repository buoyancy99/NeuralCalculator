import cv2
import numpy as np
import tensorflow as tf
import pickle as pkl
import dataset as Data
#import matplotlib.pyplot as plt
#from graph import *

MAXBOX=30
ind = 0

def divSign(asNum):
    if True:
    #if ind<=3: #change ind to the number of sign of division in the formula. This line will not draw the proper sign but can output the paramenters you wanna use only once 
	upDot = 0
	downDot = 0
	for dot in dots:
	    if (rect[0]<=dot[0] + 0.5*dot[2]<=(rect[0]+rect[2])): #position
	    #TODO: adjusting parameters
		if ((rect[2]*rect[3]*0.21)*0<=(dot[2]*dot[3])<=(rect[2]*rect[3]*0.21)): #size
		    if (abs(dot[1]-rect[1])<=0.7*rect[2]): #distance
		    #print str(dot[0]) + " " + str(dot[1])
			if dot[1]<rect[1]:
			    upDot +=1

			elif dot[1]>rect[1]:
			    downDot +=1
			else:
			    print "Fail"
    
    if upDot != 0 and downDot != 0:
        asNum = 47

    return asNum


# to test if two boxes are in the same line
def inline(box1,box2):
    state=True
    operators=[ord('-'),ord('*'),ord('+')]
    hl1=box1[2]/2
    hl2=box2[2]/2
    cy1=box1[0]+hl1
    cy2=box2[0]+hl2
    if (cy1<box2[0] and cy2>box1[0]+box1[2]) or (cy1>box2[0]+box2[2] and cy2<box1[0]):
        state=False
    if ((box1[4] in operators) and cy1>box2[0] and cy1<box2[0]+box2[2]) or ((box2[4] in operators) and cy2>box1[0] and cy2<box1[0]+box1[2]):
        state=True
    return state


def getline(ci):
    element=[]

    #if it is the last element of this line
    if ci==symbol.shape[0]-1:
        if(symbol[ci,5])==0 :
            symbol[ci, 5] = 1
            return [symbol[ci].tolist()]
    else:
        for headindex in range(ci+1,symbol.shape[0]):
            if(symbol[headindex,5]==0 and inline(symbol[ci],symbol[headindex])):
                element=getline(headindex)
                break
        symbol[ci, 5] = 1
        element.append(symbol[ci].tolist())

        return element

print "Loading Dataset.."
dataset = pkl.load(open("./packeddata/full.pkl"))
category = dataset['category']  # numbers of categories
pictures = dataset['pictures']
labels = dataset['labels']
mathset = Data.read_data_sets(pictures, labels, category, one_hot=True, reshape=True)
maptable = pkl.load(open("./maptable.pkl"))
print "completed!"

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    with tf.device("/cpu:0"):
        x = tf.placeholder("float", shape=[None, 784])
        y_ = tf.placeholder("float", shape=[None, category])

        W = tf.Variable(tf.zeros([784, category]))
        b = tf.Variable(tf.zeros([category]))

        y = tf.nn.softmax(tf.matmul(x, W) + b)
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, category])
        b_fc2 = bias_variable([category])
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver = tf.train.Saver()
        saver.restore(sess, "./Models/full.ckpt")


        prediction = tf.argmax(y_conv, 1)
    #    print prediction.eval(feed_dict={x: mathset.test.images[0:2], keep_prob: 1.0})



       # im = cv2.imread("./test9.jpg")
        cap = cv2.VideoCapture(0)


        k=0
        p=0
        count=0
        tempfirst=[]
        while(True):
            p=p+1
         
            _,im=cap.read()

            '''
            upb=im.shape[0]*0.2
            downb=im.shape[0]*0.8
            leftb=im.shape[1]*0.2
            rightb=im.shape[1]*0.8
            im = im[upb:downb,leftb:rightb]
            '''

            #setup
            cv2.imshow("origin", im)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)
           # ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
            im_th=cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
            framenum=10
            """
            if count==0:
                sum=np.zeros((im_th.shape[0],im_th.shape[1]))
                count+=1
            else:
                if count<framenum:
                    sum=sum+im_th
                    count+=1
                    tempfirst.append(im_th)
                else:
                    sum=sum-tempfirst[0]+im_th
                    tempfirst=tempfirst[1:]
                    tempfirst.append(im_th)
                    im_th=np.rint(sum/framenum)
                    im_th =im_th.astype(np.uint8)
                    _,im_th = cv2.threshold(im_th,127,255, cv2.THRESH_BINARY)
            """
            #morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
           # im_th = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
            im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)
            #im_th = cv2.dilate(im_th, kernel, iterations=1)

            #scope of the noise
            AREA = im.size
            LargeBox = 0.03 * AREA
            SmallBox = 0.0001 * AREA

            #reverse: im_th is balck while frame is white
            frame = 255-im_th
            frame2=frame

            #contours and boxes
            _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


            rects = [cv2.boundingRect(ctr) for ctr in ctrs if SmallBox <= cv2.contourArea(ctr) <= LargeBox]
            dots = [cv2.boundingRect(ctr) for ctr in ctrs if (cv2.contourArea(ctr)<SmallBox) ]
            boxes = []


            #when the number of boxes is limited
            if  len(rects)<=MAXBOX:
                for rect in rects:

                    roi = np.ones((28, 28)) * 255
                    section = frame2[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                    w = rect[2]
                    h = rect[3]

                    #control the size within 24*24 to match the training samples
                    scale = 24.0 / max(w, h)
                    rw = int(w * scale) //2 * 2
                    rh = int(h * scale) //2 * 2


                    #if it is a number(not noise), resize and predict with hog feather
                    if rw*rh > 20:

                        #TODO?
                        roi[14 - (rh // 2):14 - (rh // 2) + rh, 14 - (rw // 2):14 - (rw // 2) + rw] = cv2.resize(section, (rw, rh),interpolation=cv2.INTER_CUBIC)

                        nbr = prediction.eval(feed_dict={x: np.array([roi.reshape(784)], 'float32'), keep_prob: 1.0})
                        nbr = [maptable[nbr[0]]]
                        #if the digit is complete, put text and put it in temp
                        if (rect[0]!=1 and rect[1]!=1 and rect[0] + rect[2]!=frame2.shape[1]-1 and rect[1] + rect[3]!=frame2.shape[0]-1):
                            #text

                            if nbr[0] == 45:
                                nbr[0] = divSign(nbr[0])

                            cv2.putText(frame2, chr(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

                            #temp
                            temp = [rect[1], rect[0], rect[3], rect[2], int(nbr[0]),0]
                            boxes.append(temp)

                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)
                    #sort the boxes
                    boxes.sort(key=lambda i: i[1])  # lambda x:x[1]



            symbol=np.array(boxes)
            command=[]
            i=0
            j=0
            group=[]

            while(i<symbol.shape[0]):
                cmdh=[]
                #get a line h
                if(symbol[i,5]==0):
                    h=getline(i)[::-1]
                    symbol[i, 5]=1

                    for item in h:
                        cmdh.append(item[4])
                    if h:
                        group.append(h)
                if cmdh:
                    command.append(cmdh)
                i+=1
            commands=np.array(command)
            #print group
            #print commands

           # print symbol
            strcmd=[]
            for i in command:
                cmd=""
                for j in i:
                    cmd+=chr(j)

                try:
                    strcmd.append(eval(cmd))

                except:
                    strcmd.append("Error")

            #print strcmd
            for index in range(len(strcmd)):
                  temp=group[index][-1]
                  cv2.putText(frame2, "="+str(strcmd[index]), ( temp[1]+temp[3],temp[0]+temp[2]), cv2.FONT_HERSHEY_DUPLEX, temp[2]/20, (0, 255, 255), 4)

                  '''
                  if strcmd[index] != '#':
                    drawGraph(cmd)
                  '''
            result = cv2.namedWindow('result', flags=0)
            cv2.imshow("result", frame2)
            cv2.waitKey(25)



      #  cap.release()
        cv2.destroyAllWindows()

"""
    symbol = np.array([[105, 33, 93, 68, 57, 0],
                       [259, 40, 100, 76, 43, 0],
                       [422, 58, 18, 66, 45, 0],
                       [104, 136, 111, 55, 57, 0],
                       [256, 176, 96, 87, 43, 0],
                       [406, 201, 20, 67, 45, 0],
                       [95, 266, 117, 55, 57, 0],
                       [251, 321, 92, 79, 43, 0],
                       [398, 340, 21, 71, 45, 0],
                       [96, 391, 98, 56, 57, 0],
                       [252, 445, 90, 77, 52, 0],
                       [400, 471, 14, 60, 45, 0],
                       [84, 514, 122, 81, 57, 0]]
                      )
print getline(symbol)
"""
