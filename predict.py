import tensorflow as tf
import pickle as pkl
import numpy as np
import dataset as Data
import time

print "Loading Dataset.."
dataset =pkl.load(open("./packeddata/full.pkl"))
category=dataset['category']  #numbers of categories
pictures=dataset['pictures']
labels=dataset['labels']

mnist = Data.read_data_sets(pictures,labels,category, one_hot=True,reshape=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


with tf.Session() as sess:
  #  with tf.device("/cpu:0"):
        x = tf.placeholder("float", shape=[None, 784])
        y_ = tf.placeholder("float", shape=[None, category])

        W = tf.Variable(tf.zeros([784,category]))
        b = tf.Variable(tf.zeros([category]))


        y = tf.nn.softmax(tf.matmul(x,W) + b)
        cross_entropy = -tf.reduce_sum(y_*tf.log(y))

        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1,28,28,1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder("float")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, category])
        b_fc2 = bias_variable([category])
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        saver = tf.train.Saver()
        saver.restore(sess,"./Models/full.ckpt")

        start=time.time()
        print "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        end =time.time()
        print (end-start)*10000/mnist.test.images.shape[0]


        prediction=tf.argmax(y_conv, 1)
        print prediction.eval(feed_dict={x:mnist.test.images[0:2], keep_prob: 1.0})
      #  print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images[0:2], y_: mnist.test.labels[0:2], keep_prob: 1.0})