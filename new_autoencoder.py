# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 01:39:42 2018

@author: cbel-amira
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed
#from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0,
					   help='0 for training, 1 for testing')
parser.add_argument('--resume', type=bool, default=True,
					   help='True for resuming the model; False for initialization')
parser.add_argument('--NotMNIST', type=int, default=1,
					   help='Use NotMnist dataset for testing')
parser.add_argument('--learning_rate', type=float, default=1e-2,
					   help='learning rate of model')
parser.add_argument('--epochs', type=int, default=2000,
					   help='epoch of training')
parser.add_argument('--batch_size', type=int, default=256,
					   help='batch size of training')
parser.add_argument('--display_step', type=int, default=10,
					   help='display step to show info during training')
parser.add_argument('--example_to_show', type=int, default=4,
					   help='number of example to show testing cases')
parser.add_argument('--model_file', type=str, default="./new_model/save_net.ckpt",
					   help='the path of model')
parser.add_argument('--notmnist_path', type=str, default="./notMNIST_sample/",
					   help='the path of notmnist dataset')

args = parser.parse_args()


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


n_input = 784 # image size reshape from 28*28

# hidden layer settings
n_hidden_1 = 50 # 1st layer num features
n_hidden_2 = 50 # 2nd layer num features
n_hidden_3 = 50 # 3th Layer number features


X = tf.placeholder(tf.float32, [None, n_input])


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3,n_input]))
       
	}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_input]))
	}

def img2array(img_name):           
    img = Image.open(img_name)
    img = np.array(img)
    return img

# Building the encoder
def auto_encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    # Encoder Hidden layer with sigmoid activation #2    
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    # Output layer of signmoid function
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                   biases['encoder_b4']))   
    return layer_4


encoder_op = auto_encoder(X)

y_pred = encoder_op
y_true = X

cost = tf.reduce_mean(tf.square(y_pred - y_true))
optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)


saver = tf.train.Saver()
init = tf.global_variables_initializer()

# training mode
if args.mode == 0:
    with tf.Session() as sess:
        sess.run(init)
        if args.resume:
            saver.restore(sess,args.model_file)
            
        total_batch = int(mnist.train.num_examples/args.batch_size)
        for ep in range(args.epochs):
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(args.batch_size)  # max(x) = 1, min(x) = 0
                
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                
                if i % args.display_step == 0:
                    print("Epoch:", '%d' % (ep+1), "cost=", "{:.9f}".format(c))
            
            #save model every epoch
            save_path = saver.save(sess, args.model_file)
            print("Save to path",save_path)

# testing mode                                      
else:
    with tf.Session() as sess:
        saver.restore(sess, args.model_file)    
        
        if args.NotMNIST == 1:# test on NotMnist samples
            
            input_nmnist = []
            for filename in os.listdir(args.notmnist_path):
                input_nmnist.append(img2array(args.notmnist_path+filename))
               
            input_nmnist = np.array(input_nmnist)
            input_nmnist = input_nmnist.astype('float32')
            input_nmnist = input_nmnist/255           
            input_nmnist = np.reshape(input_nmnist, [-1, n_input])
            
            for sample in input_nmnist:
                sample = np.reshape(sample, [1,n_input])
                encode_decode, c_test = sess.run([y_pred, cost], feed_dict={X:sample})
                
                f,sub = plt.subplots(2,1,figsize=(1,2))
                sub[0].imshow(np.reshape(sample,(28,28)))
                sub[1].imshow(np.reshape(encode_decode,(28,28)))
                
                plt.show()
                print("cost=", "{:.9f}\n".format(c_test))
            
        else:# test on Mnist samples
            for sample_t in range(args.example_to_show):
                test_in = np.reshape(mnist.test.images[sample_t],[1,n_input])
                encode_decode, c_test = sess.run([y_pred, cost], feed_dict={X:test_in})
                
                f,sub = plt.subplots(2,1,figsize=(1,2))
                sub[0].imshow(np.reshape(test_in,(28,28)))
                sub[1].imshow(np.reshape(encode_decode,(28,28)))
                    
                plt.show()
                print("cost=", "{:.9f}".format(c_test))
    



















