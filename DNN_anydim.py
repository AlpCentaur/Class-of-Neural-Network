## Author: Luca Albert Wulf, python class DNN



import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from scipy.linalg import qr
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.client import timeline
import time
import os, sys
import cv2
import random
from sklearn import preprocessing

class NeuralNet(object):
    
    def __init__(self, learning_rate, epochs, batch_size, min_valW, max_valW, min_valb1, max_valb1, n_trainingsamples, n_testsamples, Seed, Logging_Activated, alpha, Graphnum, NoW, inputdim, outputdim, SaveOrNot):

        
        # Python optimisation variables
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.min_val = min_valW
        self.max_val = max_valW
        self.min_valW = min_valW
        self.max_valW = max_valW
        self.min_valb1 = min_valb1 
        self.max_valb1 = max_valb1
        self.n_trainingsamples = n_trainingsamples 
        self.n_testsamples = n_testsamples
        self.Seed = Seed
        self.Logging_Activated = Logging_Activated
        self.alpha = alpha
        self.Graphnum = Graphnum
        self.SaveOrNot = SaveOrNot
        
        self.NoW = NoW
        # Extend the size of the network, dealing with higher dimensionality
        # Number of waveletterms depending on the function that we want to approximate (sinus in  an interval of 2pi needs NoW=2)
        
        # Dimension of input needs to e specified
        self.d = 2
        # max value of function to be approximated needs to be specified, in case of sinus --> q equals 1
        self.q = 1
        
        # Get the current working directory for later
        self.cwd = os.getcwd()
    
    def InitT1(self):
        
        # training data placeholders
        # input x 
        self.x = tf.placeholder(tf.float32, [None, 2])
        # output data placeholder - should return the sinus values
        self.y = tf.placeholder(tf.float32, [None, 1])


        self.weight_decay = tf.constant(self.alpha, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
        with tf.variable_scope("T1", reuse=tf.AUTO_REUSE):
        
            # define weights connecting the input to the first hidden layer
            W1 = tf.get_variable(name='W1', shape=[ 2 , self.NoW * 4], initializer=tf.random_uniform_initializer(minval=self.min_val , maxval=self.max_val, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            
            self.W1 = W1
            
            b1 = tf.get_variable(name='b1', shape=[self.NoW * 4], initializer=tf.random_uniform_initializer(minval=self.min_val , maxval=self.max_val, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            
            self.b1 = b1
            
            # define weights connecting the  first hidden layer to the second hidden layer
            W2 = tf.get_variable(name='W2', shape=[4 * self.NoW, self.NoW], initializer=tf.random_uniform_initializer(minval=self.min_val , maxval=self.max_val, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            
            self.W2 = W2
            
            b2 = tf.get_variable(name='b2', shape=[self.NoW], initializer=tf.random_uniform_initializer(minval=self.min_val , maxval=self.max_val, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            
            self.b2 = b2
            
            # define weights connecting the  second hidden layer to the output layer
            W3 = tf.get_variable(name='W3', shape=[self.NoW , 1], initializer=tf.random_uniform_initializer(minval=self.min_val , maxval=self.max_val, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
            
            self.W3 = W3
        


        # calculate the output of the first hidden layer
        hidden_out1 = tf.add(tf.matmul(self.x, self.W1), self.b1)
        hidden_out1 = tf.nn.relu(hidden_out1)
        
        self.hidden_out1 = hidden_out1
            
        # calculate the output of the first hidden layer
        hidden_out2 = tf.add(tf.matmul(self.hidden_out1, self.W2), self.b2)
        hidden_out2 = tf.nn.relu(hidden_out2)
        
        self.hidden_out2 = hidden_out2
        
        # calculate the output of the first hidden layer
        out = tf.matmul(self.hidden_out2, self.W3)
        
        self.out = out
        
        if self.SaveOrNot == True:
            saver = tf.train.Saver()
        else:
            pass
        
    def InitT2(self):
        
        # training data placeholders
        # input x 
        self.x = tf.placeholder(tf.float32, [None, 2])
        # output data placeholder - should return the sinus values
        self.y = tf.placeholder(tf.float32, [None, 1])


        self.weight_decay = tf.constant(self.alpha, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
        with tf.device('/cpu:0'):
            with tf.variable_scope("T2", reuse=tf.AUTO_REUSE):
                # define weights connecting the input to the first hidden layer
                self.W1 = tf.get_variable(name='W1', shape=[ 2 , self.NoW * 4], initializer=tf.random_uniform_initializer(minval=self.min_valW , maxval=self.max_valW, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                self.b1 = tf.get_variable(name='b1', shape=[self.NoW * 4], initializer=tf.random_uniform_initializer(minval=self.min_valb1 , maxval=self.max_valb1, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                # define W2 to restrict topology based on paper of Kutyniok and Petersen
                W2 = np.zeros((self.NoW , self.NoW * 4), dtype=np.float32)

                for i in range(self.NoW):    
                    W2[i][i*4] = 1
                    W2[i][i*4+1] = -1
                    W2[i][i*4+2] = -1
                    W2[i][i*4+3] = 1


                self.W2 = np.transpose(W2)

                # as d = 1 => b2 = 0 following the paper approx prop of deep nns
                #b2 = tf.Variable(tf.random_uniform([2], minval=min_val , maxval=max_val, seed=Seed), name='b2')

                b2 = np.zeros((self.NoW,1), dtype=np.float32)
                for i in range(self.NoW):
                    b2[i][0] = - (self.d - 1) * self.q


                b2 = np.transpose(b2)
                
                self.b2 = b2
                
                # define weights connecting the  second hidden layer to the output layer
                W3 = tf.Variable(tf.random_uniform([self.NoW, 1], minval=self.min_valW , maxval=self.max_valW, seed=self.Seed), name='W3')
                
            self.W3 = W3


            # calculate the output of the first hidden layer
            hidden_out1 = tf.add(tf.matmul(self.x, self.W1), self.b1)
            self.hidden_out1 = tf.nn.relu(hidden_out1)
            # calculate the output of the first hidden layer
            hidden_out2 = tf.add(tf.matmul(self.hidden_out1, self.W2), self.b2)
            self.hidden_out2 = tf.nn.relu(hidden_out2)
            # calculate the output of the first hidden layer
            self.out = tf.matmul(self.hidden_out2, self.W3)
            
            if self.SaveOrNot == True:
                saver = tf.train.Saver()
            else:
                pass
        
    def InitT3(self):
        # training data placeholders
        # input x 
        self.x = tf.placeholder(tf.float32, [None, 2])
        # output data placeholder - should return the sinus values
        self.y = tf.placeholder(tf.float32, [None, 1])


        with tf.device('/cpu:0'):
            with tf.variable_scope("T3", reuse=tf.AUTO_REUSE):
                # define weights connecting the input to the first hidden layer
                #W1 = tf.Variable(tf.random_uniform([2 , NoW * 4], minval=min_valW , maxval=max_valW, seed=Seed), name='W1')

                W1 = np.zeros((2, self.NoW * 4), dtype=np.float32)

                for i in range(int(self.NoW * 0.5)):
                    W1[0][i*8] = 1
                    W1[0][i*8+1] = 1
                    W1[0][i*8+2] = 1
                    W1[0][i*8+3] = 1
                    W1[0][i*8+4] = 0
                    W1[0][i*8+5] = 0
                    W1[0][i*8+6] = 0
                    W1[0][i*8+7] = 0
                    W1[1][i*8] = 0
                    W1[1][i*8+1] = 0
                    W1[1][i*8+2] = 0
                    W1[1][i*8+3] = 0
                    W1[1][i*8+4] = 1
                    W1[1][i*8+5] = 1
                    W1[1][i*8+6] = 1
                    W1[1][i*8+7] = 1
                    
                self.W1 = W1
                
                weight_decay = tf.constant(self.alpha, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
                self.weight_decay = weight_decay    
                
                
                # define weights connecting the  first hidden layer to the second hidden layer
                #W2 = tf.Variable(tf.random_uniform([8, 2], minval=min_val , maxval=max_val, seed=Seed), name='W2')
                #W1 = tf.get_variable(name='W1', shape=[ 1 , self.NoW * 4], initializer=tf.random_uniform_initializer(minval=self.min_valW , maxval=self.max_valW, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                #self.W1 = W1
                
                #W1 = tf.Variable(tf.random_uniform([ 1 , NoW * 4], minval=min_valW , maxval=max_valW, seed=Seed), name='W1')
                b1 = tf.get_variable(name='b1', shape=[self.NoW * 4], initializer=tf.random_uniform_initializer(minval=self.min_valb1 , maxval=self.max_valb1, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                self.b1 = b1
                
                #b1 = tf.Variable(tf.random_uniform([NoW * 4], minval=min_valb1 , maxval=max_valb1, seed=Seed) ,name='b1')
                # define weights connecting the  first hidden layer to the second hidden layer
                #W2 = tf.Variable(tf.random_uniform([8, 2], minval=min_val , maxval=max_val, seed=Seed), name='W2')


                

                # define W2 to restrict topology based on paper of Kutyniok and Petersen
                W2 = np.zeros((self.NoW , self.NoW * 4), dtype=np.float32)
                
                for i in range(self.NoW):    
                    W2[i][i*4] = 1
                    W2[i][i*4+1] = -1
                    W2[i][i*4+2] = -1
                    W2[i][i*4+3] = 1


                W2 = np.transpose(W2)
                
                self.W2 = W2
                
                # as d = 1 => b2 = 0 following the paper approx prop of deep nns
                #b2 = tf.Variable(tf.random_uniform([2], minval=min_val , maxval=max_val, seed=Seed), name='b2')

                b2 = np.zeros((self.NoW * 4 ,1), dtype=np.float32)
                for i in range(self.NoW):
                    b2[i*4][0] = 0
                    b2[i*4+1][0] = -1
                    b2[i*4+2][0] = -2
                    b2[i*4+3][0] = -3


                b2 = np.transpose(b2)
                
                self.b2 = b2
                
                # define weights connecting the  second hidden layer to the output layer

                W3 = tf.get_variable(name='W3', shape=[self.NoW, 1], initializer=tf.random_uniform_initializer(minval=self.min_valW , maxval=self.max_valW, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                self.W3 = W3
                
                #W3 = tf.Variable(tf.random_uniform([NoW, 1], minval=min_valW , maxval=max_valW, seed=Seed), name='W3')
                b3 = tf.get_variable(name='b3', shape=[1], initializer=tf.random_uniform_initializer(minval=self.min_valW , maxval=self.max_valW, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                self.b3 = b3
                
                #b3 = tf.Variable(tf.random_uniform([1], minval=min_valW , maxval=max_valW, seed=Seed) , name='b3')
                
            

            # calculate the output of the first hidden layer
            hidden_out1 = tf.add(tf.matmul(self.x, self.W1) , self.b1)
            
            self.hiddenout1 = hidden_out1
            
            #hidden_out1 = tf.nn.relu(hidden_out1)
            # calculate the output of the first hidden layer
            hidden_out2 = tf.matmul(tf.nn.relu(tf.add(hidden_out1, self.b2)) , self.W2)
            hidden_out2 = tf.nn.relu(hidden_out2)
            
            self.hiddenout2 = hidden_out2
            
            # calculate the output of the first hidden layer
            out = tf.matmul(hidden_out2, self.W3) + self.b3
            
            self.out = out
            
            if self.SaveOrNot == True:
                saver = tf.train.Saver()
            else:
                pass
    
    def InitT4(self):
        # training data placeholders
        # input x 
        self.x = tf.placeholder(tf.float32, [None, self.inputdim])
        # output data placeholder - should return the sinus values
        self.y = tf.placeholder(tf.float32, [None, self.outputdim])
        
        
        with tf.device('/cpu:0'):
            with tf.variable_scope("T4", reuse=tf.AUTO_REUSE):
                # define weights connecting the input to the first hidden layer
                #W1 = tf.Variable(tf.random_uniform([2 , NoW * 4], minval=min_valW , maxval=max_valW, seed=Seed), name='W1')

                W1 = np.zeros((self.inputdim, self.inputdim * self.NoW * 4), dtype=np.float32)

                for i in range(self.inputdim):
                    for j in range(self.NoW):
                        W1[i][i*self.NoW*4+j*4] = 1
                        W1[i][i*self.NoW*4+j*4 + 1] = 1
                        W1[i][i*self.NoW*4+j*4 + 2] = 1
                        W1[i][i*self.NoW*4+j*4 + 3] = 1
                    
                self.W1 = W1
                
                weight_decay = tf.constant(self.alpha, dtype=tf.float32) # your weight decay rate, must be a scalar tensor.
                self.weight_decay = weight_decay    
                
                
                # define weights connecting the  first hidden layer to the second hidden layer
                #W2 = tf.Variable(tf.random_uniform([8, 2], minval=min_val , maxval=max_val, seed=Seed), name='W2')
                #W1 = tf.get_variable(name='W1', shape=[ 1 , self.NoW * 4], initializer=tf.random_uniform_initializer(minval=self.min_valW , maxval=self.max_valW, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                #self.W1 = W1
                
                #W1 = tf.Variable(tf.random_uniform([ 1 , NoW * 4], minval=min_valW , maxval=max_valW, seed=Seed), name='W1')
                b1 = tf.get_variable(name='b1',dtype=tf.float32, shape=[self.NoW * 4 * self.inputdim], initializer=tf.random_uniform_initializer(minval=self.min_valb1 , maxval=self.max_valb1, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                self.b1 = b1
                
                #b1 = tf.Variable(tf.random_uniform([NoW * 4], minval=min_valb1 , maxval=max_valb1, seed=Seed) ,name='b1')
                # define weights connecting the  first hidden layer to the second hidden layer
                #W2 = tf.Variable(tf.random_uniform([8, 2], minval=min_val , maxval=max_val, seed=Seed), name='W2')


                

                # define W2 to restrict topology based on paper of Kutyniok and Petersen
                W2 = np.zeros((self.NoW * self.inputdim , self.NoW * 4 * self.inputdim), dtype=np.float32)
                
                for i in range(self.NoW * self.inputdim):    
                    W2[i][i*4] = 1
                    W2[i][i*4+1] = -1
                    W2[i][i*4+2] = -1
                    W2[i][i*4+3] = 1


                W2 = np.transpose(W2)
                
                self.W2 = W2
                
                # as d = 1 => b2 = 0 following the paper approx prop of deep nns
                #b2 = tf.Variable(tf.random_uniform([2], minval=min_val , maxval=max_val, seed=Seed), name='b2')

                b2 = np.zeros((self.NoW * 4 * self.inputdim ,1), dtype=np.float32)
                for i in range(self.NoW):
                    b2[i*4][0] = 0
                    b2[i*4+1][0] = -0.3
                    b2[i*4+2][0] = -0.3
                    b2[i*4+3][0] = -0.6


                b2 = np.transpose(b2)
                
                self.b2 = b2
                
                # define weights connecting the  second hidden layer to the output layer

                W3 = tf.get_variable(name='W3',dtype=tf.float32, shape=[self.NoW * self.inputdim, self.outputdim], initializer=tf.random_uniform_initializer(minval=self.min_valW , maxval=self.max_valW, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                self.W3 = W3
                
                #W3 = tf.Variable(tf.random_uniform([NoW, 1], minval=min_valW , maxval=max_valW, seed=Seed), name='W3')
                b3 = tf.get_variable(name='b3',dtype=tf.float32, shape=[self.outputdim], initializer=tf.random_uniform_initializer(minval=self.min_valW , maxval=self.max_valW, seed=self.Seed) , regularizer=tf.contrib.layers.l2_regularizer(self.weight_decay))
                
                self.b3 = b3
                
                #b3 = tf.Variable(tf.random_uniform([1], minval=min_valW , maxval=max_valW, seed=Seed) , name='b3')
                
            

            # calculate the output of the first hidden layer
            hidden_out1 = tf.add(tf.matmul(self.x, self.W1) , self.b1)
            
            self.hiddenout1 = hidden_out1
            
            #hidden_out1 = tf.nn.relu(hidden_out1)
            # calculate the output of the first hidden layer
            hidden_out2 = tf.matmul(tf.nn.relu(tf.add(hidden_out1, self.b2)) , self.W2)
            hidden_out2 = tf.nn.relu(hidden_out2)
            
            self.hiddenout2 = hidden_out2
            
            # calculate the output of the first hidden layer
            out = tf.matmul(hidden_out2, self.W3) + self.b3
            
            self.out = out
            
            if self.SaveOrNot == True:
                self.saver = tf.train.Saver()
            else:
                pass
    
    def DefineCostFunction(self):
        
        #define cost function
        self.cost = tf.reduce_mean(abs( self.out - self.y ))   # reduce_mean makes 1/m times sum to m 
    
    def DefineGradientDescentAlgorithm(self):
        # define Stochastic Gradient Descent optimiser
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    
    
    
    def Get_Image_TrainData(self, directory_im, directory_labels):
        
        images = os.listdir(self.cwd + '/' + directory_im)
        labels = os.listdir(self.cwd + '/' + directory_labels)
        self.Xtrain = []
        self.Ytrain = []
        for image in images:
            print('next image..')
            for label in labels:
                if label[:3] == image[:3]:
                    #print(image)
                    #print(label)
                    #print(directory_im)
                    img = cv2.imread(directory_im + '/' + image, 1)
                    img_label = cv2.imread(directory_labels + '/' + label, 1)
                    print(type(img))
                    print('len',len(img))
                    print('np to list',np.ndarray.tolist(img))
                    # Calculate gradient 
                    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
                    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
                    print('gx',len(gx), gx , 'gy', len(gy), gy)
                    # Python Calculate gradient magnitude and direction ( in degrees ) 
                    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                    print('mag',len(mag) , mag ,'angle',len(angle), angle)
                    edges = cv2.Canny(img,100,200)
                    print(edges)
                    #cv2.imshow("Image",angle)
                    #cv2.waitKey(0)
                    
                    ## Normalize the feature data
                    #edges_normalized = preprocessing.normalize(edges, norm='l2')
                    #gx_normalized = preprocessing.normalize(gx, norm='l2')
                    #gy_normalized = preprocessing.normalize(gy, norm='l2')
                    #mag_normalized = preprocessing.normalize(mag, norm='l2')
                    #angle_normalized = preprocessing.normalize(angle, norm='l2')
                    #img_normalized = preprocessing.normalize(img, norm='l2')
                    #img_label_normalized = preprocessing.normalize(img_label, norm='l2')
                    
                    edges_scaled = edges/ 255
                    mag_scaled = mag / 983.71234
                    angle_scaled = angle / 359.87012
                    img_scaled = img / 255
                    img_label_scaled = img_label / 255
                    
                    
                    
                    X = img
                    #print(max(np.concatenate(X, axis=None)))
                    #print(min(np.concatenate(X, axis=None)))
                    
                    
                    #binarizer = preprocessing.Binarizer().fit(img_label_scaled)
                    #img_label_binary = binarizer.transform(img_label_scaled)
                    
                    lenim = len(img)
                    for m in range(lenim):
                        for n in range(lenim):
                            
                            if lenim - 2 >= n >= 1:
                                if lenim - 2 >= m >= 1:
                                    trainsample = []
                                    trainsamp = []
                                    trainsam = []
                                    
                                    trainsample.extend(np.concatenate(img_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                                    trainsample.extend(np.concatenate(edges_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                                    #trainsample.extend(gx[m - 1:m+1, n-1:n+1].tolist())
                                    #trainsample.extend(gy[m - 1:m+1, n-1:n+1].tolist())
                                    trainsample.extend(np.concatenate(mag_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                                    trainsample.extend(np.concatenate(angle_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                                    trainsample.extend(np.concatenate(img_label_scaled[n,m].tolist(), axis=None))
                                    
                                    trainsamp = np.concatenate(trainsample, axis=None)
                                    #print(len(trainsamp))
                                    
                                    #print(trainsamp)
                                    #print(np.array(trainsamp))
                                    #print(np.array(trainsamp)[:-3])
                                    self.Xtrain.append(np.array(trainsamp)[:-3])
                                    self.Ytrain.append(np.mean(np.array(trainsamp)[-2:]))
                                    
                        
                                else:
                                    pass
                            else:
                                pass
                
                
                
                #cv2.imshow("Image",img)
                #cv2.waitKey(0)
            
            #cv2.destroyAllWindows()
        #print(self.Xtrain)
        #print(self.Ytrain)
        Xtra = np.array(self.Xtrain)
        Ytra = np.array(self.Ytrain)
        #print(type(Xtra))
        #print(Xtra)
        print('saving the train data...')
        np.savetxt('Xtrain.txt', Xtra)
        np.savetxt('Ytrain.txt', Ytra)
    
    def Replicate_Shuffled_Traindata_Epoch_Times(self):
        
        print('loading data.......')
        
        self.Xtrain = np.loadtxt('Xtrain.txt')
        self.Ytrain = np.loadtxt('Ytrain.txt')
        
        print(len(self.Xtrain))
        #print(self.Xtrain)
        #print(self.Ytrain)
        
        XYtrain = []
        
        for n in range(len(self.Xtrain)):
            XYtrain.append(np.ndarray.tolist(np.append(self.Xtrain[n], self.Ytrain[n])))
        
        
        #print(XYtrain)
        print(type(XYtrain))
        
        #XYtrain = np.array(XYtrain)
        # shuffle the training data
        print('Shuffle the train data and append n epochs...')
        
        XtrainSh = []
        YtrainSh = []
        for n in range(self.epochs):
            print('1')
            print(type(XYtrain))
            print(len(XYtrain))
            XYsh = random.sample(XYtrain, len(XYtrain))
            #print(XYsh)
            #print(type(XYsh))
            for m in range(len(XYsh)):
                XtrainSh.append(XYsh[m][:-1])
                YtrainSh.append([XYsh[m][-0]])
                
        self.XtrainSh = XtrainSh
        self.YtrainSh = YtrainSh
        print('Saving to text file...')
        np.savetxt('YtrainSh.txt', self.YtrainSh)
        np.savetxt('XtrainSh.txt', self.XtrainSh)

    
    
    def GenerateSinData(self):
    
        
        #N_Epochs = 3
        #BatchSize = 10
        print('Generating the train data..')
        # Generate the training data, sinus on -pi to pi

        Xtrain = np.linspace(- 4 * np.pi, 4 * np.pi , self.n_trainingsamples)

        print('Shuffle the train data..')
        # shuffle the training data
        XtrainSh = []
        YtrainSh = []
        for n in range(self.epochs):
            np.random.shuffle(Xtrain)
            X = Xtrain
            #Y = [x[1] for x in TrainData]
            for n in range(len(X)):
                XtrainSh.append([X[n], X[n]])
                YtrainSh.append(self.Graphnum * np.sin(X[n]) + self.Graphnum * np.sin(X[n]))
        
        self.XtrainSh = XtrainSh
        self.YtrainSh = YtrainSh
        
        #Generate Test Data
        print('Generate Test Data..')

        Xtest = []
        Xtestpre = np.linspace(- 4 * np.pi,  4 * np.pi , self.n_testsamples)
        for n in range(len(Xtestpre)):
            for m in range(len(Xtestpre)):
                Xtest.append([Xtestpre[n], Xtestpre[m]])
        #Ytrain = [np.sin(x) + np.sin(y) for x in Xtrain2dim[0] for y in Xtrain2dim[1]]
        Ytest = []
        for i in range(len(Xtest)):
            Ytest.append(self.Graphnum * np.sin(Xtest[i][0] + self.Graphnum * np.sin(Xtest[i][1])))

        #Ytest = [np.sin(x) + np.sin(y) for x in Xtest[0] for y in Xtest[1]]
        
        self.Xtest = Xtest
        self.Ytest = Ytest
        
        print('Len Xtest:', len(Xtest))
        #print(Xtest)
        print('Len Ytest:', len(Ytest))
        #print(Ytest)

    
    def TrainTheNetwork(self):
        print('loading data to train the network')
        self.XtrainSh = np.loadtxt('XtrainSh.txt')
        self.YtrainSh = np.loadtxt('YtrainSh.txt')
        print(len(self.XtrainSh)/self.epochs)
        print(len(self.XtrainSh[1]))
        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        ## define an accuracy assessment operation
        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # define mean error to supervise progress
        mean_error = tf.reduce_mean(abs(self.out))
        
        # start the session
        with tf.Session(config=tf.ConfigProto(log_device_placement=self.Logging_Activated)) as sess:
        # initialise the variables
            start = time.time()
            sess.run(init_op)
            
            # run the graph with full trace option to measure time of calculation
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            total_batch = int(len(self.XtrainSh) / self.batch_size)
            for epoch in range(self.epochs):
                avg_cost = 0
                for i in range(total_batch):
                    batch_x = []
                    batch_y = []
                    for n in range(self.batch_size):
                        batch_x.append(self.XtrainSh[i+n]) 
                        batch_y.append(self.YtrainSh[i+n]) 
                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)
                    
                    batch_x = np.reshape(batch_x, (self.batch_size,self.inputdim))
                    batch_y = np.reshape(batch_y, (self.batch_size,self.outputdim))
                                        
                    _ , c = sess.run([self.train_step, self.cost], 
                                feed_dict={self.x: batch_x, self.y: batch_y} , options=run_options, run_metadata=run_metadata)
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                
                #with open('timeline.json', 'w') as f:
                    #f.write(ctf)
            end = time.time()        
            #Xtest = np.array(self.Xtest)
            #Xtest = np.reshape(Xtest,(len(Xtest),2))
            #Ytest = np.array(self.Ytest)
            #Ytest = np.reshape(Ytest,(len(Ytest),1))
            #print(sess.run(mean_error, feed_dict={self.x: Xtest, self.y: Ytest}))
            #self.predicted_Sinus = sess.run(self.out, feed_dict={self.x: Xtest})
            
            self.Traintime = end - start
            
            # Save the variables to disk.
            if self.SaveOrNot == True:
                save_path = self.saver.save(sess, self.cwd + "/model/model.ckpt")
                print("Model saved in path: %s" % save_path)
            else:
                pass
    
    def Predict_On_Image(self, Image):
        
        print(Image)
        img = cv2.imread(Image, 1)
        print(img)
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        print('gx',len(gx), gx , 'gy', len(gy), gy)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        print('mag',len(mag) , mag ,'angle',len(angle), angle)
        edges = cv2.Canny(img,100,200)
        
        edges_scaled = edges/ 255
        mag_scaled = mag / 983.71234
        angle_scaled = angle / 359.87012
        img_scaled = img / 255
        
        self.Xtest = []
        
        lenim = len(img)
        for m in range(lenim):
            for n in range(lenim):
                
                if lenim - 2 >= n >= 1:
                    if lenim - 2 >= m >= 1:
                        trainsample = []
                        testsamp = []
                        
                        trainsample.extend(np.concatenate(img_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                        trainsample.extend(np.concatenate(edges_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                        #trainsample.extend(gx[m - 1:m+1, n-1:n+1].tolist())
                        #trainsample.extend(gy[m - 1:m+1, n-1:n+1].tolist())
                        trainsample.extend(np.concatenate(mag_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                        trainsample.extend(np.concatenate(angle_scaled[m - 1:m+1, n-1:n+1].tolist(), axis=None))
                        
                        testsamp = np.concatenate(trainsample, axis=None)
                        print(len(testsamp))
                        
                        
                        
                        
                        
                        self.Xtest.append(np.array(testsamp))
                        
                        
            
                    else:
                        pass
                else:
                    pass
                
        
        # finally setup the initialisation operator
        init_op = tf.global_variables_initializer()

        # define mean error to supervise progress
        mean_error = tf.reduce_mean(abs(self.out))
        
        # start the session
        with tf.Session(config=tf.ConfigProto(log_device_placement=self.Logging_Activated)) as sess:
        # initialise the variables
            
            sess.run(init_op)
            
            # run the graph with full trace option to measure time of calculation
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            
            self.saver.restore(sess, self.cwd + "/model/model.ckpt")
            Xtest = self.Xtest
            print(Xtest)
            print(len(Xtest))
            print(len(Xtest[1]))
            self.predicted_bin = sess.run(self.out, feed_dict={self.x: Xtest})
            
            print(self.predicted_bin)
            print(len(self.predicted_bin))
            
            predictedB = self.predicted_bin
            predictedB = np.array(predictedB)
            np.savetxt('predictedB.txt', predictedB)
    
    
    def Display_Prediction(self, predictedBinary, Image):
        
        self.predicted_bin = np.loadtxt(predictedBinary)
        
        img = cv2.imread(Image, 1)
        print(img)
        print(len(img))
        print(len(img[0]))
        print(256**2)
        print(255**2)
        
        predImage = []
        for pixel in self.predicted_bin:
            predImage.append([pixel,pixel,pixel])
        
        #print(predImage)
        print(type(predImage))
        print(len(predImage))
        print(len(predImage[0]))
        
        cv2.imshow("Image",predImage)
        cv2.waitKey(0)
            
            
        

    
    
    def GeneratePlotGrid(self):
        
        Xtest1 = []
        Xtest2 = []
        for n in range(len(self.Xtest)):
            Xtest1.append(self.Xtest[n][0])
            Xtest2.append(self.Xtest[n][1])
        
        self.Xtest1 = Xtest1
        self.Xtest2 = Xtest2
        
        meanX = self.Graphnum * np.sin(Xtest1) + self.Graphnum * np.sin(Xtest2)
        meanY = self.predicted_Sinus

        self.mean = sum(abs(meanX - np.transpose(meanY))[0])/len(meanX)

        
    
    
    def PlotSinNetworkOutput(self):
        # plot sinus and by model predicted values
        
        # Create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        
        # Plot the values
        ax.scatter(self.Xtest1, self.Xtest2, self.Graphnum * np.sin(self.Xtest1) + self.Graphnum * np.sin(self.Xtest2), s = 1, c = 'b', marker='o')
        ax.scatter(self.Xtest1, self.Xtest2, self.predicted_Sinus, s = 1, c = 'r', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        #print("time elapsed in training:", self.Traintime)

        plt.show()

