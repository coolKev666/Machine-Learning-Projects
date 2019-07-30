# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:56:33 2019

@author: David
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


############# SET GLOBAL VARIABLES for part 1 #############
N = 1000 # number of data points [Originally 1000]
dim = 784   # dimension of image
classes = 10    # number of classes present
alpha = 0.05    # alpha value to be used
gamma = 0.9     # gamma value to be used
epochs = 200


# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

#---------------------------------------------------------------------------#
#-----------------------------------Part-1----------------------------------#
#---------------------------------------------------------------------------#

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


    
def relu(x):
    # TODO
    return np.maximum(x,0)
    
#Need to modify this
def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
    
def softmax(x):
    # TODO
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

    
def computeLayer(W, X, b):
    # TODO
    mult = np.dot(X,W)
    return np.add(mult,b)
    
def CE(target, prediction):
    # TODO
    print("Calculating loss, please wait...")
    matrixOutput = target*np.log(prediction)
    ce = -1*np.sum(matrixOutput)/target.shape[0]
    return ce
    

def gradCE(target, prediction):
    # TODO
    return prediction - target


# Exercise 1.2 backpropagation
#Helper functions: 


def backOuterWeight(L1, CE_, N_):
    return L1.T.dot(CE_)/N_


# (1 x 10) shape matrix
def backOuterBias(CE_, N_):
    return (CE_)/N_  #np.ones((1,N_)).dot



# (F x K) shape matrix
def backHiddenWeight(X, L2, y, W1, S1, N_):
    return X.T.dot((gradCE(y, L2).dot(W1.T) * reluDerivative(S1)))/N_

# (1 x 10) shape matrix    
def backHiddenBias(y, L, W, S, N_):
    return np.ones((1,N_)).dot(gradCE(y, L).dot(W.T) * reluDerivative(S))/N_



def forwardpropagate(X, labels):
    #Forward propagation
    #W0, W1= Weight matrices for layer 1 and layer 2
    #x= Input data node vector
    #H= Hidden layer node values
    #O= Outer layer value

    #dim = 784 
    #classes= 10
    nio_1=[dim, N]
    nio_2=[N, classes]

    #Initialize weights with Xavier
    np.random.seed(1)

    #My weights:
    W0 = np.random.randn(dim,N)*np.sqrt(1/(nio_1[0]+nio_1[1]))             #First layer (hidden layer)
    W1 = np.random.randn(N,classes)*np.sqrt(1/(nio_2[0]+nio_2[1]))         #Second layer (output)

    #Other people's weights: 
    #np.random.randn(dim,N)*(2/(nio_1[0]+nio_1[1])) 
    #np.random.randn(N,classes)*(2/(nio_2[0]+nio_2[1]))   

    b0 = 0
    b1 = 0

    #Populate the hidden layer, expect shape (10000, 10000) matrix
    S1= computeLayer(W0,X,b0)
    L1= relu(S1)

    #Calculate output value based on H, expect shape (10000, 10) matrix
    S2= computeLayer(W1,L1,b1)
    L2= softmax(S2)

    #loss
    loss=CE(labels, L2)
    print("Forward prop complete ")
    return W0, W1, S1, S2, L1, L2, b0, b1, loss


    
def backpropagate(X_input,y):

    ######## LOADING DATA #########    
    
    X = X_input.reshape(X_input.shape[0],-1)
    
    v0_w = np.full((dim, N), 0.00001)          # (784, 10000)
    v1_w = np.full((N, classes), 0.00001)      # (10000, 10)
    v0_b = np.zeros((1, N))
    v1_b = np.zeros((1, 10))
    
    trueLabel = np.argmax(y, axis = 1)
    

    #Foward propagation to initialize variables:
    #We will continously update them during the training
    W0, W1, S1, S2, L1, L2, b0, b1,loss = forwardpropagate(X,y)
    
    CE_func = gradCE(y,L2) 
    #CEinner = gradCE(y,L1)
    
    
    loss = []
    accuracy = []
    #loop: 
    for i in range(epochs):
        
        dL_dWo = backOuterWeight(L1, CE_func, X.shape[0])
        dL_dWh = backHiddenWeight(X, L2, y, W1, S1, X.shape[0])
        dL_dBo = backOuterBias(CE_func, X.shape[0])
        dL_dBh = backHiddenBias(y,L2, W1, S1, X.shape[0])

        #Update momentum matrix:
        v0_w= gamma*v0_w + alpha* dL_dWh
        v1_w= gamma*v1_w + alpha* dL_dWo
        v0_b= gamma*v0_b + alpha* dL_dBh
        v1_b= gamma*v1_b + alpha* dL_dBo
        
        #Update weight matrix 
        W0= W0- v0_w
        W1= W1- v1_w
        b0= b0- v0_b
        b1= b1- v1_b

        #Update the layer 1 and layer 2 matrices: 
        S1= computeLayer(W0, X, b0)
        L1= relu(S1)
        S2= computeLayer(W1, L1, b1)
        L2= softmax(S2)

        #Loss
        #loss[i]=CE(y, L2)
        loss.append(CE(y,L2))
        predicted_label = np.argmax(L2, axis = 1)
        accuracy.append(np.mean(np.equal(trueLabel,predicted_label)))
        
        print("Iteration: " + str(i) + " Loss: "+ str(loss[i]))
    
    return loss, accuracy
    
    
'''
def accuracy(y_pred_list,trainTarget):
    
    Na=len(y_pred_list)
    accuracyListA = [None]*epochs
    accuracyListA[0]=0

    #RMSE calculations
    for i in range(1, epochs):
        RMSEA= np.sqrt(((y_pred_list[i]-trainTarget[i])**2))
        normalizedRMSEA= np.sum(RMSEA)/epochs
        if(normalizedRMSEA <3.2):
            accuracyListA[i]= normalizedRMSEA
        else: 
            accuracyListA[i]= 0
            
    
    return 
'''
    

### refresh
trainData, validData, testData, trainTarget, validTarget, testTarget= loadData()
newTrainTarget, newValidTarget, newTestTarget= convertOneHot(trainTarget, validTarget, testTarget) 
    
    
#PLOTTING ERROR and ACCURACY CURVE
trainLoss, trainAccuracy = backpropagate(trainData,newTrainTarget)
validLoss, validAccuracy = backpropagate(validData,newValidTarget)
testLoss, testAccuracy = backpropagate(testData,newTestTarget)

#trainAccuracy = accuracy(train_yPred,newTrainTarget)
#validAccuracy = accuracy(valid_yPred,newTrainTarget)
#testAccuracy = accuracy(test_yPred,newTrainTarget)


#print("testLoss: " + str(testLoss))
#print("testAccuracy: " + str(testAccuracy))


# FIRST PLOT LOSS
plt.grid()
plt.figure(1)
plt.plot(np.arange(0, epochs), trainLoss, marker='x', color='r', label="train")
plt.plot(np.arange(0, epochs), validLoss, marker='o', color='g', label="valid")
plt.plot(np.arange(0, epochs), testLoss, marker='*', color='b', label="test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Gradient Descent")
plt.legend(loc='upper right')
plt.show()



plt.figure(2)
plt.plot(np.arange(0, 200), trainAccuracy, marker='x', color='r',label="train")
plt.plot(np.arange(0, 200), validAccuracy, marker='o', color='g',label="valid")
plt.plot(np.arange(0, epochs), testAccuracy, marker='*', color='b',label="test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend(loc='right')
plt.show()


#-----------------------------Process-Data-Set------------------------------#
#Load data + Convert to one hot
trainData, validData, testData, trainTarget, validTarget, testTarget= loadData()
newTrainTarget, newValidTarget, newTestTarget= convertOneHot(trainTarget, validTarget, testTarget)

#Training Data: (Reshaped)
RS_trainData = trainData.reshape(trainData.shape[0],-1)
print("Training set (images) shape: {shape}".format(shape=RS_trainData.shape))        #(10000, 784)
print("Training set (labels) shape: {shape}".format(shape=newTrainTarget.shape))      #(10000, 10)

#Valid Data: (Reshaped)
RS_validData = validData.reshape(validData.shape[0],-1) 
print("Valid set (images) shape: {shape}".format(shape=RS_validData.shape))           #(6000, 784)
print("Valid set (labels) shape: {shape}".format(shape=newValidTarget.shape))         #(6000, 10)

#Test Set: (Reshaped)
RS_testData = testData.reshape(testData.shape[0],-1)
print("Test set (images) shape: {shape}".format(shape=RS_testData.shape))             #(2724, 784)
print("Test set (images) shape: {shape}".format(shape=newTestTarget.shape))           #(2724, 10)





'''

#---------------------------------------------------------------------------#
#-----------------------------------Part-2----------------------------------#
#---------------------------------------------------------------------------#


#------------------------------Useful-functions-----------------------------#
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

#-----------------------------Process-Data-Set------------------------------#
#Load data + Convert to one hot
trainData, validData, testData, trainTarget, validTarget, testTarget= loadData()
newTrainTarget, newValidTarget, newTestTarget= convertOneHot(trainTarget, validTarget, testTarget)

#Training Data: (Reshaped)
RS_trainData = trainData.reshape(trainData.shape[0],-1)
print("Training set (images) shape: {shape}".format(shape=RS_trainData.shape))        #(10000, 784)
print("Training set (labels) shape: {shape}".format(shape=newTrainTarget.shape))      #(10000, 10)

#Valid Data: (Reshaped)
RS_validData = validData.reshape(validData.shape[0],-1) 
print("Valid set (images) shape: {shape}".format(shape=RS_validData.shape))           #(6000, 784)
print("Valid set (labels) shape: {shape}".format(shape=newValidTarget.shape))         #(6000, 10)

#Test Set: (Reshaped)
RS_testData = testData.reshape(testData.shape[0],-1)
print("Test set (images) shape: {shape}".format(shape=RS_testData.shape))             #(2724, 784)
print("Test set (images) shape: {shape}".format(shape=newTestTarget.shape))           #(2724, 10)

#Create a dictionary with class labels: 
label_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
}


#------------------------------Display-Image-------------------------------#


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(RS_trainData[0], (28,28))
curr_lbl = np.argmax(newTrainTarget[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Train_Label: " + str(label_dict[curr_lbl]) + ")")

#Display the first image in valid data
plt.subplot(121)
curr_img = np.reshape(RS_validData[0], (28,28))
curr_lbl = np.argmax(newTrainTarget[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Valid_Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(RS_testData[0], (28,28))
curr_lbl = np.argmax(newTestTarget[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Test_Label: " + str(label_dict[curr_lbl]) + ")")
plt.show()

print("Image displayed")

#----------------------------Resizing before CNN-----------------------------#
# Reshape training and testing image, labels already have the correct dim: newTrainTarget, newValidTarget, newTestTarget
train_X = RS_trainData.reshape(-1, 28, 28, 1)
train_y= newTrainTarget
test_y= newTestTarget
test_X = RS_testData.reshape(-1,28,28,1)
print("Train shape: {shape}".format(shape=train_X.shape))    #(10000, 28, 28, 1)
print("Test shape: {shape}".format(shape=test_X.shape))    #(2724, 28, 28, 1)


#--------------------------Building Neural network----------------------------#
#Layer 1: 32-3x3 filters
#Layer 2: 64-3x3 filters
#Layer 3: 128-3x3 filters

#Hyperparameters: 
training_iters= 200
learning_rate= 0.001
batch_size= 128

#Network parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_classes= 10 # NMIST total classes (0-9 alphabets)

#Placeholders
x= tf.placeholder("float", [None, 28, 28, 1])
y= tf.placeholder("float", [None, n_classes])

#weights
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}

pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


#----------------------------------Training----------------------------------#
with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()

    '''
