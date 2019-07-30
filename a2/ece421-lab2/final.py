import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



############# SET GLOBAL VARIABLES for part 1 #############
N = 100 # number of data points [Originally 1000]
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
    return (CE_)/N_ 



# (F x K) shape matrix
def backHiddenWeight(X, L2, y, W1, S1, N_):
    return X.T.dot((L2-y).dot(W1.T) * reluDerivative(S1))/N_

# (1 x 10) shape matrix    s
def backHiddenBias(y, L, W, S, N_):
    return np.ones((1,N_)).dot((L-y).dot(W.T) * reluDerivative(S))/N_



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
plt.title("Gradient Descent_100")
plt.legend(loc='upper right')
plt.show()



plt.figure(2)
plt.plot(np.arange(0, 200), trainAccuracy, marker='x', color='r',label="train")
plt.plot(np.arange(0, 200), validAccuracy, marker='o', color='g',label="valid")
plt.plot(np.arange(0, epochs), testAccuracy, marker='*', color='b',label="test")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy_100")
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




#---------------------------------------------------------------------------#
#-----------------------------------Part-2----------------------------------#
#---------------------------------------------------------------------------#

#Special thanks to this tutorial: Reference: https://www.datacamp.com/community/tutorials/cnn-tensorflow-python


#------------------------------Useful-functions-----------------------------#
def conv2d(x, W, b):
    #Conv2d wrapper
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1]), b, padding='SAME')) 

def conv2d_normalize(x, W, b):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1]), b, padding='SAME')
    epsilon = 1e-3

    #Second layer (?, 14, 14, 64)
    batch_mean, batch_var = tf.nn.moments(x,[0])
    scale = tf.Variable(tf.ones([14, 14, 64]))
    beta = tf.Variable(tf.zeros([14, 14, 64]))

    #Batch normalized x
    BNX = tf.nn.batch_normalization(x,batch_mean,batch_var,beta,scale,epsilon)
    print("BNX dim " + str(BNX.shape))
    return tf.nn.relu(BNX) 

def maxpool2d(x):
    #Maxpool layer
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

def neuralNet(x, weights, biases):  
    #Layer 1 with maxpooling to 14*14 matrix
    L1 = conv2d(x, weights['w1'], biases['b1'])
    L1 = maxpool2d(L2)

    #Layer 2 with batch normalization and max pooling to 7*7 matrix
    L2 = conv2d_normalize(L1, weights['w2'], biases['b2'])  
    L2 = maxpool2d(L2)

    #Layer 3 with pooling to 4*4
    L3 = conv2d(L2, weights['w3'], biases['b3'])
    L3 = maxpool2d(L3)

    # Fully connected layer
    FC = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    FC = tf.add(tf.matmul(FC, weights['wd1']), biases['bd1'])
    FC = tf.nn.relu(FC)

    #Output
    return tf.add(tf.matmul(FC, weights['out']), biases['out'])


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


#----------------------------Resizing before CNN-----------------------------#
# Reshape training and testing image, labels already have the correct dim: newTrainTarget, newValidTarget, newTestTarget

train_X = RS_trainData.reshape(-1, 28, 28, 1) #(10000, 28, 28, 1)
train_y= newTrainTarget 
test_X = RS_testData.reshape(-1,28,28,1) #(2724, 28, 28, 1)
test_y= newTestTarget
#train_X= RS_validData.reshape(-1, 28, 28, 1)
#train_y= newValidTarget

#--------------------------Building Neural network----------------------------#
#Hyperparameters: 
training_iters= 50
learning_rate= 0.0001
batch_size= 32

#Network parameters
n_input = 28 #Image shape is 28 by 28
n_classes= 10 #10 Alphabets

#Placeholders
x= tf.placeholder("float", [None, 28, 28, 1])
y= tf.placeholder("float", [None, n_classes])

#Dictionary of weights and biases
weights = {'w1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 'w2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 'w3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),'b2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'b3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}

prob = tf.placeholder(tf.float32)
predicted = neuralNet(x, weights, biases)

#drop out layer
drop_out = tf.nn.dropout(predicted, prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimzer = tf.contrib.opt.AdamWOptimizer(0.01, learning_rate=learning_rate).minimize(cost) #Used for regularization 

#Check whether image is equal to the actual labelled image. and both will be a column vector.
compared_prediction = tf.equal(tf.argmax(predicted, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out. 
accuracy = tf.reduce_mean(tf.cast(compared_prediction, tf.float32))

#Initialize global variables
init = tf.global_variables_initializer()


#----------------------------------Training----------------------------------#
train_loss = []
test_loss = []
validation_loss=[]
train_accuracy = []
test_accuracy = []
validation_accuracy=[]

with tf.Session() as sess:
    sess.run(init) 
    length= len(train_X)/batch_size
    for i in range(training_iters):
        for batch in range(length):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            #Back propagation and accuracy calculation
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y, prob: 0.5})
            loss, _accuracy = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y, prob: 0.5})
        print("Iteration " + str(i) + " Loss " + \
                      "{:.4f}".format(loss) + " Training Accuracy " + \
                      "{:.4f}".format(acc))
        

        #Accuracy and Loss
        testAcc,validLoss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_accuracy.append(_accuracy)
        train_loss.append(loss)
        test_accuracy.append(testAcc)
        test_loss.append(validLoss)
        print("Testing Accuracy:","{:.5f}".format(testAcc))

#Plot 1: Training loss

'''
plt.figure(0)
plt.plot(range(training_iters), train_loss, '-b', label='loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper left')
plt.title("2.2_Training_Loss")
plt.savefig("2.2_Training_Loss.png")
'''
'''
#Plot 2: Training accuracy
plt.figure(1)
plt.plot(range(training_iters), train_accuracy, '-r', label='accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.title("2.3.2_Training_Accuracy")
plt.savefig("2.3.2_Training_Accuracy.png")
'''
'''
#Plot 3: Test loss
plt.figure(2)
plt.plot(range(training_iters), test_loss, '-b', label='loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper left')
plt.title("2.2_Test_Loss")
plt.savefig("2.2_Test_Loss.png")
'''
'''
#Plot 4: Test accuracy
plt.figure(3)
plt.plot(range(training_iters), test_accuracy, '-r', label='accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.title("2.3.2_Test_Accuracy")
plt.savefig("2.3.2_Test_Accuracy.png")
'''
'''
plt.figure(4)
plt.plot(range(training_iters), train_loss, '-b', label='loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc='upper left')
plt.title("2.2_Validation_Loss")
plt.savefig("2.2_Validation_Loss.png")
'''


#Plot 2: Validate accuracy
plt.figure(5)
plt.plot(range(training_iters), test_accuracy, '-r', label='accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.title("2.3.1_Validation_Accuracy")