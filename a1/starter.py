import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import time

#_____________________________________________________________________________
#This file includes MSE, gradMSE, CE, gradCE, and gradient descent functions
#In both tensorflow and numpy format. 
#Go to "NUMPY_IMPLEMENTATION" section for numpy implementations
#Go to "TENSORFLOW_IMPLEMENTATION" section for tensorflow implementations

#The tests in the main function are all commented out. 
#You may uncomment to examine a working plot fo the code.
#_____________________________________________________________________________

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget



#_____________________________________________________________________________
#__________________________TENSORFLOW_IMPLEMENTATION__________________________
#_____________________________________________________________________________
#MSE implementation
def MSE(W, b, x, y, reg):
    return tf.reduce_mean((tf.matmul(x, W)+b- y)**2)

#WeightDecay implementation
def weightDecay(W, b, x, y, reg):
    return reg * tf.nn.l2_loss(W)

#Loss function: 
def Loss(W, b, x, y, reg):
    #return MSE(W, b, x, y, reg)
    return MSE(W, b, x, y, reg) + weightDecay(W, b, x, y, reg)

#gradient MSE: 
def grad_MSE(W, b, x, y, reg):
    #Calculate the partial derivatives: 
    gradWeight= tf.multiply(x, (tf.matmul(x, W)+b)-y) + tf.multiply(reg, tf.reduce_sum(W)) #Weight: -x(y-(Wx+b))
    gradBias= (tf.matmul(x, W)+b)-y #Bias: -(y-(Wx+b))
    return gradWeight, gradBias

#TENSORFLOW gradient descent: (batch)
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol):
    return tf.train.GradientDescentOptimizer(alpha).minimize(Loss(W, b, x, y, reg))

#stochastic gradient descent
def SD_grad_descent(W, b, x, y, alpha, reg):
    return W- alpha*matmul(((tf.matmul(x, W)+b)-y), x)

#normal equation: 
def normal_equation(X, y):
    XT = tf.transpose(X)
    return tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)




#_____________________________________________________________________________
#______________________________NUMPY_IMPLEMENTATION___________________________
#_____________________________________________________________________________
#NUMPY MSE implementation
def NP_MSE(W, b, x, y, reg):
    N= len(x)
    y_pred = x.dot(W) + np.ones(N)*b
    y_initial = (y.T)
    diff = y_pred-y_initial[0]
    diff = diff.dot(diff)
    return diff/N

#NUMPY weight decay: 
def NP_weightDecay(W, b, x, y, reg): 
    return (W.dot(W)*reg)/2

#Loss function: 
def NP_Loss(W, b, x, y, reg):
    return NP_MSE(W, b, x, y, reg) + NP_weightDecay(W, b, x, y, reg)

#NUMPY of gradient of MSE:
def NP_grad_MSE(W, b, x, y, reg):
    N= len(x)
    y_pred = x.dot(W) + b
    gradientWeight = (((x.T).dot((y_pred - y.T).T).T /N) + reg * W)[0]
    gradB = (x.dot(W) - y.T).sum()/ N + b
    return gradientWeight, gradB

#normal equation: 
def NP_normal_equation(X, y):
    XT = X.T
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(XT, X)), XT), y)
    return theta

#Numpy Cross Entropy 
def crossEntropyLoss_NADAM(W, b, x, y, reg):
    WX = tf.matmul(x,W)
    sigma_input = WX + b
    y_hat = 1/(1+tf.exp(-1*sigma_input))
    avg = tf.reduce_mean((-y)*tf.log(y_hat)-(1-y)*tf.log(1-y_hat))
    return avg + ((reg/2)*tf.reduce_sum(tf.square(W))**2)

#NUMPY cross entropy loss
def crossEntropyLoss(W, b, x, y, reg):
    N = x.shape[0]
    x.reshape(N,784)
    W.reshape(784,1)
    sigma_input = np.dot(x,W)+b
    y_hat = 1/(1+np.exp(-1*sigma_input))  #3500 lisst
     
    #part1 = (1/N)*(np.subtract((-y)*np.log(y_hat),(1-y)*np.log(1-y_hat)))
    parta = (-1*y)*np.log(y_hat)
    print ("partA value is : ", np.linalg.norm(parta))
    partb = (1 - y)*np.log(y_hat*np.exp(-1*sigma_input))
    print ("partB value is : ", np.linalg.norm(partb))
    partc = parta - partb     
    part1 = (1/N)*partc
    part2 = ((reg/2)*np.linalg.norm(W)**2)
    result = np.linalg.norm(part1) + part2
    return (result)

#gradient of CE:
def gradCE(W, b, x, y, reg):
    # Your implementation here
    N = x.shape[0]
    x.reshape(N,784)
    w_gradient.reshape(784,1)
    WX = np.dot(x,W) # use w_average for weight gradient , W should have 784 entries x 1
    sigma_input = -1*(WX + b)
    y_hat = 1/(1+np.exp(sigma_input))
    part1 = 1/N*(-y*np.exp(sigma_input)-(1-y))
    
    w_gradient = y_hat*part1
    w_gradient = np.matmul(x,w_gradient)
    #w_gradient.reshape(784,1)
    w_gradient = np.add(w_gradient, reg*W)
    
    b_gradient = np.dot(y_hat,part1)
    return w_gradient, b_gradient

#Numpy CE gradient descent
def CE_grad_descent(W, b, x, y, alpha, iterations, reg, EPS):
    #Initialize new weights: 
    grad_W, grad_B = gradCE(W, b, x, y, reg)
    newWeight, newBias = W-alpha*grad_W, b-alpha*grad_B
    W, b= newWeight, newBias
    error=0  #EPS error comparison

    #Initialize lists
    lossList=[]
    for i in range(iterations):
        grad_W, grad_B= gradCE(W, b, x, y, reg)
        newWeight, newBias= W-alpha*grad_W, b-alpha*grad_B
        W, b= W-alpha*grad_W, b-(grad_B*alpha)

        #Append the y value and error
        y_pred_List.append(x.dot(W)+b)
        lossList.append(crossEntropyLoss(W, b, x, y, reg))
        
        #error computation: 
        if(np.linalg.norm(grad_W*alpha)<EPS):
            break
    
    return W, b, lossList


#NUMPY gradient descent: (batch)
def np_grad_descent(W, b, x, y, alpha, iterations, reg, EPS):
    #Initialize new weights: 
    grad_W, grad_B = NP_grad_MSE(W, b, x, y, reg)
    newWeight, newBias = W-alpha*grad_W, b-alpha*grad_B
    W, b= newWeight, newBias
    error=0  #EPS error comparison

    #Initialize lists
    lossList, y_pred_List=[], []

    for i in range(iterations):
        grad_W, grad_B= NP_grad_MSE(W, b, x, y, reg)
        newWeight, newBias= W-alpha*grad_W, b-alpha*grad_B
        W, b= W-alpha*grad_W, b-(gy
        y_pred_List.append(x.dot(W)+b)
        lossList.append(NP_MSE(W, b, x, y, reg))
        
        #error computation: 
        if(np.linalg.norm(grad_W*alpha)<EPS):
            break
    
    return W, b, lossList, y_pred_List




#For ADAM optimization: 
def buildGraph(lossType=None, isAdam= False):
    trainData, validData, testData, trainTarget, validTarget, testTarget= loadData()
    newGraph = tf.Graph()
    tf.set_random_seed(421)

    with newGraph.as_default():  
        #___________Variable initialization including weight and bias tensors_______:
        #Reshape input data to be by 28*28= 784 dimensions
        RS_trainData = trainData.reshape(trainData.shape[0],-1) #(3500, 784)
        RS_validData = validData.reshape(validData.shape[0],-1)
        RS_testData =  testData.reshape(testData.shape[0],-1)

        #Define parameters used in the model
        alpha = 0.001
        regularization = 0
        iterations = 3500
        batchSize = 1750
        total_batches = int(len(RS_trainData)/batchSize)
        epochs = int(iterations/total_batches)

        #Initialize data array trains
        lossTrain = np.zeros(epochs+1)      
        lossValTrain = np.zeros(epochs+1)  
        accuracyTrain = np.zeros(epochs+1)     
        accuracyValTrain= np.zeros(epochs+1)

        #Tensors to hold training inputs and variables
        XTrainIn = tf.constant(RS_trainData, dtype=tf.float32)        
        yTrainIn = tf.constant(trainTarget, dtype=tf.float32)
        X, y = tf.train.slice_input_producer([XTrainIn, yTrainIn], num_epochs=None)          
        X_batch, y_batch = tf.train.batch([X, y], batch_size=batchSize)
                
        #Tensors for validation and test
        X_val = tf.constant(RS_validData, dtype=tf.float32)        
        y_val = tf.constant(validTarget, dtype=tf.float32)
        X_test = tf.constant(RS_testData, dtype=tf.float32)        
        y_test = tf.constant(testTarget, dtype=tf.float32)

        #Weight and Bias Tensors
        W = tf.Variable(tf.truncated_normal([28*28, 1], 0.0, 0.5, dtype=tf.float32, seed=None, name="w"))      
        b = tf.Variable(tf.truncated_normal([1], 0.0, 0.5, dtype=tf.float32, seed=None, name="b"))  
        y_pred = tf.matmul(X_batch, W)+b
        learning_rate = tf.placeholder(tf.float32, name="learning-rate")  

        #MSE case analysis
        if lossType == "MSE":
            loss = Loss(W, b, X_batch, y_batch, 0)
            optimizer = grad_descent(W, b, X_batch, y_batch, learning_rate, epochs, 0, 0.00001)
        #CE case analysis
        elif lossType == "CE":
            loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_batch, logits=y_pred)) \
                + 0.5*regularization * tf.nn.l2_loss(W)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        
        #__________________________________Not_Adam__________________________________:
        startTime = time.time()
        with tf.Session() as sess:
            #initializie the session and global variables
            sess.run([
                tf.local_variables_initializer(),
                tf.global_variables_initializer(),
            ])
            #Queue for batch training
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

            
            
            #__________________________________SGD__________________________________:        
            for j in range(iterations):
                #Optimizer
                __, loss_value = sess.run([optimizer,loss], feed_dict={learning_rate: alpha})
                    
                duration = time.time() - startTime
                batchUnit= int(j/total_batches)

                #Calculate the losses and accuracies for integer epochs in batch    
                if j% total_batches == 0:                       
                    #Print 
                    print('Epoch No: {:4}, Loss: {:3f}, Duration: {:2f}'. \
                        format(batchUnit, loss_value, duration))        

                    #Shuffling training set
                    #X_batch = tf.random.shuffle(X_batch)         

                    #calculate validation predict and predicted values
                    y_pred = tf.matmul(X_batch, W) + b 
                    y_val_pred = tf.matmul(X_val, W) + b

                    #Training loss and accuracy
                    lossTrain[batchUnit] = loss_value
                    accuracyTensor= tf.equal(tf.round(tf.sigmoid(y_pred)), y_batch)
                    accuracyTrain[batchUnit] = sess.run(tf.count_nonzero(accuracyTensor))

                    #Validation loss and accuracy
                    lossValTrain[batchUnit] = Loss(W, b, X_val, y_val, 0).eval()
                    validationTensor= tf.equal(tf.round(tf.sigmoid(y_val_pred)), y_val)
                    accuracyValTrain[batchUnit] = sess.run(tf.count_nonzero(validationTensor))
                                              
            coord.request_stop()
            coord.join(threads)
                        
        # Plots
      
        plt.figure(figsize=(10,15))
        plt.title('Losses')      
        plt.scatter(np.arange(epochs), lossTrain[1:epochs+1], marker='o', color='r', label = 'train')
        plt.scatter(np.arange(epochs), lossValTrain[1:epochs+1], marker='x', color='g', label = 'validation')
        plt.legend(loc='upper right')        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
                
        plt.figure(figsize=(10,15))
        plt.title('Accuracies')
        plt.scatter(np.arange(epochs), accuracyTrain[1:epochs+1]/batchSize, marker='*', color='r', label = 'training')
        plt.scatter(np.arange(epochs), accuracyValTrain[1:epochs+1]/100, marker='.', color='b', label = 'validation')
        plt.legend(loc='upper right')        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.show()
        
    
        #__________________________________is_Adam__________________________________:
        if isAdam: 
            # using Adam Optimizer
            B1, B2, epsilon= 0.99, 0.99, 1e-04
            use_locking = False
            optimizer_adam = tf.train.AdamOptimizer(learning_rate, B1, B2, epsilon, use_locking, 'Adam').minimize(loss)   
            loss_adam = np.zeros(epochs+1)           

            #Define parameters used in the model
            alpha = 0.001
            regularization = 0
            iterations = 1000
            batchSize = 500
            total_batches = int(len(RS_trainData)/batchSize)
            epochs = int(iterations/total_batches)

            accuracyTrain = np.zeros(epochs+1)     
            accuracyValTrain= np.zeros(epochs+1)

      
            startTime = time.time()
            with tf.Session() as sess:
                #initializie the session and global variables
                sess.run(tf.global_variables_initializer())            
                #start input enqueue threads.
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    
                            
                for j in range(iterations):                
                    _, loss_value = sess.run([optimizer_adam,loss], feed_dict={learning_rate: 0.001})
                                    
                    duration = time.time() - startTime
                    batchUnit = int(j/total_batches)


                    if j % total_batches == 0:  
                        y_pred = tf.matmul(X_batch, W) + b 
                        y_val_pred = tf.matmul(X_val, W) + b
                        print('Epoch: {:4}, Loss: {:5f}, Duration: {:2f}'. \
                            format(batchUnit, loss_value, duration)) 

                        #Training accuracy value:             
                        accuracyTensor= tf.equal(tf.round(tf.sigmoid(y_pred)), y_batch)
                        accuracyTrain[batchUnit] = sess.run(tf.count_nonzero(accuracyTensor))
                        accuracyTrain[batchUnit] = accuracyTrain[batchUnit]/5

                        #Validation accuracy value: 
                        validationTensor= tf.equal(tf.round(tf.sigmoid(y_val_pred)), y_val)
                        accuracyValTrain[batchUnit] = sess.run(tf.count_nonzero(validationTensor))

                    
                    

                #Test accuracy    
                y_test_pred = tf.matmul(X_test, W) + b
                test_acc = tf.count_nonzero(tf.equal(tf.round(tf.sigmoid(y_test_pred)), y_test))
                print("the test accuracy is: ")
                print(sess.run(test_acc))
                
                #close queue
                coord.request_stop()
                coord.join(threads)            
                    
            #Plots
            plt.figure(figsize=(10,15))
            plt.title('SGD v.s. Adam')            
            plt.scatter(np.arange(epochs), accuracyTrain[1:epochs+1], marker='o', color='g', label = 'SGD')
            plt.scatter(np.arange(epochs), accuracyValTrain[1:epochs+1], marker='x', color='b', label = 'Adam')   
            plt.legend(loc='upper right')        
            plt.xlabel('Epoch')
            plt.ylabel('Accuraacy')
            plt.show()


#Initialize Test data: 
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
#Initialize Params
size = len(trainData[0])**2
W, b = np.ones(size)/100, 0
    
#Reshape trainData to pass into functions
TrainingX = np.zeros((len(trainData), size))
for i, j in enumerate(trainData):
    TrainingX[i] = np.reshape(j, size)

#Validation data
validX = np.zeros((len(validData), size))
for i, j in enumerate(validData):
    validX[i] = np.reshape(j, size)

#Test data
testX = np.zeros((len(testData), size))
for i, j in enumerate(testData):
    testX[i] = np.reshape(j, size)

#_____________________________________________________________________________
#________________________________Params_4_Tests_______________________________
#_____________________________________________________________________________
alphas=[0.001, 0.005, 0.0001]
epochs = 5000
reg = [0.001, 0.1, 0.5]
EPS = 1e-7


#_____________________________________________________________________________
#______________________________Learning_Rate_Test_____________________________
#_____________________________________________________________________________
'''
#Training: 
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, TrainingX, trainTarget, alphas[0], epochs, 0, EPS)
W_new, b_new, errorListB, yResultsB=  np_grad_descent(W, b, TrainingX, trainTarget, alphas[1], epochs, 0, EPS)
W_new, b_new, errorListC, yResultsC=  np_grad_descent(W, b, TrainingX, trainTarget, alphas[2], epochs, 0, EPS)


#Error Plot
plt.figure(figsize=(10,15))
plt.grid()
plt.figure(1)
plt.scatter(np.arange(0, epochs), errorListA, marker='o', color='r',)
plt.scatter(np.arange(0, epochs), errorListB, marker='o', color='g',)
plt.scatter(np.arange(0, epochs), errorListC, marker='o', color='b',)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Gradient Descent")
plt.show()
'''
'''
#Validation: 
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, validX, validTarget, alphas[0], epochs, 0, EPS)
W_new, b_new, errorListB, yResultsB=  np_grad_descent(W, b, validX, validTarget, alphas[1], epochs, 0, EPS)
W_new, b_new, errorListC, yResultsC=  np_grad_descent(W, b, validX, validTarget, alphas[2], epochs, 0, EPS)


#Error Plot
plt.figure(figsize=(10,10))
plt.grid()
plt.figure(2)
plt.scatter(np.arange(0, epochs), errorListA, marker='o', color='r',)
plt.scatter(np.arange(0, epochs), errorListB, marker='o', color='g',)
plt.scatter(np.arange(0, epochs), errorListC, marker='o', color='b',)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Gradient Descent")
plt.show()
'''

'''
#Test Data: 
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, testX, testTarget, alphas[0], epochs, 0, EPS)
W_new, b_new, errorListB, yResultsB=  np_grad_descent(W, b, testX, testTarget, alphas[1], epochs, 0, EPS)
W_new, b_new, errorListC, yResultsC=  np_grad_descent(W, b, testX, testTarget, alphas[2], epochs, 0, EPS)


#Error Plot
plt.figure(figsize=(10,10))
plt.grid()
plt.figure(2)
plt.scatter(np.arange(0, epochs), errorListA, marker='o', color='r',)
plt.scatter(np.arange(0, epochs), errorListB, marker='o', color='g',)
plt.scatter(np.arange(0, epochs), errorListC, marker='o', color='b',)
plt.xlabel("Epochs")
plt.ylabel("Test Loss")
plt.title("Gradient Descent")
plt.show()

'''


#_____________________________________________________________________________
#__________________________________Reg_Param_Test_____________________________
#_____________________________________________________________________________
#Training: 
'''
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, TrainingX, trainTarget, 0.005, epochs, reg[0], EPS)
W_new, b_new, errorListB, yResultsB=  np_grad_descent(W, b, TrainingX, trainTarget, 0.005, epochs, reg[1], EPS)
W_new, b_new, errorListC, yResultsC=  np_grad_descent(W, b, TrainingX, trainTarget, 0.005, epochs, reg[2], EPS)

#Error Plot
plt.figure(figsize=(10,15))
plt.grid()
plt.figure(1)
plt.scatter(np.arange(0, epochs), errorListA, marker='x', color='r',)
plt.scatter(np.arange(0, epochs), errorListB, marker='x', color='g',)
plt.scatter(np.arange(0, epochs), errorListC, marker='x', color='b',)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Gradient Descent")
plt.show()
'''

'''
#Validation: 
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, validX, validTarget, 0.005, epochs, reg[0], EPS)
W_new, b_new, errorListB, yResultsB=  np_grad_descent(W, b, validX, validTarget, 0.005, epochs, reg[1], EPS)
W_new, b_new, errorListC, yResultsC=  np_grad_descent(W, b, validX, validTarget, 0.005, epochs, reg[2], EPS)


#Error Plot
plt.figure(figsize=(10,10))
plt.grid()
plt.figure(2)
plt.scatter(np.arange(0, epochs), errorListA, marker='o', color='r',)
plt.scatter(np.arange(0, epochs), errorListB, marker='o', color='g',)
plt.scatter(np.arange(0, epochs), errorListC, marker='o', color='b',)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Gradient Descent")
plt.show()
'''

'''
#Test Data: 
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, testX, testTarget, 0.005, epochs, reg[0], EPS)
W_new, b_new, errorListB, yResultsB=  np_grad_descent(W, b, testX, testTarget, 0.005, epochs, reg[1], EPS)
W_new, b_new, errorListC, yResultsC=  np_grad_descent(W, b, testX, testTarget, 0.005, epochs, reg[2], EPS)


#Error Plot
plt.figure(figsize=(10,10))
plt.grid()
plt.figure(2)
plt.scatter(np.arange(0, epochs), errorListA, marker='o', color='r',)
plt.scatter(np.arange(0, epochs), errorListB, marker='o', color='g',)
plt.scatter(np.arange(0, epochs), errorListC, marker='o', color='b',)
plt.xlabel("Epochs")
plt.ylabel("Test Loss")
plt.title("Gradient Descent")
plt.show()
'''


#_____________________________________________________________________________
#_____________________________Normal_Equation_Test____________________________
#_____________________________________________________________________________
'''
#Normal Equation:
X, y= TrainingX, trainTarget
startTime = time.time()
theta = NP_normal_equation(X, y)
elapsedTime = time.time() - startTime
print("The optimal theta value is: ", theta)
print("The elapsed time is: ", elapsedTime)



#Gradient descent: 

startTime = time.time()
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, TrainingX, trainTarget, 0.005, epochs, reg[0], EPS)
elapsedTime = time.time() - startTime
print("The elapsed time is: ", elapsedTime)


plt.figure(figsize=(10,10))
plt.grid()
plt.figure(2)
plt.scatter(np.arange(0, len(theta)), theta, marker='o', color='g', label = "theta")
plt.scatter(np.arange(0, len(theta)), W_new, marker='o', color='r', label = "w")
plt.xlabel("Epochs")
plt.ylabel("Weight")
plt.legend(loc='upper right')
plt.title("Gradient Descent")
plt.show()

'''
#_____________________________________________________________________________
#__________________________________Build_Graph________________________________
#_____________________________________________________________________________


#Calling the build graph function

buildGraph("MSE", False)




#_____________________________________________________________________________
#_______________________________Accuracy_Graph________________________________
#_____________________________________________________________________________
'''


#Plotting Accuracy
EPS=0.000001
W_new, b_new, errorListA, yResultsA=  np_grad_descent(W, b, TrainingX, trainTarget, 0.0001, epochs, 0, EPS)
#W_new, b_new, errorListB, yResultsB=  np_grad_descent(W, b, TrainingX, trainTarget, 0.001, epochs, 0, EPS)
#W_new, b_new, errorListC, yResultsC=  np_grad_descent(W, b, TrainingX, trainTarget, 0.0001, epochs, 0, EPS)

Na=len(yResultsA)
accuracyListA = [None]*Na
accuracyListA[0]=0

#RMSE calculations
for i in range(1, 3000):
    RMSEA= np.sqrt(((yResultsA[i]-trainTarget[i])**2))
    normalizedRMSEA= np.sum(RMSEA)/3000
    if(normalizedRMSEA <3.2):
        accuracyListA[i]= normalizedRMSEA
    else: 
        accuracyListA[i]= 0

plt.grid()
plt.figure(2)
plt.scatter(np.arange(0, Na), accuracyListA, marker='x', color='b',)
#plt.scatter(np.arange(0, N), accuracyListB, marker='x', color='g',)
#plt.scatter(np.arange(0, N), accuracyListC, marker='x', color='b',)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy for 3 Learning Rate Values")
plt.show()
'''


