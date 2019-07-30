import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import time
import sys

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')

[num_pts, dim] = np.shape(data)

is_valid = 1

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]  ## are we supposed to do this to get 1/3 data? *0.66667
  data = data[rnd_idx[valid_batch:]]

# value of K
K = 5

#sort points based on closest Cluster K
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO

    tempClust = {}
    for x in X:
      #  print (x-MU)
        dist = np.linalg.norm(x-MU, axis=1)
        index_ = np.where(dist == np.amin(dist))
        index = int(index_[0]) #which 
        try:
            tempClust[index].append(x)
        except KeyError:
            tempClust[index] = [x]
        
    return tempClust
    
# update value of mu
def update(mu, clusters):
    tempmu = []
    for k in sorted(clusters.keys()):
        tempmu.append(np.mean(np.array(clusters[k]), axis = 0))
    newmu_ = np.array(tempmu)
    return newmu_
    
# get optimized mu along with points corresponding
# BELOW source: https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/  
def get_clusters(X, K):
    # Initialize to K random centers
    tempmu = np.random.standard_normal(size=(K,dim))
    mu = np.random.standard_normal(size=(K,dim))
    
    while not np.array_equal(mu,tempmu):
        tempmu, clusters =  mu, distanceFunc(X, mu)
        mu = update(tempmu,clusters)
    return(mu, clusters)    
    
# loss for tf
def distanceFunc_tf(X, MU): 
    # return loss
    tempList = []
#    lossSum.reshape((N,k_))
    X_unpack = tf.unstack(X)
    for x in X_unpack:
        temp = tf.linalg.norm(x-MU, axis=1)
        #print (temp.shape)
        tempList.append(temp)
  #  lossSum.reshape((N,k_))
    #lossArray = np.array(tempList)
    loss_tf = tf.convert_to_tensor(tempList, np.float64)
    #sum_tf = 

    return tf.nn.l2_loss(tf.reduce_mean(tf.math.reduce_min(loss_tf, axis=1)))
    
# numpy loss
def distanceFunc_np(X, MU): 
    # return loss
    #tempList = []
    minSum = 0
    for x in X:
        temp = np.linalg.norm(x-MU, axis=1)
        minSum += np.amin(temp)
    
    return minSum / X.shape[0]
    #return tf.nn.l2_loss(tf.reduce_mean(tf.math.reduce_min(lossArray, axis=1)))
    

mu_, clusters = get_clusters(data,K)
print(mu_)
#print(mu)  

plt.figure(figsize=(10,15))
plt.style.use('ggplot')    
#clusters = np.array(list_cluster)
for i,k in enumerate(sorted(clusters.keys()),1):
    arr = np.array(clusters[k])
    l = len(clusters[k])
    p = l / val_data.shape[0]*100
    print("Cluster K = " + str(i) + "; Num elements: " + str(l) + "; percentage: " + str(p) + "%")
    plt.scatter(arr[:,0], arr[:,1],c=np.random.rand(3,1))
plt.scatter(mu_[:,0], mu_[:,1],marker='*',s=250,c='k')
plt.title('Scatter: K = ' + str(K))            
plt.legend(loc='upper right')        
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()  



#----------------------------------Training----------------------------------#
newGraph = tf.Graph()
tf.set_random_seed(421)

with newGraph.as_default():
    
    #___________Variable initialization including weight and bias tensors_______:
    #Reshape input data to be by 28*28= 784 dimensions
    RS_Data = data.reshape(data.shape[0],-1) #(3500, 784)
    RS_validData = val_data.reshape(val_data.shape[0],-1)

    #Define parameters used in the model
    learning_rate= 0.001
    
    
    alpha = 0.0005
    regularization = 0
    #iterations = 200
    batchSize = 20
    total_batches = int(len(RS_Data)/batchSize)
    #epochs = int(iterations/total_batches)
    epochs = 10

    #Initialize data array trains
    lossTrain = np.zeros(epochs+1)      
    lossValTrain = np.zeros(epochs+1)  
    accuracyTrain = np.zeros(epochs+1)     
    accuracyValTrain= np.zeros(epochs+1)

    #Bias Tensors
    learning_rate = tf.placeholder(tf.float32, name="learning-rate")  

    #loss = lossFunc(data,mu)

 # using Adam Optimizer
    B1, B2, epsilon= 0.9, 0.99, 1e-5
    loss_ = distanceFunc_np(data,mu_)
    #use_locking = False
    #loss = tf.convert_to_tensor(loss, np.float32)
    mu_ = np.random.standard_normal(size=(K,dim))
    mu = tf.Variable(mu_)
    mu = tf.Variable(t)
    print("Before optimizer")
    data_ = tf.convert_to_tensor(val_data, np.float64)
    loss = distanceFunc_tf(data_,mu)
    #print('Loss: ' + str(loss_))
    sys.exit()
  #  loss = getLoss(loss_tensor)
    print("After distance_func")
    optimizer_adam = tf.train.AdamOptimizer(0.001, B1, B2, epsilon).minimize(loss)   
    print("Passed optimizer")
    loss_adam = np.zeros(epochs+1)    
    learning_rates = [0.001]       
    for idx, i in enumerate(learning_rates):
        print('Learning Rate: {:2}'.format(i))        
        startTime = time.time()
        with tf.Session() as sess:
            #initializie the session and global variables
            sess.run(tf.global_variables_initializer())            
            #start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    
                            
            for j in range(epochs):                
                _, loss_value = sess.run([optimizer_adam,loss], feed_dict={learning_rate: i})
                                    
                duration = time.time() - startTime
                batchUnit = int(j)
                #if j % total_batches == 0:  
                print('Epoch: {:4}, Validation_Loss: {:5f}, Duration: {:2f}'. \
                      format(batchUnit, loss_value, duration))                        
                loss_adam[j] = loss_value
                
                 # test accuracy
                coord.request_stop()
                coord.join(threads)            
                    
     # Plots
    plt.figure(figsize=(10,15))
    plt.title('Adam loss')            
    plt.scatter(np.arange(epochs), loss_adam[1:epochs+1], marker='x', color='b', label = 'Adam')   
    plt.legend(loc='upper right')        
    plt.xlabel('Num. of Updates')
    plt.ylabel('Loss')
    plt.show()       

