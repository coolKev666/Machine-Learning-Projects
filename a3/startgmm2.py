import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp


# For Validation set
is_valid=0

if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

#------------------------------------------------------------------------
#--------------------------Code for part 2.2.2---------------------------
#------------------------------------------------------------------------

K=5
alpha=0.1

#Distance function for GMM
def distanceFunc(X, MU):
    return tf.reduce_sum(tf.square(tf.expand_dims(X, 1) - MU), 2)
    
#Gaussian PDF
def Gaussian_PDF(X,mu,sigma):
    Dims = tf.shape(X)[1]
    pi = tf.convert_to_tensor(3.14,tf.float64)
    distance = distanceFunc(X,mu)
    gaussLog = tf.cast(-Dims,tf.float64)/2.*tf.log(2.*pi*sigma**2) + tf.div(-distance,2.*tf.square(sigma))
    return gaussLog, tf.arg_max(gaussLog, 1)

#Joint prob
def Joint_prob(X,mu,prior,sigma):
    gaussLog, ass = Gaussian_PDF(X,mu,sigma)
    posterior = gaussLog+prior
    return posterior, ass

#----------------------------------------------------------------------
#---------------------------Helper functions---------------------------
#----------------------------------------------------------------------

def reduce_logsumexp(input_tensor, reduction_indices=1, keep_dims=False):
    max_input_tensor1 = tf.reduce_max(input_tensor, reduction_indices, keep_dims=keep_dims)
    max_input_tensor2 = max_input_tensor1
    if not keep_dims:
      max_input_tensor2 = tf.expand_dims(max_input_tensor2, 
                                       reduction_indices) 
    return tf.log(tf.reduce_sum(tf.exp(input_tensor - max_input_tensor2), 
                                reduction_indices, keep_dims=keep_dims)) + max_input_tensor1

def logsoftmax(input_tensor):
  return input_tensor - reduce_logsumexp(input_tensor, keep_dims=True)


#----------------------------------------------------------------------
#-------------------------------Training-------------------------------
#----------------------------------------------------------------------

data = tf.placeholder(tf.float64, [None,dimD])   
randomNormalVar1= tf.Variable(tf.random_normal([K, dimD]))
avg =  tf.cast(randomNormalVar1,tf.float64)
randomNormalVar2= tf.Variable(tf.random_normal([K,]))
sigma = tf.cast(randomNormalVar2,tf.float64)
sigExponent = tf.exp(sigma)
prior = tf.cast(tf.Variable(tf.random_normal([1,K])),tf.float64)
softMaxPrior = logsoftmax(prior)
    
#Lost variables
log_post, assignment = Joint_prob(data,avg,softMaxPrior,sigExponent)
loss =-1* tf.reduce_sum (reduce_logsumexp(log_post,keep_dims = True))

#Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=alpha,beta1=0.99,beta2=0.999,epsilon=1e-5)
train = optimizer.minimize(loss)
train,data,avg,loss, assignment,sigExponent,softMaxPrior
    

error_train = []

#Initialize and Train
init = tf.initialize_all_variables()
sess.run(init)
for i in range(1000):
    _, error, center, assignSize,si,pri= sess.run([train, loss, avg, assignment,sigExponent,softMaxPrior], feed_dict={data:trainData})
    error_train.append(error/10000)
    print('iteration:',i,'loss:',error/10000)
    
color = np.int32(assignSize)
x, y = input_data[:, 0], input_data[:, 1]

'''
plt.figure(figsize=(10,10))
plt.grid()
plt.figure(2)
plt.scatter(np.arange(0, len(error_train)), error_train, marker='o', color='r',)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
'''

#Scatter plots
plt.title('Scatter: K = ' + str(K))
plt.style.use('ggplot')
plt.scatter(x,y,c=color)
plt.grid()
plt.show()



    
