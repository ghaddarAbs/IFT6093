'''
Created on Nov 15, 2017

@author: ghaddara
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings, gzip
import cPickle as pickle
import time, collections, json

warnings.filterwarnings("ignore")

np.random.seed(12345)

####################################################################
######## Activation and output functions and utils ################
####################################################################

def softmax(x, der=False):
    """Stable implmentation to compute softmax values for each sets of scores in x.
    Args:
    x of shape[batch_size, dim]    
    """
    
    if der:
        return softmax(x) * (1.0 - softmax(x))
    else:
        x_T = np.transpose(x)
        a_max = np.amax(x_T)
        expon = np.exp(x_T - a_max)
        out  = expon / np.sum(expon, axis=0)
        return np.transpose(out)



def relu(x, der=False):
    if der:
        return (x > 0).astype(np.int32)
    else:  
        return np.maximum(x,np.zeros(x.shape))


def rand_ini(dim_out, dim_in):
    """
    Xavier Glorot and Yoshua Bengio (2010):
           [Understanding the difficulty of training deep feedforward neural]
    """
    return np.random.uniform(-(3./(dim_in+dim_out))**.5 , (3./(dim_in+dim_out))**.5, (dim_out, dim_in) )

def one_hot(x, m):
    return np.eye(m, dtype=np.int32)[x]

####################################################################
##################### NeuralNetwork Class #########################
####################################################################

class NeuralNetwork:
    
    def __init__(self, config):
        
        self.d, self.d_h, self.m = config.n_input, config.n_hidden, config.n_output    
        
        
        self.W1 = rand_ini(self.d_h, self.d)
        self.b1 = np.zeros(self.d_h)
        
        self.W2 = rand_ini(self.m, self.d_h)
        self.b2 = np.zeros(self.m)

        
          

    
    
    
    ####################################################################
    ###################### Loop methodes ###############################
    ####################################################################

    def back_prob_loop(self, X, Y):
        
        self.grad_W1 = np.zeros(self.W1.shape)
        self.grad_b1 = np.zeros(self.b1.shape)
        self.grad_W2 = np.zeros(self.W2.shape)
        self.grad_b2 = np.zeros(self.b2.shape)
        
        batch_size = X.shape[0]
        
        cost, acc = [], .0
                
        for i in range(batch_size):
            ha = relu(X[i,:].dot(self.W1.T) + self.b1)
            oa = softmax(ha.dot(self.W2.T) + self.b2)            
            
            cost.append(-np.log(oa[Y[i]]))
            acc += (np.argmax(oa) == Y[i]).astype(np.int)
                        
            # Output Gradient
            
            grad_oa = oa - one_hot(Y[i], self.m).reshape((-1))
            grad_W2_mean = np.outer(grad_oa, ha)
           
            grad_b2_mean = grad_oa            
            grad_hs = self.W2.T.dot(grad_oa)
            
            # hidden Gradient
            grad_ha = grad_hs * relu(ha, True)
            grad_W1_mean = np.outer(grad_ha, X[i,:])
            grad_b1_mean = grad_ha

            # Accumulate Gradient
            self.grad_W1 += grad_W1_mean /batch_size
            self.grad_b1 += grad_b1_mean /batch_size
            self.grad_W2 += grad_W2_mean /batch_size           
            self.grad_b2 += grad_b2_mean /batch_size
        

        return cost, acc

    
    def grad_finite_diff_loop(self, X, Y, losses):
        '''
        This method should always be called after its vectorized version [grad_finite_diff] 
        '''
        # gradients
        self.grad_W1_diff_loop = np.zeros(self.W1.shape)
        self.grad_b1_diff_loop = np.zeros(self.b1.shape)
        self.grad_W2_diff_loop = np.zeros(self.W2.shape)
        self.grad_b2_diff_loop = np.zeros(self.b2.shape)
        batch_size = X.shape[0]

        for i in range(batch_size):
            self.grad_finite_diff(X[i].reshape((1,-1)), Y[i].reshape((1,-1)), losses[i])
            
            self.grad_W1_diff_loop += self.grad_W1_diff / batch_size
            self.grad_b1_diff_loop += self.grad_b1_diff / batch_size
            self.grad_W2_diff_loop += self.grad_W2_diff / batch_size
            self.grad_b2_diff_loop += self.grad_b2_diff / batch_size
    

    ####################################################################
    ###################### Matrix Methodes##############################
    ####################################################################
        
    def feed_forward(self, X, Y):
        """
        X.shape = [batch_size, n_input]
        Y.shape = [batch_size, n_output]
        
        It works for any batch size. For a single example the input shape is [1, n_input]
        """
        
        # Hidden layer 
        self.ha = relu(X.dot(self.W1.T) + self.b1)
        
        # Output layer
        self.oa = softmax(self.ha.dot(self.W2.T) + self.b2)
                
        loss = sum([-np.log(self.oa[i][Y[i]]) for i in range(Y.shape[0])])
        acc =  np.sum(np.argmax(self.oa, axis= 1) == Y)
        
        return  loss, acc 
               
    def back_prob(self, X, Y):
        
        batch_size = X.shape[0]
        # Gradient output
        grad_oa = self.oa - one_hot(Y, self.m)
        self.grad_b2 = np.mean(grad_oa, axis = 0)
        self.grad_W2 = np.dot(grad_oa.T, self.ha) / batch_size 
        
        # Gradient hidden 
        grad_hs = self.W2.T.dot(grad_oa.T) 
        grad_ha = grad_hs * relu(self.ha, True).T 
        self.grad_W1 = np.dot(grad_ha, X) / batch_size  
        self.grad_b1 = np.mean(grad_ha, axis =1) 

    

            
    def grad_finite_diff(self, X, Y, loss_1, eps=1e-5):

        # gradients
        self.grad_W1_diff = np.zeros(self.W1.shape)
        self.grad_b1_diff = np.zeros(self.b1.shape)
        self.grad_W2_diff = np.zeros(self.W2.shape)
        self.grad_b2_diff = np.zeros(self.b2.shape)
        batch_size = X.shape[0]
        
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                self.W1[i,j] += eps
                loss_2, _ = self.feed_forward(X, Y)
                self.grad_W1_diff[i, j] = (loss_2 - loss_1) / (eps * batch_size)
                self.W1[i,j] -= eps

        for i in range(self.b1.shape[0]):
            self.b1[i] += eps
            loss_2, _ = self.feed_forward(X, Y)            
            self.grad_b1_diff[i] = (loss_2 - loss_1) / (eps * batch_size)
            self.b1[i] -=  eps
        
               
        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                self.W2[i,j] += eps
                loss_2, _ = self.feed_forward(X, Y)
                self.grad_W2_diff[i, j] = (loss_2 - loss_1) / (eps * batch_size)
                self.W2[i,j] -= eps
        
        for i in range(self.b2.shape[0]):
            self.b2[i] += eps
            loss_2, _ = self.feed_forward(X, Y)
            self.grad_b2_diff[i] = (loss_2 - loss_1) / (eps * batch_size)
            self.b2[i] -=  eps
    
    
    def reset(self):
        self.W1 = rand_ini(self.d_h, self.d)
        self.b1 = np.zeros(self.d_h)
        
        self.W2 = rand_ini(self.m, self.d_h)
        self.b2 = np.zeros(self.m)
        
    def train(self, config, X, Y, X_dev, Y_dev, grad_meth="vectorized", verbose=0):
        maximum = 0.
        para = []
        
        for e in range(config.n_epoch):
            cost, acc = .0, .0
            for i in np.arange(0, X.shape[0], config.batch_size):
                input_data = X[i:i + config.batch_size] if config.batch_size > 1 else X[i,:].reshape((1, -1))
                output_data = Y[i:i + config.batch_size]
                
                if grad_meth == "loop":
                    avg_cost, avg_acc = self.back_prob_loop(input_data, output_data)
                    avg_cost = sum(avg_cost)
                else:
                    avg_cost, avg_acc = self.feed_forward(input_data, output_data)
                    self.back_prob(input_data, output_data)
                
                self.update_gradient(config)
                
                cost += avg_cost
                acc += avg_acc
                
            cost /= X.shape[0]
            acc /= X.shape[0] 
            
            # Validation Data
            
            _, acc_dev = self.feed_forward(X_dev, Y_dev)
            acc_dev /= X_dev.shape[0] * 1.
            
            if acc_dev > maximum:
                maximum = acc_dev
                para = [self.W1, self.b1, self.W2, self.b2]      
            
            if verbose == 1:
                print 'EPOCH: {:4d}\t\tCost Train: {:4f}\t\tAccuaracy Train: {:5f}\t\tAccuaracy Dev: {:5f}'\
                .format(e, cost, acc, acc_dev)
            
            
        self.W1, self.b1, self.W2, self.b2 = para
        _, acc_dev = self.feed_forward(X_dev, Y_dev)
        acc_dev /= X_dev.shape[0] * 1.
        if verbose == 1:
            print 'Best Accuaracy Dev: {:5f}'.format(acc_dev)
        
        return  acc_dev 
       
    def update_gradient(self, config):               
            
        l12_grad_W1 = config.lmd[0][0] * np.sign(self.W1) + 2 * config.lmd[0][1] * self.W1
        l12_grad_W2 = config.lmd[1][0] * np.sign(self.W2) + 2 * config.lmd[1][1] * self.W2
                
        self.W1 = self.W1 - config.lr * (self.grad_W1 + l12_grad_W1)
        self.b1 = self.b1 - config.lr * self.grad_b1
        
        self.W2 = self.W2 - config.lr * (self.grad_W2 + l12_grad_W2)
        self.b2 = self.b2 - config.lr * self.grad_b2

             
####################################################################
####################### Grid View  #################################
####################################################################

class problems:

    def __init__(self, n_points=50):
        config = MoonConfig()
        data = np.loadtxt(open(config.data_path,'r'))
        np.random.shuffle(data)
        self.data_train, self.data_test, self.target_train, self.target_test = \
                 train_test_split(data[:,:-1], data[:,-1].astype(np.int8), test_size=0.3, random_state=123)        
        
        self.n_points = n_points
              
        #grid_data
        self.X = np.linspace(np.min(self.data_train[:,0]), np.max(self.data_train[:,0]), n_points)
        self.Y = np.linspace(np.min(self.data_train[:,1]), np.max(self.data_train[:,1]), n_points)
        self.grid_data = np.transpose([np.tile(self.X, len(self.Y)), np.repeat(self.Y, len(self.X))])
        
        config = MNISTConfig()
        
        with gzip.open(config.data_path, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)  
               
        self.mnist_train_data, self.mnist_train_target = train_set
        self.mnist_valid_data, self.mnist_valid_target = valid_set
        self.mnist_test_data, self.mnist_test_target = test_set
        
    def problem_one_and_two(self):
        '''
        1. As a beginning, start with an implementation that computes the gradients
        for a single example, and check that the gradient is correct using the finite
        difference method described above.
        '''
        
        input_data, output_data = self.data_train[0].reshape((1, -1)), self.target_test[0].reshape((1, -1))
        config = MoonConfig()
        model = NeuralNetwork (config)
        
        
        
        losses, _ = model.back_prob_loop(input_data, output_data)
        
        # We verify  the gradient using the diff finite method (iterative version)
        model.grad_finite_diff_loop(input_data, output_data, losses)
        
        # The ratio are between 1.01 and 0.99, and NaN represent zero
        #gradiant using both methods        
        print "W1 Diff", model.grad_W1_diff_loop / model.grad_W1, "\n"
        print "b1 Diff", model.grad_b1_diff_loop / model.grad_b1, "\n"
            
        print "W2 Diff", model.grad_W2_diff_loop / model.grad_W2, "\n"
        print "b2 Diff", model.grad_b2_diff_loop / model.grad_b2, "\n"
        
        
        print "W1 finite = ", "\n" ,model.grad_W1_diff_loop, "\n"
        print "W1  normal = ", "\n", model.grad_W1, "\n"
         
        print "The ratio are between 1.01 and 0.99, and NaN represent zero"+ \
              "gradiant using both methods."
            
    
    def problem_three(self):       
        '''
        Add a hyperparameter for the minibatch size K to allow compute the
        gradients on a minibatch of K examples (in a matrix), by looping over
        the K examples (this is a small addition to your previous code).
        ''' 
        
        # The method train use a mini batch implementation
        # Here a demo with our default batch_size
        config = MoonConfig()
        model = NeuralNetwork (config)
        model.train(config, self.data_train, self.target_train, self.data_test,\
                     self.target_test, grad_meth="loop", verbose=1)
    
    def problem_four(self):       
        '''
        Display the gradients for both methods (direct computation and finite
        difference) for a small network (e.g. d = 2 and dh = 2) with random
        weights and for a minibatch with 10 examples (you can use examples from
        both classes from the dataset 2 moons).
        ''' 
        
        # The method train use a mini batch implementation
        # Here a demo with our default batch_size (100)
        config = MoonConfig()
        config.batch_size = 10
        config.n_hidden = 2
        
        input_data, output_data = self.data_train[:10,:], self.target_test[:10]
        config = MoonConfig()
        model = NeuralNetwork (config)
        losses, _ = model.back_prob_loop(input_data, output_data)
        model.grad_finite_diff_loop(input_data, output_data, losses)
        
    
    def problem_five(self):
        '''
        Train your neural network using gradient descent on the dataset of the
        two moons. Plot the decision regions for several different values of the
        hyperparameters (weight decay, number of hidden units, early stopping)
        so as to illustrate their effect on the capacity of the model.
        '''
        
        def plot_data(config, title= ""):
            _, ax = plt.subplots(1)                
    
            model = NeuralNetwork (config)
            acc_dev = model.train(config, self.data_train, self.target_train, self.data_test, self.target_test)
            print "Best performance on Valid=", acc_dev
            model.feed_forward(self.grid_data, np.zeros((self.grid_data.shape[0]), dtype=int))
            pred_grid_data = np.argmax(model.oa, axis=1).reshape((self.n_points, self.n_points))
            
            ax.scatter(self.grid_data[:,0], self.grid_data[:,1], c = pred_grid_data,\
                        marker = 'o', s=150, label='grid_data', alpha=.3)
            ax.scatter(self.data_train[:,0], self.data_train[:,1], c = self.target_train,\
                        marker = 'v', s=150, label='train_data')        
            ax.scatter(self.data_test[:,0], self.data_test[:,1], c = self.target_test,\
                        marker = 's', s=150, label='test_data')
            
            plt.legend()
            plt.title(title)
            plt.show()
            
            
        config = MoonConfig()
        print "Good config: n_hidden=", config.n_hidden, "n_epochs=", config.n_epoch, "weight decay=", config.lmd[0,0]
        plot_data(config, title= "Appropriate Hyper Parameters") 
        print "\n=======================\n"
         
        config = MoonConfig()
        config.n_epoch = 10
        print "Small iteration number: n_epochs=", config.n_epoch, "\n"
        plot_data(config, title= "Small iteration number")
        print "\n=======================\n"
         
         
        config = MoonConfig()
        config.n_hidden = 2
        print "Low capacity (small number of hidden units): n_hidden=", config.n_hidden, "\n"
        plot_data(config, title= "Small Hidden units ") 
        print "\n=======================\n"
         
        config = MoonConfig()
        config.lmd.fill (0.01)
        print "High Lambdas: weight decay=", config.lmd[0,0], "\n"
        plot_data(config, title= "High Lambdas ") 
        print "\n=======================\n"
         
        config = MoonConfig()
        config.lr = .5
        print "High Learning rate=", config.lr, "\n"
        plot_data(config, title= "High Learning rate ") 
        print "\n=======================\n"
        
        config = MoonConfig()
        config.batch_size = 300
        print "High Batch Size=", config.batch_size, "\n"
        plot_data(config, title= "High Batch Size") 
        print "\n=======================\n"
        
    #TODO
    def problem_six(self):
        '''
        As a second step, copy your existing implementation to modify it to a new
        implementation that will use matrix calculus (instead of a loop) on batches
        of size K to improve efficiency. Take the matrix expressions in numpy
        derived in the first part, and adapt them for a minibatch of size
        K. Show in your report what you have modified (precise the
        former and new expressions with the shapes of each matrices).
        '''
        
        print "======================================================"
        print "feed_forward() use a matrix implementation, shapes are:"
        print "======================================================\n\n"
        
        print "Shapes of: self.ha = relu(X.dot(self.W1.T) + self.b1)"
        print "====================================================="
        print "X.shape=[batch_size, input_dim]"
        print "W1.shape=[hidden_dim, input_dim]"
        print "b1.shape=[hidden_dim,]"
        print "X.dot(self.W1.T) + self.b1 -> shape=[batch_size, hidden_dim]"
        print "relu(X.dot(self.W1.T) + self.b1) -> shape=[batch_size, hidden_dim]\n\n"
        
        print "Shapes of: self.oa = softmax(self.ha.dot(self.W2.T) + self.b2)"
        print "=============================================================="
        print "self.ha.shape=[batch_size, hidden_dim]"
        print "W2.shape=[n_ouput, hidden_dim]"
        print "b1.shape=[n_ouput]"
        print "self.ha.dot(self.W2.T) + self.b2 -> shape=[batch_size, n_ouput]"
        print "softmax(X.dot(self.W1.T) + self.b1) -> shape=[batch_size, n_ouput]\n\n"
        
        
        print "======================================================================\n"
        
        print "==================================================="
        print "back_prob() use a matrix implementation, shapes are:"
        print "==================================================\n\n"
        
        print "Shapes of:  grad_oa = self.oa - one_hot(Y, self.m)"
        print "=================================================="
        print "self.oa.shape=[batch_size, n_ouput]"
        print "self.Y.shape=[batch_size]"
        print "one_hot(Y, self.m).shape=[batch_size, n_ouput]"
        print "grad_oa.shape=[batch_size, n_ouput]\n\n"
        
        print "Shapes of:  self.grad_b2 = np.mean(grad_oa, axis = 0)"
        print "====================================================="
        print "self.grad_b2.shape=[n_ouput]\n\n"
        
        print "Shapes of:  self.grad_W2 = np.dot(grad_oa.T, self.ha) / batch_size "
        print "==================================================================="
        print "self.grad_W2.shape=[n_ouput, n_hidden] (other dim are listed above)\n\n"
        
        print "Shapes of:  grad_hs = self.W2.T.dot(grad_oa.T)"
        print "=============================================="
        print "grad_hs.shape=[n_hidden, batch_size] (other dim are listed above)\n\n"
        
        print "Shapes of:  grad_ha = grad_hs * relu(self.ha, True).T"
        print "====================================================="
        print "relu(self.ha, True).shape=[batch_size, hidden_dim]"
        print "grad_ha.shape=[n_hidden, batch_size] (other dim are listed above)\n\n"
        
        print "Shapes of:  self.grad_W1 = np.dot(grad_ha, X) / batch_size"
        print "=========================================================="
        print "self.grad_W1.shape=[n_hidden, n_input] (other dim are listed above)\n\n"
        
        print "Shapes of:  self.grad_b1 = np.mean(grad_ha, axis =1)"
        print "===================================================="
        print "self.grad_b1.shape=[n_hidden] (other dim are listed above)\n\n"
       
    def problem_seven(self):
        '''
        Compare  both  implementations  (with  a  loop  and  with  matrix  calculus)
        to  check  that  they  both  give  the  same  values  for  the  gradients  on  the
        parameters, first for K= 1, then for K= 10. Display the gradients for both methods.
        '''
        config = MoonConfig()
        config.n_hidden = 2
        for k in [1, 10]:
            config.batch_size = k
            model = NeuralNetwork (config)
            W1, W2 = model.W1, model.W2
            # loop model
            
            model.back_prob_loop(self.data_train[:k,], self.target_train[:k])
            grad_W1_loop, grad_b1_loop, grad_W2_loop, grad_b2_loop = \
            model.grad_W1, model.grad_b1, model.grad_W2, model.grad_b2
            
            
            # matrix model
            model = NeuralNetwork (config)
            model.W1, model.W2 = W1, W2 
            model.feed_forward(self.data_train[:k,], self.target_train[:k])
            model.back_prob(self.data_train[:k,], self.target_train[:k])
            
            grad_W1_matrix, grad_b1_matrix, grad_W2_matrix, grad_b2_matrix = \
            model.grad_W1, model.grad_b1, model.grad_W2, model.grad_b2

            
            print "For batch size=", k
            print "==================="
            print "W1 Diff", grad_W1_loop / grad_W1_matrix, "\n"
            print "b1 Diff", grad_b1_loop / grad_b1_matrix, "\n"
              
            print "W2 Diff", grad_W2_loop / grad_W2_matrix, "\n"
            print "b2 Diff", grad_b2_loop / grad_b2_matrix, "\n"
            
    def problem_eight(self):
        '''
        Time  how  long  takes  an  epoch  on  MNIST  (1  epoch  =  1  full  traversal
        through the whole training set) for K= 100 for both versions (loop over
        a minibatch and matrix calculus)
        '''
        start_time = time.time()
        config = MNISTConfig()
        config.batch_size = 100
        config.n_epoch  = 7
        config.n_hidden = 20
        
        model = NeuralNetwork (config)
        model.train(config, self.mnist_train_data, self.mnist_train_target, self.mnist_valid_data,\
                     self.mnist_valid_target, grad_meth="loop")
        print"Loop version"
        print "--- %s seconds ---" % (time.time() - start_time)

        start_time = time.time()
        model = NeuralNetwork (config)
        model.train(config, self.mnist_train_data, self.mnist_train_target, self.mnist_valid_data, \
                    self.mnist_valid_target)
        print"Matrix version"
        print "--- %s seconds ---" % (time.time() - start_time)
        
    def problem_nine(self):
        '''Adapt your code to compute the error (proportion of misclassified examples)
        on the training set as well as the total loss on the training set during each
        epoch of the training procedure, and at the end of each epoch, it computes
        the error and average loss on the validation set and the test set. Display
        the 6 corresponding figures (error and average loss on train/valid/test),
        and write them in a log file.
        '''
        
        config = MNISTConfig()
        model = NeuralNetwork (config)
        X, Y = self.mnist_train_data, self.mnist_train_target
        X_dev, Y_dev = self.mnist_valid_data, self.mnist_valid_target
        X_test, Y_test = self.mnist_test_data, self.mnist_test_target
        
        dico_loss= collections.defaultdict(list)
        for e in range(config.n_epoch):
            cost, acc = .0, .0
            for i in np.arange(0, X.shape[0], config.batch_size):
                input_data = X[i:i + config.batch_size] if config.batch_size > 1 else X[i,:].reshape((1, -1))
                output_data = Y[i:i + config.batch_size]
                avg_cost, avg_acc = model.feed_forward(input_data, output_data)
                model.back_prob(input_data, output_data)
                model.update_gradient(config)
                
                cost += avg_cost
                acc += avg_acc
    
            cost /= X.shape[0]
            acc /= X.shape[0] 
            dico_loss['cost_train'].append(cost)
            dico_loss['acc_train'].append(acc)
            
            # Validation Data
            cost_dev, acc_dev = model.feed_forward(X_dev, Y_dev)
            cost_dev /= X_dev.shape[0] * 1.
            acc_dev /= X_dev.shape[0] * 1.
            dico_loss['cost_dev'].append(cost_dev)
            dico_loss['acc_dev'].append(acc_dev)
            
            # Test Data
            cost_test, acc_test = model.feed_forward(X_test, Y_test)
            cost_test /= X_test.shape[0] * 1.
            acc_test /= X_test.shape[0] * 1.
            dico_loss['cost_test'].append(cost_test)
            dico_loss['acc_test'].append(acc_test)
            print 'EPOCH: {:4d}\t\tCost Train: {:4f}\t\tAccuaracy Train: {:5f}\t\tAccuaracy Dev:'+\
                    '{:5f}\t\tAccuaracy test: {:5f}'\
            .format(e, cost, acc, acc_dev, acc_test)

        with open("dico_loss.json", 'w') as fp:
            json.dump(dico_loss, fp)
        
    def problem_ten(self):   
        '''
        Train your network on the MNIST dataset. Plot the training/valid/test
        curves (error and loss as a function of the epoch number, corresponding
        to what you wrote in a file in the last question). Include in your report
        the curves obtained using your best hyperparameters, i.e. for which you
        obtained your best error on the validation set. We suggest 2 plots : the
        first one will plot the error rate (train/valid/test with different colors,
        precise which color in a legend) and the other one for the averaged loss
        (on train/valid/test). You should be able to get less than 5% test error.
        Indicate the values of your best hyperparameters corresponding to the
        curves. Bonus points are given for a test error of less that 2%.
        '''
        
        print "With the hyper-parameters: n_hidden= 256 lr= 0.25 n_epoch=50 batch_size= 50," + \
            " the model is able to get less than 2% error."
        
        with open("dico_loss.json") as data_file:    
            dico_loss = json.load(data_file)
        
        print "Best Dev Performance= ", np.amax(dico_loss['acc_dev']) * 100 , "%"  
        print "Best Test Performance according to best valid performance= ", dico_loss['acc_test']\
        [np.argmax(dico_loss['acc_dev'])] * 100 , "%"
               
        
        for key in dico_loss:
            dico_loss[key] = np.asarray(dico_loss[key])  

        epoch = np.arange(dico_loss['cost_train'].shape[0])
        plt.plot(epoch, (1- dico_loss['acc_train']) * 100,label='train')
        plt.plot(epoch, (1- dico_loss['acc_dev']) * 100, label='dev')
        plt.plot(epoch, (1- dico_loss['acc_test']) * 100,label='test')    
        
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('% of Error')
        plt.title('Classification Errors')
        plt.show()
        
        plt.plot(epoch, dico_loss['cost_train'],label='train')
        plt.plot(epoch, dico_loss['cost_dev'], label='dev')
        plt.plot(epoch, dico_loss['cost_test'],label='test')    
        
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('Cost')
        plt.title('cost')
        plt.show()
        
##############################################################################
class MoonConfig(object):
    """2Moon config."""
    data_path = '2moons.txt'
 
    
    #regularization [[l1_w1, l2_w1], [l1_w2, l2_w2]]
    lmd = np.zeros((2,2))
    lmd.fill (0.00001)
    n_input = 2
    n_hidden = 20
    n_output = 2
    n_epoch = 50
    batch_size = 5
    lr = 0.1             

class MNISTConfig(object):
    """MNIST config."""
    data_path = 'mnist.pkl.gz' 
    
    #regularization [[l1_w1, l2_w1], [l1_w2, l2_w2]]
    lmd = np.zeros((2,2))
    lmd.fill (0)
    n_input = 784
    n_hidden = 256
    n_output = 10
    n_epoch = 50
    batch_size = 50
    lr = 0.25             
    
def main():
    ###############################################
    ######## Abbas Ghaddar #######################
    ###############################################
    
    #################################################################
    # Important Notes:
    # 0- The code use Python 2.7
    # 1- The class problems contains methods that will
    #    run the code according to the enumeration in the
    #    question sheet.
    # 2- 
    #    a- one and two are merged in a single method
    #    b- six is the shapes of matrix of ff and bp methods
    #    c- I already run nine and save the output in dico_loss.json
    #       problem ten will auto load the .json and plot the figs
    #################################################################
    
    prb = problems()
    
    print "Problem 1 and 2"
    prb.problem_one_and_two()
    
    print "Problem 3"
    prb.problem_three()
    
    print "Problem 4"
    prb.problem_four()
    
    print "Problem 5"
    prb.problem_five()
    
    print "Problem 6"
    prb.problem_six()
    
    print "Problem 7"
    prb.problem_seven()
    
    print "Problem 8"
    prb.problem_eight()
    
    print "Problem 9"
#     prb.problem_nine ()
    
    print "Problem 10"
    prb.problem_ten()

if __name__ == "__main__":
    main()
            
