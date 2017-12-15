'''
Created on Oct 7, 2017

@author: Abbas Ghaddar
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class gaussian_diagonal:
    def __init__(self, data_train, data_test, labels_train, labels_test, labels):
        self.data_train, self.data_test, self.labels_train, self.labels_test, \
                self.labels = np.transpose(data_train), np.transpose(data_test),\
                 labels_train, labels_test, labels
        
        self.n= self.data_train.shape[1]
        self.d= self.data_train.shape[0]
        self.n_test= self.data_test.shape[1]
    
    def train(self):
        self.mu= np.transpose(np.matrix(np.mean(self.data_train, axis=1)))
        sub = self.data_train - self.mu
        self.sigma= np.multiply(np.dot(sub, np.transpose(sub)) / self.n,  np.identity(self.d)) 
        self.mu = self.mu.reshape((-1))
        
        
    def compute_predictions(self):
        det = np.linalg.det(self.sigma)
        inv = np.linalg.inv(self.sigma)
        
        part1 = - .5 * ( self.d * np.log(2 * np.pi) + np.log(det))                     
        part2 = -.5 * np.asarray([ np.asscalar(np.dot(np.dot((self.data_test[:,i] - self.mu), inv), \
                                                      np.transpose(self.data_test[:,i] - self.mu)))  \
                                                      for i in range(self.n_test)] )
        
        self.pred = part1 + part2
    
    #  For implementation proposes......same as compute_predictions() but it take the train
    # data as a parameter to compute predictions on it    
    def compute_predictions_train(self, data_train):
        det = np.linalg.det(self.sigma)
        inv = np.linalg.inv(self.sigma)
        data_train = np.transpose(data_train)
        part1 = - .5 * ( self.d * np.log(2 * np.pi) + np.log(det))                   
        part2 = -.5 * np.asarray([ np.asscalar(np.dot(np.dot((data_train[:,i] - self.mu), inv), \
                                                      np.transpose(data_train[:,i] - self.mu)))  \
                                                      for i in range(data_train.shape[1])] )
        self.pred_train = part1 + part2
            
class parzen_guassian_isotropique:
    def __init__(self, data_train, data_test, labels_train, labels_test, labels, sigma):
        self.data_train, self.data_test, self.labels_train, self.labels_test, self.labels = \
                np.transpose(data_train), np.transpose(data_test), labels_train, labels_test, labels
        
        self.n= self.data_train.shape[1]
        self.d= self.data_train.shape[0]
        self.n_test= self.data_test.shape[1]
        self.sigma= sigma
    
    def train(self):
        self.x=self.data_train
        
    def compute_predictions(self):
        self.pred= np.zeros((self.n_test))
        part1 = 1.0/(( (2 * np.pi) ** (self.d/2.0))* (self.sigma ** self.d) )
        
        for i in range(self.n_test):
            for j in range(self.n):
                sub = self.data_test[:,i] - self.x[:,j]
                self.pred[i] += part1 * np.exp(-sub.dot(sub)/ (2 * (self.sigma ** 2)))
            self.pred[i] /= self.n
            
        self.pred = np.log(self.pred)
    
    # For implementation proposes......same as compute_predictions() but it take the train data as a 
    # parameter to compute predictions on it
    def compute_predictions_train(self, data_train):
        data_train= np.transpose(data_train)
        n_train = data_train.shape[1]
        self.pred_train= np.zeros((n_train))
        
        part1 = 1.0/(( (2 * np.pi) ** (self.d/2.0))* (self.sigma ** self.d) )
        
        for i in range(n_train):
            for j in range(self.n):
                sub = data_train[:,i] - self.x[:,j]
                self.pred_train[i] += part1 * np.exp(-np.sqrt(sub.dot(sub))/ (2 * (self.sigma ** 2)))
            self.pred_train[i] /= self.n

        self.pred_train = np.log(self.pred_train)
        
class density_1:
    def __init__(self, cls, dim, n_points ):
        data = load_iris()
        data, target, _ = data['data'], data['target'], data['target_names']
        self.data_train = np.asarray([data[i] for i in range(target.shape[0]) \
                                      if target[i] == cls])[:,dim].reshape((-1, 1))
        self.n_points = n_points
        self.data_test = np.linspace(np.min(self.data_train), np.max(self.data_train), n_points).reshape((-1, 1))
    
    def plot_data(self):
        _, ax = plt.subplots(1)
        ax.set_ylim(ymin=-.02)
        data_x_plot = self.data_train.reshape(-1)
        data_test_plot = self.data_test.reshape(-1)
        
                
        # part (a)
        ax.plot(data_x_plot,np.zeros_like(data_x_plot),'x')
        
        # part (b)
        guassian_model=gaussian_diagonal(self.data_train, self.data_test, None, None, None)
        guassian_model.train()
        guassian_model.compute_predictions()
        ax.plot(data_test_plot,np.exp(guassian_model.pred),'-',label='Guassian')
        
        # part (c)
        parzen_model_sigma_small=parzen_guassian_isotropique(self.data_train, self.data_test, None, None, None, 0.01)
        parzen_model_sigma_small.train()
        parzen_model_sigma_small.compute_predictions()
        ax.plot(data_test_plot,np.exp(parzen_model_sigma_small.pred),'--',label='sigma small')
        
        # part (d)
        parzen_model_sigma_big=parzen_guassian_isotropique(self.data_train, self.data_test, None, None, None, 5)
        parzen_model_sigma_big.train()
        parzen_model_sigma_big.compute_predictions()
        ax.plot(data_test_plot,np.exp(parzen_model_sigma_big.pred),'-.',label='sigma big')
        
        # part (e)
        parzen_model_sigma_right=parzen_guassian_isotropique(self.data_train, self.data_test, None, None, None, 0.3)
        parzen_model_sigma_right.train()
        parzen_model_sigma_right.compute_predictions()
        ax.plot(data_test_plot,np.exp(parzen_model_sigma_right.pred),':',label='sigma appropriate')
        
        
        plt.title("3. Density Estimator (3)")       
        plt.legend()
        plt.show()

class density_2:
    # dim should be < 3
    def __init__(self, cls, dim, n_points):
        data = load_iris()
        data, target, _ = data['data'], data['target'], data['target_names']
        self.data_train = np.asarray([data[i] for i in range(target.shape[0]) if target[i] == cls])[:,dim:dim+2]
        self.n_points = n_points
               
        #grid_data
        self.X = np.linspace(np.min(self.data_train[:,0]), np.max(self.data_train[:,0]), n_points)
        self.Y = np.linspace(np.min(self.data_train[:,1]), np.max(self.data_train[:,1]), n_points)
        self.data_test = np.transpose([np.tile(self.X, len(self.Y)), np.repeat(self.Y, len(self.X))])
                
        
    def plot_data(self, part):
        _, ax = plt.subplots(1)                
        #plot point
        ax.scatter(self.data_train[:,0], self.data_train[:,1], s=50, c=u'b')      
        
        
        if part == 'a':
            guassian_model=gaussian_diagonal(self.data_train, self.data_test, None, None, None)
            guassian_model.train()
            guassian_model.compute_predictions()
            Z1= guassian_model.pred.reshape((self.n_points, self.n_points))
            
            ax.contour(self.X,self.Y,Z1, 30)
            plt.title("3. Density Estimator (4.a) - Guassian")
            plt.show()    
         

        if part == 'b':
            parzen_model_sigma_small=parzen_guassian_isotropique(self.data_train, self.data_test,\
                                                                  None, None, None, 0.01)
            parzen_model_sigma_small.train()
            parzen_model_sigma_small.compute_predictions()
            Z2= parzen_model_sigma_small.pred.reshape((self.n_points, self.n_points))
            ax.contour(self.X,self.Y,Z2, 10)
            plt.title("3. Density Estimator (4.b)- Parzen small sigma")
            plt.show()
         
        if part == 'c':
            parzen_model_sigma_big=parzen_guassian_isotropique(self.data_train, self.data_test, None, None, None, 5)
            parzen_model_sigma_big.train()
            parzen_model_sigma_big.compute_predictions()
            Z3= parzen_model_sigma_big.pred.reshape((self.n_points, self.n_points))
            ax.contour(self.X,self.Y,Z3, 30)
            plt.title("3. Density Estimator (4.c)- Parzen big sigma")
            plt.show()
        
        if part == 'd':
            parzen_model_sigma_right=parzen_guassian_isotropique(self.data_train, self.data_test,\
                                                                  None, None, None, 0.3)
            parzen_model_sigma_right.train()
            parzen_model_sigma_right.compute_predictions()
            Z4= parzen_model_sigma_right.pred.reshape((self.n_points, self.n_points))
            ax.contour(self.X,self.Y,Z4, 30)
            plt.title("3. Density Estimator (4.d)- Parzen appropriate sigma")
            plt.show()
        
class bayes_classifier:
    def __init__(self, n_points=50):
        data = load_iris()
        self.data, self.target, self.labels = data['data'], data['target'], data['target_names']
        self.data_train, self.data_test_d4, self.labels_train, self.labels_test = \
                train_test_split(self.data, self.target, test_size=0.20, random_state=123)
        
        self.n_train = self.data_train.shape[0]
        self.n_test = self.data_test_d4.shape[0]
        self.n_class = len(self.labels)
        self.class_count = np.asarray([np.count_nonzero(self.labels_train == i) for i in range(self.n_class)])
        self.class_prior= (1. * self.class_count / self.n_train)
        
        tmp = [[], [], []]
        for i in range(self.n_train):
            idx= self.labels_train[i]
            tmp[idx].append(self.data_train[i])
        
        self.data_train_d4= [np.asarray(tmp[i]) for i in range(self.n_class)]
        self.data_train_d2= [self.data_train_d4[i][:,:2] for i in range(self.n_class)]
        self.data_test_d2= self.data_test_d4[:,:2]
        
        
        #print grid_data
        (min_x1,max_x1) = (min(self.data[:,0]),max(self.data[:,0]))
        (min_x2,max_x2) = (min(self.data[:,1]),max(self.data[:,1]))
        xgrid = np.linspace(min_x1,max_x1,num=n_points)
        ygrid = np.linspace(min_x2,max_x2,num=n_points)
        self.grid_data= np.transpose([np.tile(xgrid, len(ygrid)), np.repeat(ygrid, len(xgrid))]) 
        
    def train(self, dim=2, model='gaussian', sigma=0.3):
        """
         Parameters
        ----------
        dim= 2 or 4
        model='gaussian' or 'parzen'
        sigma= h parameter for parzen only
        """
        
        data_train= self.data_train_d2 if dim == 2 else self.data_train_d4
        data_test= self.data_test_d2 if dim == 2 else self.data_test_d4
        
        
        self.models= [gaussian_diagonal(data_train[i], data_test, \
                            None, None, None) for i in range(self.n_class)] if model == 'gaussian' else \
                     [parzen_guassian_isotropique(data_train[i], data_test, None, None, None, sigma) \
                      for i in range(self.n_class)]
            
        [self.models[i].train() for i in range(self.n_class)]   
        self.dim= dim
    def compute_predictions(self, run_grid= False):
        data_train= self.data_train[:,:2] if self.dim == 2 else self.data_train
        
        # pred on test
        [self.models[i].compute_predictions() for i in range(self.n_class)]
        self.pred_test = np.argmax(np.asarray([ np.exp(self.models[i].pred) * self.class_prior[i] \
                                                for i in range(self.n_class)]).transpose(), axis=1)
        self.err_test= 100 - (np.sum(np.equal(self.pred_test, self.labels_test)) * 100. / self.n_test)
        
        # pred on train
        [self.models[i].compute_predictions_train(data_train) for i in range(self.n_class)]
        self.pred_train = np.argmax(np.asarray([ np.exp(self.models[i].pred_train) * self.class_prior[i] \
                                                 for i in range(self.n_class)]).transpose(), axis=1)
        self.err_train= 100 - (np.sum(np.equal(self.pred_train, self.labels_train)) * 100. / self.n_train)
        
        
        #pred data_grid
        if self.dim == 2 and run_grid:
            [self.models[i].compute_predictions_train(self.grid_data) for i in range(self.n_class)]
            self.pred_grid_data = np.argmax(np.asarray([ np.exp(self.models[i].pred_train) * self.class_prior[i] \
                                                         for i in range(self.n_class)]).transpose(), axis=1)
        
    def visualize_d2(self, title="", n_points=50): 
        _, ax = plt.subplots(1)
        ax.scatter(self.grid_data[:,0], self.grid_data[:,1], c = self.pred_grid_data, \
                        marker = 'o', s=150, label='grid_data')
        ax.scatter(self.data_train[:,0], self.data_train[:,1], \
                   c = self.labels_train, marker = 'v', s=150, label='train_data')        
        ax.scatter(self.data_test_d4[:,0], self.data_test_d4[:,1],\
                    c = self.labels_test, marker = 's', s=150, label='test_data')
        plt.legend()
        plt.title(title)
        plt.show()
        
    def learning_curve(self, dimension=2):
        _, ax = plt.subplots(1)
        sigma=np.linspace(0.001,5,100)
        err_train=np.zeros(100)
        err_valid=np.zeros(100)
        
        
        for i in range(len(sigma)):
            self.train(dim=dimension, model='parzen', sigma=sigma[i])
            self.compute_predictions()
            err_train[i], err_valid[i] = self.err_train, self.err_test
            
            if i % 25 == 0:
                print "Complete", i , "% iteration"
        

        print "\nBayes Classifier - Best Sigma values (on valid data) with", \
                self.dim  ,"features= ",sigma[np.where(err_valid == err_valid.min())], \
                "Train Error=", err_train[err_valid == err_valid.min()], '%',"Valid Error=",\
                 round(err_valid.min(),2), '%'
        
        
        ax.plot(sigma, err_train, label='Training data error')
        ax.plot(sigma, err_valid, color='red', label='Validation data error')
        plt.ylabel('Error (%)')
        plt.xlabel('Sigma')
        plt.legend()
        plt.show()



    
def main():
    
    
    ##############################################
    ################ Abbas Ghaddar ###############
    ##############################################
    
    
    ##############################################
    # Important Notes:
    # 0- The code use Python 2.7
    # 1- The main method will plot the figures sequentially. Close a figure to process the next one.
    # 2- The code use load_iris() method from sklearn library in order to import iris data
    # 3- compute_predictions methods return the log p(x) as required in the question sheet.
    #         In order to have a clearer figure, i plot the np.exp(predictions). 
    # 4- The answer of question 4.4 is at the end of the main method.  
    ##############################################
    
      
       
    
    ##############################################
    # Practice : Density estimation
    ##############################################
     
    #part 3    
    d1 = density_1(1,2, 50)# parameters=  class, dim, number of point
    d1.plot_data()
     
    #part 4
    d2 = density_2(1,2,50)# parameters= class, dim and  dim+1, number of point
    [d2.plot_data(item) for item in ['a', 'b','c', 'd']]
 
           
       
    ##############################################
    #Practice : Bayes classifier
    ##############################################
    bc = bayes_classifier()
    
    #part 4.2.b
    bc.train(dim=2, model='gaussian')
    bc.compute_predictions(run_grid=True)
    bc.visualize_d2(title="Bayes Classifier - Guassian")
     
    #part 4.2.c
    print "Bayes Classifier - Gaussian (dim=2)\t Training error= ", round(bc.err_train,2), \
            '%', "Validation error=", round(bc.err_test,2), '%'
     
    #part 4.2.d
    bc.train(dim=4, model='gaussian')
    bc.compute_predictions()
    print "Bayes Classifier - Gaussian (dim=4)\t Training error= ", round(bc.err_train,2), \
            '%', "Validation error=", round(bc.err_test,2), '%'
     
     
    #part 4.3.b
    sig = (('Bayes Classifier - Parzen Sigma Small', 0.01), ('Bayes Classifier - Parzen Sigma Big', 1),\
            ('Bayes Classifier - Parzen Sigma Appropriate',0.3))
    for item in sig:
        bc.train(dim=2, model='parzen', sigma=item[1])
        bc.compute_predictions(run_grid=True)
        bc.visualize_d2(title=item[0])
        
    #part 4.3.c
    print "Learning curve dim=2"
    bc.learning_curve(dimension=2)
    
    #part 4.3.d
    print "Learning curve dim=4"
    bc.learning_curve(dimension=4)
    

    
    #part 4.4
    '''
    Features (2 vs. 4):
    ------------------
    
    In all cases, using 4 features gives better performance on the validation set.

    Parametrized  Gaussian vs. Parzen windows (2 features):
    -------------------------------------------------------
    
    The Bayes classifier based on Parzen density performs better than the classifier
    based on diagonal Gaussian parametric density using 2 features. The classifier 
    based on diagonal Gaussian parametric density obtains an error rate of 16.67\%
    compared to 13.33\% for the classifier based on the Parzen density on the validation
    set (with an appropriate sigma value).

    Parametrized  Gaussian vs. Parzen windows (4 features):
    -------------------------------------------------------
    
    The Bayes classifier based on Parzen density performs as well as the classifier based
    on diagonal Gaussian parametric density using 4 features. Both classifiers obtain an
    error rate of 3.33 on the validation set; however the Bayes classifier based on Parzen
    density obtains better results (0\% error) compared with the Gaussian parametric density
    (3.3\% error).
    
    Conclusion:
    -----------
    The results are not surprising because the models using Parzen density have a higher capacity 
    than the models based on diagonal Gaussian parametric density 
        
    '''
    
    
    
if __name__ == "__main__":
    main() 
