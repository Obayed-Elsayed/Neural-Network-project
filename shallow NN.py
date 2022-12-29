import numpy as np
import pandas as pd                          
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(400136475)
# functions used in computing the gradient and doing the backpropagation
def relu(x):                     
    return np.where(x <= 0, 0, x)
def relu_prime(x):                
    return np.where(x <= 0, 0, 1)

def sig(x):                                
    return 1/(1 + np.exp(-x))
def sig_prime(x):                            
    return sig(x) * (1 - sig(x))
# loss function
def cross_E(y_true, y_pred):   
    if(y_true ==0):
        return np.sum(np.log(1+np.exp(y_pred)))          
    else:       
        return np.sum(np.log(1+np.exp(-y_pred)))
# loss function prime
def cross_E_grad(y_true, y_pred):               
    if(y_true ==0):
        return np.sum(np.exp(y_pred)/(np.exp(y_pred)+1))          
    else:       
        return np.sum(-1/(np.exp(y_pred)+1))
#use this to shuffle both input and output matricies so that they correspond to each other 
def unison_shuffled_copies(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

# get number of miss-classified examples
def compare_classification(arr1:np.ndarray,arr2:np.ndarray):
    errs =0 
    for i in range(len(arr1)):
        if (arr1[i] != arr2[i]):
            errs+=1
    return errs

def to_class(inp):
    classified =[]
    for i in range(len(inp)):
        if(inp[i]>0.5):
            classified.append(1)
        else:
            # print(inp[i])
            classified.append(0)
    return classified
# input validation set to find which is the best layout for the network
def find_best_network(x,t):
    for i in range(1,5):
        for j in range(1,5):
            mean_err =[]
            for k in range(5):
                np.random.seed(6475+k)
                network = NN(4,i,j,1)
                network.train(75,0.005,x,t)
                mean_err.append(network.cross_error[-1])
            
            print("H1 nodes: "+ str(i)+ " H2 nodes: "+str(j) +" "+ str(sum(mean_err)/len(mean_err)) +" missclassification: "+
             str(compare_classification(to_class(network.predict(x)),t)))
class NN:

    def __init__(self,input_nodes,hidden_1_nodes,hidden_2_nodes ,output_nodes):
        self.input_nodes = input_nodes                                                    
        self.hidden_1_nodes = hidden_1_nodes 
        self.hidden_2_nodes = hidden_2_nodes 
        self.output_nodes = output_nodes 

        # weights and bias for each layer
        # going into Hidden Layer1
        self.w1 = np.random.random(size = (hidden_1_nodes, input_nodes))    
        self.b1 = np.zeros((hidden_1_nodes, 1))
        self.w2 = np.random.random(size = (hidden_2_nodes, hidden_1_nodes)) 
        self.b2 = np.zeros((hidden_2_nodes, 1))   
        self.w3 = np.random.random(size = (output_nodes, hidden_2_nodes))   
        self.b3 = np.zeros((output_nodes, 1))
        self.cross_error = []
        self.classification_err =[]

    def plot_error(self):
        epochs = np.linspace(1,len(self.cross_error),len(self.cross_error))
        plt.plot(epochs, self.classification_err)
        plt.plot(epochs, self.cross_error)
        plt.show()
    def predict(self, input):
        y_pred =[]
        for j in range(input.shape[0]):
            x=input[j:j+1,:]
            x=x.T
            z1 = self.w1.dot(x) + self.b1                      
            h1 = relu(z1)

            z2 = self.w2.dot(h1) + self.b2
            h2 = relu(z2)

            z3 = self.w3.dot(h2) + self.b3
            y_hat = sig(z3)  
            y_pred.append(y_hat[0][0])

        return y_pred    

    def train(self,epochs, a, input_matrix,t):
        for i in range(epochs):
            data, out = unison_shuffled_copies(input_matrix,t)
            # as per document, go through all shuffled training points and update weights
            sum_cross =0
            for j in range(input_matrix.shape[0]):
                x=data[j:j+1,:]
                y=out[j:j+1]
                x=x.T
                y.reshape(1,1)
                z1 = self.w1.dot(x) + self.b1                      
                h1 = relu(z1)

                z2 = self.w2.dot(h1) + self.b2
                h2 = relu(z2)

                z3 = self.w3.dot(h2) + self.b3
                y_hat = sig(z3)     
                                                
                cross_e = cross_E(y, y_hat)
                sum_cross+=cross_e
                # layer 3
                grad_w3 = cross_E_grad(y, y_hat) * sig_prime(z3).dot(h2.T )           
                grad_b3 = cross_E_grad(y, y_hat) * sig_prime(z3)

                # layer 2
                error_grad_upto_H2 = np.sum(cross_E_grad(y, y_hat) *sig_prime(z3) * self.w3, axis = 0).reshape((-1, 1))
                grad_w2 = error_grad_upto_H2 * relu_prime(z2).dot(h1.T )
                grad_b2 = error_grad_upto_H2 * relu_prime(z2)
                # layer 1
                error_grad_upto_H1 = np.sum(error_grad_upto_H2 *sig_prime(z2) * self.w2, axis = 0).reshape((-1, 1))
                grad_w1 = error_grad_upto_H1 * relu_prime(z1).dot(x.T)
                grad_b1 = error_grad_upto_H1 * relu_prime(z1)

                # update weights and gradients
               
                self.w1 += - a * grad_w1                               
                self.b1 += - a * grad_b1                                
                
                self.w2 += - a * grad_w2                                 
                self.b2 +=  - a * grad_b2                                 
                
                self.w3 += - a * grad_w3                                
                self.b3 += - a * grad_b3

            # print(str(i) +": "+str(sum_cross/input_matrix.shape[0]))
            classes = to_class(self.predict(input=data))
            self.cross_error.append(sum_cross/input_matrix.shape[0])
            self.classification_err.append(compare_classification(classes,out)/float(input_matrix.shape[0]))

def main():
    # setting up data
    df = pd.read_csv("data_banknote_authentication.txt", sep=",")
    inp = df.iloc[:, :-1].values
    t = df.iloc[:, -1].values

    sc = StandardScaler()

    inp,t = unison_shuffled_copies(inp,t)
    x_train= inp[0:960,:]
    y_train= t[0:960]


    x_valid=inp[960:1166,:]
    y_valid=  t[960:1166]

    x_test=inp[1166:,:]
    y_test=  t[1166:]

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    x_valid = sc.transform(x_valid)

    # try 16 different configs
    # find_best_network(x_valid,y_valid)

    print(x_valid.shape)
    network = NN(4,4,4,1)
    network.train(75,0.005,x_train,y_train)
    network.plot_error()
    classes = to_class(network.predict(input=x_test))
    print(compare_classification(classes,y_test)/float(x_test.shape[0]))



main()
