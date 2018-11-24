#This algorithm was taken from
# https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python
# and adapted for it to recieve a .txt file

import numpy as np
import fileinput


class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 9
    self.outputSize = 1
    self.hiddenSize = 10000

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (100x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (100x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propgate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
    self.z2_error = np.dot(self.W2, self.W2.T) #self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 = X.T.dot(self.z2_delta) + self.W1 # adjusting first set (input --> hidden) weights
    self.W2 = self.z2.T.dot(self.o_delta) + self.W2# adjusting second set (hidden --> output) weights

  def train (self, X, y):
    #print(X)
    o = self.forward(X)
    self.backward(X, y, o)

def main():
    file_input = fileinput.input()
    d = int(file_input[0])
    m_size = int(file_input[1])
    n_size = int(file_input[2])
    x_training_set = []
    y_training_set = []
    x_test_set = []

    for i in range(m_size):
        training_set_str = file_input[i + 3].replace(" ", "").replace("\t", "").replace("\n", "").split(',')
        training_set_float = [float(x) for x in training_set_str]
        x_training_row = training_set_float[0:d]
        #x_training_row.append(1)

        x_training_set.append(x_training_row)
        y_training_set.append(training_set_float[d])

    for i in range(n_size):
        test_set_str = file_input[i + m_size +3].replace(" ", "").replace("\t", "").replace("\n", "").split(',')
        test_set_float = [float(x) for x in test_set_str]
        x_test_row = test_set_float[0:d]
        #x_test_row.append(1)
        x_test_set.append(x_test_row)

    X = np.asarray(x_training_set)
    y = np.asarray(y_training_set)

    X = X/np.amax(X, axis=0) # maximum of X array
    y = y/1
    #y = y/100 # max test score is 100

    NN = Neural_Network()
    for i in range(10): # trains the NN 1,000 times
      print ("Input: \n" + str(X))
      print ("Actual Output: \n" + str(y))
      print ("Predicted Output: \n" + str(NN.forward(X)))
      print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
      print ("\n")
      NN.train(X, y)


if __name__ == "__main__":
    main()

