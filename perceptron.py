import numpy as np

'''
  X  is a Data sample set with a bias 0 at the end
'''
X = np.array([
    [0,0,0],
    [0,1,0],
    [1,0,0],
    [1, 1,0],

])

'''Y is a data set OR corresponding target of the data set containing three samples labeled with 0
 and a sample labeled with +1'''
Y = np.array([0,0,0,1])

'''perceptron_sgd = Stochastic Gradient Descent
    :param X: data samples
    :param Y: data labels
    :return: weight vector as a numpy array'''
def perceptron_sgd(X, Y):

    '''weight vector(w) for the perceptron with zeros'''
    w = np.zeros(len(X[0]))

    '''learning rate is set to 1'''
    rate = 1 

    '''epoch to tell how much to learn'''
    epochs = 20

    for t in range(epochs):
        for i, x in enumerate(X):

            '''condition Yi⟨Xi,w⟩≤0 
             Update rule for the weights w=w+Yi*Xi'''
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + rate*X[i]*Y[i]

    return w
'''calls the function'''
w = perceptron_sgd(X,Y)

'''weight vector including the bias term (x1,x2,bias)'''
print('The required weight and bias in the form [x1. x2. bias.] to generate the output is :')
print(w)
