
import numpy as np
# train the network

layers = []
loss = None
loss_prime = None



class ActivationLayer():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

# inherit from base class Layer
class FCLayer():
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        self.input_a = self.output
        self.output_a = np.tanh(self.input_a)

        #self.output = np.exp(self.output)/np.sum(np.exp(self.output),axis=0)
        return self.output_a 

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        #loss=-np.sum(y*np.log(output_error))
        #output_error = loss/float(output_error.shape[0])
        output_error = (1-np.tanh(self.input_a)**2) * output_error

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

# predict output for given input
def predict( input_data):
    # sample dimension first
    samples = len(input_data)
    result = []

    # run network over all samples
    for i in range(samples):
        # forward propagation
        output = input_data[i]
        for layer in layers:
            output = layer.forward_propagation(output)
        result.append(output)

    return result

def fit( x_train, y_train, epochs, learning_rate):
    # sample dimension first
    samples = len(x_train)

    # training loop
    for i in range(epochs):
        err = 0
        for j in range(samples):
            # forward propagation
            output = x_train[j]
            for layer in layers:
                output = layer.forward_propagation(output)

            # compute loss (for display purpose only)
            
            err += np.mean(np.power(y_train[j]-output, 2))

            # backward propagation
            
            error = 2*(output-y_train[j])/y_train[j].size
            for layer in reversed(layers):
                error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
        err /= samples
        print('epoch %d/%d   error=%f' % (i+1, epochs, err))

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network

layers.append(FCLayer(2, 3))
layers.append(FCLayer(3, 1))

# train
fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = predict(x_train)
print(out)