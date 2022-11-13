import sys
import numpy as np
arguments = sys.argv
train_data = arguments[1]
train_labels = arguments[2]
test_data = arguments[3]

levels = []
loss = None
loss_prime = None


class FClevel():
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5


    def f_prop(self, vals):
        self.input = vals
        self.output = np.dot(self.input, self.weights) + self.bias
        self.input_a = self.output
        self.output_a = np.tanh(self.input_a)

        #self.output = np.exp(self.output)/np.sum(np.exp(self.output),axis=0)
        return self.output_a 

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def b_prop(self, output_error, alpha):
        #loss=-np.sum(y*np.log(output_error))
        #output_error = loss/float(output_error.shape[0])
        output_error = (1-np.tanh(self.input_a)**2) * output_error

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= alpha * weights_error
        self.bias -= alpha * output_error
        return input_error

# predict output for given input
def predict( vals):
    data_s = len(vals)
    result = []
    for i in range(data_s):
        output = vals[i]
        for level in levels:
            output = level.f_prop(output)
        result.append(output)
    return result

def load_data():
    with open(train_data,'r') as i:
        lines = i.readlines()
    with open(train_labels,'r') as i:
        lines2 = i.readlines()
    tr_arr = []
    tr_lbs_arr = []
    for it in range(len(lines)):
        tup = lines[it].replace("\n","").split(",")
        tr_arr.append([[float(tup[0]),float(tup[1])]])
        tr_lbs_arr.append([[float(lines2[it].replace("\n",""))]])
    x_train = np.array(tr_arr)
    y_train = np.array(tr_lbs_arr)
    return x_train,y_train

def train( x_train, y_train, epochs, alpha):
    data_s = len(x_train)
    for i in range(epochs):
        tot_err = 0
        for j in range(data_s):
            output = x_train[j]
            for level in levels:
                output = level.f_prop(output)
            tot_err += np.mean(np.power(y_train[j]-output, 2))
            error = 2*(output-y_train[j])/y_train[j].size
            for level in reversed(levels):
                error = level.b_prop(error, alpha)
        tot_err /= data_s
        print('epoch %d/%d   error=%f' % (i+1, epochs, tot_err))

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
x_train, y_train = load_data()
# network

levels.append(FClevel(2, 3))
levels.append(FClevel(3, 1))

# train
train(x_train, y_train, epochs=10, alpha=0.1)

# test
out = predict(x_train)
text_file = open("test_predictions.csv", "w")
for it in out:
    val = str(it[0][0])+"\n"
    n = text_file.write(val)
text_file.close()
print(out)