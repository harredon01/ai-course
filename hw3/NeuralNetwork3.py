import sys
import numpy as np
import math
import time
arguments = sys.argv
train_data = arguments[1]
train_labels = arguments[2]
test_data = arguments[3]

levels = []
files = ['spiral','circle','gaussian','xor']

class FClevel():
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5


    def f_prop(self, vals):
        try:
            self.input = vals
            self.output = np.dot(self.input, self.weights) + self.bias
        except:
            print("Error in weights")
        try:
            self.input_a = self.output
            self.output_a = np.tanh(self.input_a)
        except:
            print("Error in acum")
        return self.output_a 

    def b_prop(self, output_error, alpha):
        #loss=-np.sum(y*np.log(output_error))
        #output_error = loss/float(output_error.shape[0])
        output_error = (1-np.tanh(self.input_a)**2) * output_error

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
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

def load_data(lines,lines2):
    tr_arr = []
    tr_lbs_arr = []
    for it in range(len(lines)):
        tup = lines[it].replace("\n","").split(",")
        try:
            tr_arr.append([[float(tup[0]),float(tup[1])]])
            tr_lbs_arr.append([[float(lines2[it].replace("\n",""))]])
        except:
            print("ERROR")
            print(tup)
            print(lines2[it])
            exit()
    x_train = np.array(tr_arr)
    y_train = np.array(tr_lbs_arr)
    return x_train,y_train

def train( x_train, y_train, alpha, epochs):
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

def build_levels(z,x,y,w,o,p):
    global levels
    levels = []
    if z == 1:
        levels.append(FClevel(2, 1))
    elif z ==2:
        levels.append(FClevel(2, x))
        levels.append(FClevel(x, 1))
    elif z ==3:
        levels.append(FClevel(2, x))
        levels.append(FClevel(x, y))
        levels.append(FClevel(y, 1))
    elif z == 4:
        levels.append(FClevel(2, x))
        levels.append(FClevel(x, y))
        levels.append(FClevel(y, w))
        levels.append(FClevel(w, 1))
    elif z == 5:
        levels.append(FClevel(2, x))
        levels.append(FClevel(x, y))
        levels.append(FClevel(y, w))
        levels.append(FClevel(w, o))
        levels.append(FClevel(o, 1))
    elif z == 6:
        levels.append(FClevel(2, x))
        levels.append(FClevel(x, y))
        levels.append(FClevel(y, w))
        levels.append(FClevel(w, o))
        levels.append(FClevel(o, p))
        levels.append(FClevel(p, 1))

def get_error(predictions,real):
    tot_err = 0
    for it in range(len(predictions)):
        tot_err += np.mean(np.power(real[it]-predictions[it], 2))
    return tot_err

def bla(z,x,y,w,o,p,lines1,lines2,lines3,lines4,batches,alpha,epochs):
    x_test,y_test = load_data(lines3,lines4)
    build_levels(z,x,y,w,o,p)
    bat_size = math.floor(len(lines1)/batches)
    counter = 0
    start_time = time.time()
    for item in range(batches):
        end = counter + bat_size
        x_train, y_train = load_data(lines1[counter:end],lines2[counter:end])
        train(x_train, y_train, alpha, epochs)
        counter = counter + bat_size
    train_time = time.time()-start_time
    x_test,y_test = load_data(lines3,lines4)
    out = predict(x_test)
    error = get_error(out,y_test)
    error = str(error)+","+str(train_time)
    return error

def run_model(lines1,lines2,lines3,lines4,it,batches,alpha,epochs):
    results = {}
    for i in range(1,7):
        key = it+"-"+str(2)+"-"+str(i)+"-"+str(0)+"-"+str(0)+"-"+str(0)+"-"+str(0)+"-"+str(batches)+"-"+str(alpha)+"-"+str(epochs)
        error = bla(2,i,0,0,0,0,lines1,lines2,lines3,lines4,batches,alpha,epochs)
        print("With: ",key," error: ",error)
        results[key]=error
        for j in range(3,7):
            key = it+"-"+str(3)+"-"+str(i)+"-"+str(j)+"-"+str(0)+"-"+str(0)+"-"+str(0)+"-"+str(batches)+"-"+str(alpha)+"-"+str(epochs)
            error = bla(3,i,j,0,0,0,lines1,lines2,lines3,lines4,batches,alpha,epochs)
            print("With: ",key," error: ",error)
            results[key]=error
            for k in range(1,7):
                key = it+"-"+str(4)+"-"+str(i)+"-"+str(j)+"-"+str(k)+"-"+str(0)+"-"+str(0)+"-"+str(batches)+"-"+str(alpha)+"-"+str(epochs)
                error = bla(4,i,j,k,0,0,lines1,lines2,lines3,lines4,batches,alpha,epochs)
                print("With: ",key," error: ",error)
                results[key]=error
                #for o in range(1,6):
                    #key = it+"-"+str(4)+"-"+str(i)+"-"+str(j)+"-"+str(k)+"-"+str(o)+"-"+str(0)+"-"+str(batches)+"-"+str(alpha)+"-"+str(epochs)
                    #error = bla(5,i,j,k,o,0,lines1,lines2,lines3,lines4,batches,alpha,epochs)
                    #print("With: ",key," error: ",error)
                    #results[key]=error
                    #for p in range(1,6):
                        #key = it+"-"+str(4)+"-"+str(i)+"-"+str(j)+"-"+str(k)+"-"+str(o)+"-"+str(p)+"-"+str(batches)+"-"+str(alpha)+"-"+str(epochs)
                        #error = bla(6,i,j,k,o,p,lines1,lines2,lines3,lines4,batches,alpha,epochs)
                        #print("With: ",key," error: ",error)
                        #results[key]=error
    text_file = open("results.csv", "a")
    for it in results:
        val = str(it)+","+str(results[it])+"\n"
        text_file.write(val)
    text_file.close()

for it in files:
    train_file = it+"_train_data.csv"
    train_labels = it+"_train_label.csv"
    test_data = it+"_test_data.csv"
    test_labels = it+"_test_label.csv"
    with open(train_file,'r') as i:
        lines1 = i.readlines()
    with open(train_labels,'r') as i:
        lines2 = i.readlines()
    with open(test_data,'r') as i:
        lines3 = i.readlines()
    with open(test_labels,'r') as i:
        lines4 = i.readlines()


    run_model(lines1,lines2,lines3,lines4,it,10,0.01,40)
    run_model(lines1,lines2,lines3,lines4,it,10,0.001,40)
    run_model(lines1,lines2,lines3,lines4,it,10,0.01,100)
    run_model(lines1,lines2,lines3,lines4,it,10,0.001,100)

exit()
epochs=100
z=1
x=1
y=1
w=1
batches=10
with open(train_data,'r') as i:
    lines1 = i.readlines()
with open(train_labels,'r') as i:
    lines2 = i.readlines()
with open(test_data,'r') as i:
    lines3 = i.readlines()
alpha=0.1
build_levels(z,x,y,w)
bat_size = math.floor(len(lines1)/batches)
counter = 0
start_time = time.time()
for item in range(batches):
    end = counter + bat_size
    x_train, y_train = load_data(lines1[counter:end],lines2[counter:end])
    train(x_train, y_train, alpha, epochs)
    counter = counter + bat_size
x_test,y_test = load_data(lines3,lines3)
out = predict(x_test)
text_file = open("test_predictions.csv", "w")
for it in out:
    val = str(it[0][0])+"\n"
    n = text_file.write(val)
text_file.close()
print(out)