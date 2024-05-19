import csv
import numpy as np
import tkinter as tk
#import time
#Layer maker---

input = [0,0,1,0,0,1,1,0,0,0,0,1]#Test inputs for first 3 datasets present in data1.csv
input2 = [1,0,0,0,1,0,1,0,0,0,1,0]
input3 = [0,1,0,0,0,1,0,0,1,0,0,1]
neurons_per_layer=7#number of nodes per hidden layer
connections=12#used for input to hidden connection
EPOCH=100#epoch stop condition
N=50000#sample size
LOSS_CONV=0.000001#This will determine the threshold for desired convergence
layers_hidden=1#number of hidden layers
Input_Data=[]


def Input(N):#makes an array containing the ideal_gene in a BFS depth 4 order if some items are missing in historic data a 1 is added indidicating that gene is non-determnitic in fitness
    try:
        with open("data1.csv", 'r', newline='') as file:#open the file
            reader = csv.reader(file)
            data=[]
            j=0
            for row in reader:#append the file contents to an array
                data.append(row)
                j+=1
            input_data = [] * N#Input data for backprop
            target_data = [] * N#target data for backprop
            random_batch=np.random.randint(0,1000000-N)#Random batch sampling
            for x in range(N):#for the sample size
                history = data[x+random_batch][0]#sample a random batch of data
                target = data[x+random_batch][1]#---
                conc_input = []
                for i in history:#normalise data to 1-hot-encoding
                    if i == "R":
                        conc_input.append(1)#Rock
                        conc_input.append(0)
                        conc_input.append(0)
                    elif i == "P":
                        conc_input.append(0)
                        conc_input.append(1)#Paper
                        conc_input.append(0)
                    elif i == "S":
                        conc_input.append(0)
                        conc_input.append(0)
                        conc_input.append(1)#Scisscors
                    else:
                        print("big error")
                input_data.append(conc_input)
                conc_targ=[]
                if target == "R":#target data for loss caluculation and training
                    conc_targ.append(0)
                    conc_targ.append(1)
                    conc_targ.append(0)
                elif target == "P":
                    conc_targ.append(0)
                    conc_targ.append(0)
                    conc_targ.append(1)
                elif target == "S":
                    conc_targ.append(1)
                    conc_targ.append(0)
                    conc_targ.append(0)
                else:
                    print("big error")
                target_data.append(conc_targ)
            return input_data, target_data
    except FileNotFoundError:
        return "CSV file not found."


#COMMNENTS FOR THE REST OF THE CODE ARE PRESENT IN TASK 2 AND 3---NOTE THIS FILE HAS A VISUALISER
# data = 'data1.csv'
# historicData(data)
def Activation_function(value):
    return 1 / (1 + np.exp(-value))

def Activation_prime(value):
    sig_x = Activation_function(value)
    return sig_x * (1 - sig_x)


def update_stuff(network):
    Z=[]*(network.hidden_size+1)
    A=[]*(network.hidden_size+1)
    B=np.zeros(network.hidden_size+1)
    F_primeZ=[]*(network.hidden_size+1)
    for q in range(network.hidden_size+1):
        if q!=network.hidden_size:
            arr1=[]
            arr2=[]
            arr3=[]
            for r in range(neurons_per_layer+1):
                if r!=neurons_per_layer:
                    arr1.append(network.layer[q][r].sum)
                    arr2.append(network.layer[q][r].Activation)
                    arr3.append(Activation_prime(network.layer[q][r].sum))
                else:
                    arr1.append(1)
                    arr2.append(1)

            Z.append(np.array(arr1))
            A.append(np.array(arr2))
            F_primeZ.append(np.array(arr3))
        else:
            arr1=[]
            arr2=[]
            arr3=[]
            for r in range(4):
                if r!=3:
                    arr1.append(network.layer[q][r].sum)
                    arr2.append(network.layer[q][r].Activation)
                    arr3.append(Activation_prime(network.layer[q][r].sum))
            Z.append(np.array(arr1))
            A.append(np.array(arr2))
            F_primeZ.append(np.array(arr3))
            # print(Z[0])
            # print(A[0])
            # print(F_primeZ[0])
    return Z,A,F_primeZ
class Neuron:
    def __init__(self,is_bias,value,connections):
        self.Activation=0
        if not is_bias:
            self.weights=[]*connections
            self.sum=0
            for x in range(connections):
                self.weights.append(np.random.uniform(-(1/np.sqrt(neurons_per_layer)),(1/np.sqrt(neurons_per_layer))))
        else:
            self.weights=None
            self.Activation=0


    def activate(self):
        self.Activation = Activation_function(self.sum)
        return self.Activation

class Layer():
    def __init__(self, num_hidden):
        self.layer = []*(num_hidden+1)
        self.hidden_size = num_hidden
        self.bias = []*(num_hidden+1)
        for i in range (num_hidden+1):
            if i != (num_hidden):
                self.layer.append([]*neurons_per_layer)
                self.bias.append([]*neurons_per_layer)
                for k in range(neurons_per_layer):
                    if i == 0:
                        self.layer[i].append(Neuron(False, 1, 12))
                    else:
                        self.layer[i].append(Neuron(False, 1, neurons_per_layer))
                    self.bias[i].append(1)
            else:
                self.layer.append([] * 3)
                self.bias.append([]*neurons_per_layer)
                for k in range(3):
                    if i == 0:
                        self.layer[i].append(Neuron(False, 1, 12))
                    else:
                        self.layer[i].append(Neuron(False, 1, neurons_per_layer))
                    self.bias[i].append(1)
    def load(self,weight_in,weight_hidden,weight_output):
        for x in range(self.hidden_size + 1):
            if x == 0:
                for i in range(neurons_per_layer):
                    for j in range(13):
                        if j == 12:
                            self.bias[x][i]= weight_in[i][j]
                        else:
                            self.layer[x][i].weights[j] = weight_in[i][j]
            elif x == (self.hidden_size):
                for i in range(3):
                    for j in range(neurons_per_layer + 1):
                        if j == neurons_per_layer:
                            self.bias[x][i]= weight_output[i][j]
                        else:
                            self.layer[x][i].weights[j] = weight_output[i][j]
            else:
                for i in range(neurons_per_layer):
                    for j in range(neurons_per_layer + 1):
                        if j == neurons_per_layer:
                            self.bias[x][i]= weight_hidden[i][j]
                        else:
                            self.layer[x][i].weights[j] = weight_hidden[i][j]
NeuralNet = Layer(layers_hidden)
def forward_propagation(hidden_layer,current_input):
    Activation=[]*(hidden_layer.hidden_size+1)
    for x in range(hidden_layer.hidden_size+1):
        current_layer=hidden_layer.layer[x]
        bias = hidden_layer.bias
        if x==0:
            Activation.append([] * 12)
            for y in range(neurons_per_layer):
                sum=0
                for z in range(12):
                    sum+=(current_layer[y].weights[z])*current_input[z]
                current_layer[y].sum=sum+bias[x][y]
                #print(current_layer[y].sum)
                Activation[x].append(current_layer[y].activate())
        elif x==hidden_layer.hidden_size:
            Activation.append([] * 3)
            for y in range(3):
                sum = 0
                for z in range(neurons_per_layer):
                    sum += (current_layer[y].weights[z])*Activation[x-1][z]
                current_layer[y].sum = sum + bias[x][y]
                Activation[x].append(current_layer[y].activate())
        else:
            Activation.append([] * neurons_per_layer)
            for y in range(neurons_per_layer):
                sum=0
                for z in range(neurons_per_layer):
                    sum+=(current_layer[y].weights[z])*Activation[x-1][z]
                current_layer[y].sum=sum+bias[x][y]
                Activation[x].append(current_layer[y].activate())
    output=np.zeros(3)
    for k in range(3):
        output[k]=hidden_layer.layer[hidden_layer.hidden_size][k].Activation
    return output

def Initialise(inp_arr):
    Network=NeuralNet
    sample=[]*N
    for n in range(N):
        sample.append(forward_propagation(Network,inp_arr[n]))#might cause issues bec you saying an index of one array is equal to another whole array... if it works then its an easy way to make it two dimentional
    return sample,Network


def backpropagation(input_data,target_data,weights_input,weights_hidden,weights_output,learning_rate,epoch,loss_convergence):
    weights_input=weights_input.transpose()#TODO move these out of backprop
    weights_hidden=weights_hidden.transpose()
    weights_output=weights_output.transpose()
    N=len(target_data)
    inp_add=np.array([1]*N)
    input_data=np.hstack((np.array(input_data),inp_add.reshape(-1,1)))
    # print(input_data)
    Loss_vector=np.zeros(epoch)
    bias_input=None
    bias_hidden=None
    bias_output=None

    for i in range(epoch):
        Dontuse, network = Initialise(input_data)
        # print(Z[0])
        # print(A[0])
        # print(F_primeZ[0])
        # print('==')
        if i!=0:
            if Loss_vector[i-1]<=loss_convergence:
                print("converged")
                break

        for x in range(N):
            NeuralNet.load(weights_input.transpose(), weights_hidden.transpose(), weights_output.transpose())
            Y=forward_propagation(NeuralNet,input_data[x])
            Z, A, F_primeZ = update_stuff(NeuralNet)
            Loss_sum=0
            loss_sample=np.zeros(3)
            for ind in range(3):
                Loss_sum+=target_data[x][ind]-Y[ind]
                loss_sample[ind]=target_data[x][ind]-Y[ind]
            if x==1:
                Loss_vector[i]=(np.power(Loss_sum,2)*0.5)
            # print(target_data[x])
            # print(Y[x])
            # print(loss_sample)
            delta=[]*(NeuralNet.hidden_size+1)
            for y in range(NeuralNet.hidden_size+1):

                if y==0:#output layer
                    # print("N:"+str(x))
                    # print(target_data[x])
                    # print(Y[x])
                    # print(loss_sample)
                    # print("---------")
                    # print(F_primeZ[NeuralNet.hidden_size])
                    delta_n=(-loss_sample)*F_primeZ[NeuralNet.hidden_size]
                    # print(A[NeuralNet.hidden_size-1].reshape(1,-1))
                    # print("-----")
                    delta.append(delta_n)

                    dJdW=np.dot((A[NeuralNet.hidden_size-1]).reshape(1, -1).transpose(),delta_n.reshape(1, -1))
                    # print("N:"+str(x)+" weight_out")
                    # print(dJdW)
                    djdB=delta_n.sum()
                    weights_output=weights_output-(learning_rate*dJdW)
                elif y==NeuralNet.hidden_size:
                    if NeuralNet.hidden_size>1:
                        delta_n=np.dot((delta.pop().reshape(1,-1)),np.delete((weights_hidden.transpose()),-1,axis=1).transpose())*F_primeZ[0]
                    else:
                        u=delta.pop()
                        delta_n=(u.reshape(1,-1)@np.delete((weights_output.transpose()),-1,axis=1))*F_primeZ[0]
                    delta.append(delta_n)
                    dJdW = np.dot(np.array(input_data[x]).reshape(1,-1).transpose(),delta_n.reshape(1,-1),)#todo add bias neuron to input
                    djdB=delta_n.sum()
                    # print("N:"+str(x)+" weight_out")
                    # print(dJdW)
                    weights_input = weights_input-(learning_rate*dJdW)
                else:
                    #if y==1:
                    delta_n=np.dot((delta.pop()).reshape(1,-1),np.delete((weights_output).transpose(),-1,axis=1))*F_primeZ[NeuralNet.hidden_size-y]
                    delta.append(delta_n)#TODO come back to make dynamic
                    # else:
                    #     delta_n=(delta.pop())*(weights_hidden_w[NeuralNet.hidden_size-y].transpose())*F_primeZ[NeuralNet.hidden_size-y+1]
                    #     delta.append(delta_n)
                    dJdW=np.dot((A[NeuralNet.hidden_size-y]).reshape(1, -1).transpose(),delta_n.reshape(1, -1))
                    # bias_hidden=np.full(neurons_per_layer,B[NeuralNet.hidden_size-y])

                    weights_hidden = weights_hidden-(learning_rate*dJdW)


            # weights_input = np.hstack((weights_input, bias_input.reshape(-1,1)))
            # if weights_hidden.any():
            #     weights_hidden = np.hstack((weights_hidden_w, bias_hidden.reshape(-1,1)))
            # weights_output = np.hstack((weights_output_w, bias_output.reshape(-1,1)))
        # print("Input")
        # print(weights_input.transpose())
        # print("-----------")
        # print("hidden")
        # print(weights_hidden.transpose())
        # print("------------")
        # print("output")
        # print(weights_output.transpose())


    if weights_hidden.any():
        return weights_input,weights_hidden,weights_output,Loss_vector
    else:
        return weights_input,weights_hidden,weights_output,Loss_vector

def Run(N):
    Input_Data,Target_Data=Input(N)
    Layers=NeuralNet.layer
    Weight_in=np.zeros((neurons_per_layer,13))
    Weight_hidden=np.empty((0,0))
    Weight_output=np.zeros((3,(neurons_per_layer+1)))
    forward_propagation(NeuralNet,input)
    root = tk.Tk()
    app = NeuralNetworkVisualizer(root, input, NeuralNet.layer)
    app.draw_network()
    for x in range(NeuralNet.hidden_size+1):
        if x==0:
            for i in range(neurons_per_layer):
                for j in range(13):
                    if j==12:
                        Weight_in[i][j]=NeuralNet.bias[x][i]
                    else:
                        Weight_in[i][j]=Layers[x][i].weights[j]
        elif x==(NeuralNet.hidden_size):
            for i in range(3):
                for j in range(neurons_per_layer+1):
                    if j==neurons_per_layer:
                        Weight_output[i][j]=NeuralNet.bias[x][i]
                    else:
                        Weight_output[i][j]=Layers[x][i].weights[j]
        else:
            Weight_hidden=np.zeros((neurons_per_layer,(neurons_per_layer + 1)))
            for i in range(neurons_per_layer):
                for j in range(neurons_per_layer + 1):
                    if j==neurons_per_layer:
                        Weight_hidden[i][j]=NeuralNet.bias[x][i]
                    else:
                        Weight_hidden[i][j]=Layers[x][i].weights[j]
    # print("Input")
    # print(Weight_in.transpose())
    # print("-----------")
    # print("hidden")
    # print(Weight_hidden.transpose())
    # print("------------")
    # print("output")
    # print(Weight_output.transpose())
    # print("-------------+++++++++")
    w_i, w_h, w_o, L_v=backpropagation(Input_Data,Target_Data,Weight_in,Weight_hidden,Weight_output,0.3,EPOCH,0.0001)
    return  w_i, w_h, w_o, L_v,NeuralNet





#writing weighting to csv
class NeuralNetworkVisualizer:#VISUALIZER for prac
    def __init__(self, master, input_values, layers):
        self.master = master
        self.input_values = input_values
        self.layers = layers
        self.canvas = tk.Canvas(master, width=1600, height=1200)
        self.canvas.pack()

    def draw_neuron(self, x, y, radius, activation):
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="black")
        self.canvas.create_text(x, y, text=str(round(activation, 2)))

    def draw_network(self):
        layer_width = 300
        neuron_radius = 20
        vertical_padding = 800
        horizontal_padding = 50

        # Draw input layer
        for i, input_val in enumerate(self.input_values):
            x = horizontal_padding
            y = (i + 1) * (vertical_padding / (len(self.input_values) + 1))
            self.draw_neuron(x, y, neuron_radius, input_val)

        # Draw hidden layers
        for layer_index, layer in enumerate(self.layers):
            num_neurons = len(layer)
            for neuron_index, neuron in enumerate(layer):
                x = (layer_index + 1) * layer_width + horizontal_padding
                y = (neuron_index + 1) * (vertical_padding / (num_neurons + 1))
                self.draw_neuron(x, y, neuron_radius, neuron.Activation)

                # Draw connections to previous layer
                if layer_index == 0:
                    prev_num_neurons = len(self.input_values)
                    for prev_neuron_index in range(prev_num_neurons):
                        prev_x = layer_index * layer_width + horizontal_padding
                        prev_y = (prev_neuron_index + 1) * (vertical_padding / (prev_num_neurons + 1))
                        weight = neuron.weights[prev_neuron_index]
                        line_color = "green" if weight >= 0 else "red"
                        self.canvas.create_line(prev_x + neuron_radius, prev_y, x - neuron_radius, y,
                                                arrow=tk.LAST, fill=line_color)
                        self.canvas.create_text((prev_x + x) / 2, (prev_y + y) / 2, text=str(round(weight, 2)),
                                                fill=line_color)
                else:
                    prev_num_neurons = len(self.layers[layer_index-1])
                    for prev_neuron_index in range(prev_num_neurons):
                        prev_x = layer_index * layer_width + horizontal_padding
                        prev_y = (prev_neuron_index + 1) * (vertical_padding / (prev_num_neurons + 1))
                        weight = neuron.weights[prev_neuron_index]
                        line_color = "green" if weight >= 0 else "red"
                        self.canvas.create_line(prev_x + neuron_radius, prev_y, x - neuron_radius, y,
                                                arrow=tk.LAST, fill=line_color)
                        self.canvas.create_text((prev_x + x) / 2, (prev_y + y) / 2, text=str(round(weight, 2)),
                                                fill=line_color)

        self.master.mainloop()
#Run
# forward_propagation(NeuralNet,input)
# update_stuff(NeuralNet)
weights_input,weights_hidden,weights_output,Loss_vector,net=Run(N)
if weights_hidden.any():
    net.load(weights_input.transpose(), weights_hidden.transpose(), weights_output.transpose())
else:
    net.load(weights_input.transpose(), None, weights_output.transpose())
print(forward_propagation(net,input))
root = tk.Tk()
app = NeuralNetworkVisualizer(root, input, net.layer)
app.draw_network()
print(forward_propagation(net,input2))
root = tk.Tk()
app = NeuralNetworkVisualizer(root, input2, net.layer)
app.draw_network()
print(forward_propagation(net,input3))
root = tk.Tk()
app = NeuralNetworkVisualizer(root, input3, net.layer)
app.draw_network()
print("Input")
print(weights_input)
print("-----------")
print("hidden")
print(weights_hidden)
print("------------")
print("output")
print(weights_output)
print("-------------")
print("$$Cost$$")
print(Loss_vector)
print("----------")
print("Input")
print(weights_input.transpose())
print("-----------")
print("hidden")
print(weights_hidden.transpose())
print("------------")
print("output")
print(weights_output.transpose())
print("-------------")




# for x in Tree.layer:
#     for y in range(len(x)):
#         neuron=x[y]
#         for z in range(len(neuron.weights)):
#             if neuron.weights[z] >= 0:
#                 print("Weight " + str(z+1) + " of neuron " + str(y+1) + " of layer " + str(ind) )  # Print in green
#                 print("\033[92m" ":::::"+str(neuron.weights[z])+"\033[0m")  # Print in green
#             else:
#                 print("Weight " + str(z+1) + " of neuron " + str(y+1) + " of layer " + str(ind))  # Print in red
#                 print("\033[91m" ":::::"+str(neuron.weights[z])+"\033[0m")
#                   # Print in green
#     ind+=1
#
# forward_propagation(Tree,input)
# root = tk.Tk()
# app = NeuralNetworkVisualizer(root, input, Tree.layer)
# app.draw_network()