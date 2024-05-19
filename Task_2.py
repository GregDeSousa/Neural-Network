import numpy as np
#Layer maker---
#Group 29
#u22492477-Gregory De Sousa
#u22511700-Daniel Banze



neurons_per_layer=7#Neurons for each hidden layer
EPOCH=100#Stopping condition for epochs iterations for the dataset
connections=12#Used for the weights initialised from the first hidden layer
N=1000#size of the total batch of data sampled
layers_hidden=1#Amount of hidden layers
Input_Data=[]#input for backpropagation
def Activation_function(value):#Sigmoid activation function
    return 1 / (1 + np.exp(-value))

def Activation_prime(value):#Derivative of sigmoid activation function
    sig_x = Activation_function(value)
    return sig_x * (1 - sig_x)

class Neuron:#Neuron class dictating the structure of the neuron
    def __init__(self,is_bias,value,connections):
        self.Activation=0
        if not is_bias:#Creation of regular neuron as opposed
            self.weights=[]*connections#connections represent the number of nodes in the previous layer
            self.sum=0#Pre-activation sum
            for x in range(connections):#Randomly initialise the weights
                self.weights.append(np.random.uniform(-(1/np.sqrt(neurons_per_layer)),(1/np.sqrt(neurons_per_layer))))
        else:#conceptually creates bias neuron,although actication is zero as will be seen further in the code its value would actually be 1
            self.weights=None
            self.Activation=0


    def activate(self):#Used in foward propagation to activate the node with its preactivation sum.
        self.Activation = Activation_function(self.sum)
        return self.Activation

class Layer():#Layer class holding the layers and subsequent neurons in the neural network
    def __init__(self, num_hidden):#num hidden is the amount of hidden layers
        self.layer = []*(num_hidden+1)#The layer array holds all layers after.(+1 is accomidating the output layer aswell)
        self.hidden_size = num_hidden
        self.bias = []*(num_hidden+1)#Array that holds bias values for the neuron
        for i in range (num_hidden+1):#For every layer after the input layer
            if i != (num_hidden):#If not the output layer
                self.layer.append([]*neurons_per_layer)#make a layer the size specified at the top of the document
                self.bias.append([]*neurons_per_layer)#correspending bias values
                for k in range(neurons_per_layer):#Traverse the layer
                    if i == 0:#Generate neurons
                        self.layer[i].append(Neuron(False, 1, 12))#For input layer
                    else:
                        self.layer[i].append(Neuron(False, 1, neurons_per_layer))#For the rest of the layers
                    self.bias[i].append(1)#initialse bias
            else:#output layer same logic applies except size of 3(one-hot encoding)
                self.layer.append([] * 3)
                self.bias.append([]*neurons_per_layer)
                for k in range(3):
                    if i == 0:
                        self.layer[i].append(Neuron(False, 1, 12))
                    else:
                        self.layer[i].append(Neuron(False, 1, neurons_per_layer))
                    self.bias[i].append(1)
    def load(self,weight_in,weight_hidden,weight_output):#Updates the classes for the neural network
        for x in range(self.hidden_size + 1):
            if x == 0:#input layer
                for i in range(neurons_per_layer):#for each connection
                    for j in range(13):#for each neuron
                        if j == 12:
                            self.bias[x][i]= weight_in[i][j]#update biases based on bias neuron weight calculated in backpropagation
                        else:
                            self.layer[x][i].weights[j] = weight_in[i][j]#update neuron weights
            elif x == (self.hidden_size):#output layer--same as input layer just different indexes and sizes
                for i in range(3):
                    for j in range(neurons_per_layer + 1):
                        if j == neurons_per_layer:
                            self.bias[x][i]= weight_output[i][j]
                        else:
                            self.layer[x][i].weights[j] = weight_output[i][j]
            else:#hidden layers-same as input layer just different indexes and sizes
                for i in range(neurons_per_layer):
                    for j in range(neurons_per_layer + 1):
                        if j == neurons_per_layer:
                            self.bias[x][i]= weight_hidden[i][j]
                        else:
                            self.layer[x][i].weights[j] = weight_hidden[i][j]
def forward_propagation(hidden_layer,current_input):#current_input refers to a singular input given to obtain a singular output
    Activation=[]*(hidden_layer.hidden_size+1)#Array housing activation for each node, for every layer. Other dimentionality added in loops below
    for x in range(hidden_layer.hidden_size+1):#For every layer after the input layer
        current_layer=hidden_layer.layer[x]#extracting parameters from the classes
        bias = hidden_layer.bias#---refer to layer and neuron for more detail
        if x==0:#Input layer
            Activation.append([] * 12)
            for y in range(neurons_per_layer):#for each neuron in the layer
                sum=0
                for z in range(12):#for every connection
                    sum+=(current_layer[y].weights[z])*current_input[z]#Calculate preactivation sum for the neuron by using each weight multiplied with input
                current_layer[y].sum=sum+bias[x][y]#preactivation sum given to neuron
                Activation[x].append(current_layer[y].activate())#neuron activated and value appended to array for storage and further calculation
        elif x==hidden_layer.hidden_size:#for output layer--same as input layer just different indexes and sizes
            Activation.append([] * 3)
            for y in range(3):
                sum = 0
                for z in range(neurons_per_layer):
                    sum += (current_layer[y].weights[z])*Activation[x-1][z]
                current_layer[y].sum = sum + bias[x][y]
                Activation[x].append(current_layer[y].activate())
        else:#hidden layers--same as input layer just different indexes and sizes
            Activation.append([] * neurons_per_layer)
            for y in range(neurons_per_layer):
                sum=0
                for z in range(neurons_per_layer):
                    sum+=(current_layer[y].weights[z])*Activation[x-1][z]
                current_layer[y].sum=sum+bias[x][y]
                Activation[x].append(current_layer[y].activate())
    output=np.zeros(3)#output vector
    for k in range(3):#append output layer to output vector
        output[k]=hidden_layer.layer[hidden_layer.hidden_size][k].Activation
    return output#return the output for the given input


if input == '':
    weight_in = np.array([[-1.47412942, 2.35303458, 0.39211827, -4.25774658, 1.84122852, 3.17562709, 1.91861101, 2.54540777, -3.5350058, -4.90324629, 2.58423246, 3.02324116, 2.26530956],
        [4.43759371, -4.49445584, 0.14832345, 0.43575486, -0.79245356, -0.32420631, -3.76729169, 4.53496786, -2.20758165, -0.72504159, -4.11659767, 3.67790251, 0.2828119],
        [1.97978394, 1.29399031, -1.46767119, 0.51481527, 1.04560881, 0.04432747, 1.37252405, -0.84722752, 0.23407872, -0.68709317, 1.19279841, 0.32881557, 2.1172374],
        [-0.56761307, -3.71542606, 4.43125888, -0.61734808, -0.67995878, 1.36715327, 3.21771024, 3.00326295, -6.26640353, -3.40689443, 0.39795855, 3.30776678, 1.34125541],
        [1.0622282, 0.57632374, 0.07489783, 4.15582641, 1.945516, -4.68583975, 3.26075431, -4.74162586, 2.35279609, 2.40220207, 4.67741822, -5.55824685, 2.3563494],
        [1.937417, -1.57822946, 0.59617169, 1.84756965, -4.20801885, 3.77776661, -3.77798268, 1.63848435, 3.56292332, 4.36682671, -6.12282144, 3.14657201, 2.76698899],
        [0.28914325, 1.30540841, 0.576176, -1.52860018, 2.5141441, 0.8702201, 0.84922887, -0.96762079, 1.21549943, 3.12533925, -1.18748788, -0.16787823, 2.64140212]])

    weight_out = np.array([[5.77913772, 6.47315608, -3.2308303, 0.41509936, -5.91279108, 3.82956453, -4.02006887, -0.71958254],
        [-5.34867564, -1.29732171, -2.20754883, -8.1965663, 5.6437278, 6.97845218, 0.17666661, -1.20052947],
        [4.78662235, -7.32660164, -2.48941624, 5.81732858, 4.48036125, -6.04291266, -1.14673696, -2.82477809]])
    NeuralNet = Layer(layers_hidden)#intialise neural network class
    NeuralNet.load(weight_in, np.empty((3, 3)), weight_out)#load the preset values into the network
    history = ['X']*4
    output = np.random.choice(['R', 'P', 'S'])
elif history[0]!='X':
    history.pop(0)
    history.append(input)
    input_data = []
    conc_input = []
    for i in history:#normalise data into 1-of-3 hot encoding for each segement of history... input is thus a concatention of 12
        if i == "R":
            conc_input.append(1)
            conc_input.append(0)
            conc_input.append(0)
        elif i == "P":
            conc_input.append(0)
            conc_input.append(1)
            conc_input.append(0)
        elif i == "S":
            conc_input.append(0)
            conc_input.append(0)
            conc_input.append(1)
        else:
            print("big error")
    out=np.array(forward_propagation(NeuralNet,conc_input))#output from fowardpropagation
    move=np.argmax(out)#output is the highest of the 3 output neurons

    if move==0:#if first neuron is most activated
            output="R"
    elif move==1:#if second neuron is the most activated
            output="P"
    elif move==2:#if third neuron is the most activated
            output="S"
    else:
            print("bad computer")

    history.pop(0)
    history.append(output)
else:#if history has not filled completely to normalise input, play randomly
    history.pop(0)
    history.append(input)
    history.pop(0)
    history.append(output)
    output = np.random.choice(['R', 'P', 'S'])







