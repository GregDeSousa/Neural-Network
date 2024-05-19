import numpy as np
#Layer maker---
#Group 29
#u22492477-Gregory De Sousa
#u22511700-Daniel Banze


hidden_layers=[]#ulti dimentional
neurons_per_layer=7#number of nodes per hidden layer
connections=12
EPOCH=100
N=10
layers_hidden=1
Input_Data=[]

def Activation_function(value):#Sigmoid activation function
    return 1 / (1 + np.exp(-value))

def Activation_prime(value):#Derivative of sigmoid activation function
    sig_x = Activation_function(value)
    return sig_x * (1 - sig_x)
def update_stuff(network):# Used to update variables needed in back propagation using the Neuron class seen below.Advised that the neuron class and backpropagation is read first
    Z=[]*(network.hidden_size+1)#pre-activation sum for all layers exluding output
    A=[]*(network.hidden_size+1)#activation for all layers excluding output
    F_primeZ=[]*(network.hidden_size+1)#Pre activation pass through derivative activation function. Used for backwards propagating gradient descent
    for q in range(network.hidden_size+1):# for each layer after the input
        if q!=network.hidden_size:#If not the output layer
            arr1=[]#storage arrays
            arr2=[]#---
            arr3=[]#---
            for r in range(neurons_per_layer+1):
                if r!=neurons_per_layer:#For all neurons in a layers except bias neuron
                    arr1.append(network.layer[q][r].sum)
                    arr2.append(network.layer[q][r].Activation)
                    arr3.append(Activation_prime(network.layer[q][r].sum))
                else:#if bias neuron
                    arr1.append(1)#Append bias neuron value to pre-activation sums
                    arr2.append(1)#Append bias neuron value to activation

            Z.append(np.array(arr1))
            A.append(np.array(arr2))
            F_primeZ.append(np.array(arr3))
        else:#output layer
            arr1=[]
            arr2=[]
            arr3=[]
            for r in range(4):#For the 3 outputs and bias neuron. Bias neuron included in output layer to ensure correct matrix multiplication in backpropagation.
                if r!=3:
                    arr1.append(network.layer[q][r].sum)#Z
                    arr2.append(network.layer[q][r].Activation)#A
                    arr3.append(Activation_prime(network.layer[q][r].sum))#A_prime
            Z.append(np.array(arr1))
            A.append(np.array(arr2))
            F_primeZ.append(np.array(arr3))
            # print(Z[0])
            # print(A[0])
            # print(F_primeZ[0])
    return Z,A,F_primeZ#These matrices are returned back to backpropagation
class Neuron:#Neuron class dictating the structure of the neuron
    def __init__(self,is_bias,value,connections):
        self.Activation=0
        if not is_bias:#Creation of regular neuron as opposed to a bias neuron
            self.weights=[]*connections#connections represent the number of nodes in the previous layer
            self.sum=0#Pre-activation sum
            for x in range(connections):#Randomly initialise the weights
                self.weights.append(np.random.uniform(-(1/np.sqrt(neurons_per_layer)),(1/np.sqrt(neurons_per_layer))))
        else:
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

def backpropagation(input_data,target_data,weights_input,weights_hidden,weights_output,learning_rate,epoch,loss_convergence):
    N=len(target_data)#size of the sample data
    inp_add=np.array([1]*N)#for adding bias neuron to the input
    input_data=np.hstack((np.array(input_data),inp_add.reshape(-1,1)))#vectorize input data
    Loss_vector=np.zeros(epoch)#initiliase loss vector
    for i in range(epoch):#For each epoch
        if i!=0:#excluding the first epoch--
            if Loss_vector[i-1]<=loss_convergence:#if the previous L2 loss exceeds threshold, halt execution
                break

        for x in range(N):#for each data sample in the batch
            NeuralNet.load(weights_input.transpose(), weights_hidden.transpose(), weights_output.transpose())#update the Neural network classes
            Y=forward_propagation(NeuralNet,input_data[x])#propagate the input
            Z, A, F_primeZ = update_stuff(NeuralNet)#see update for further info on these variables
            Loss_sum=0#initialise loss sum
            loss_sample=np.zeros(3)#initialise Y_hat vector
            for ind in range(3):#summate the total loss of the output vector
                Loss_sum+=target_data[x][ind]-Y[ind]
                loss_sample[ind]=target_data[x][ind]-Y[ind]#used for Loss calcs
            if x==1:#second sample, hence L2 loss
                Loss_vector[i]=(np.power(Loss_sum,2)*0.5)#L2 loss calculation
            delta=[]*(NeuralNet.hidden_size+1)#delta list of delta matrices of each layer and calculating gradient descent
            for y in range(NeuralNet.hidden_size+1):# for every layer after the input

                if y==0:#output layer
                    delta_n=(-loss_sample)*F_primeZ[NeuralNet.hidden_size]#output delta matrix
                    delta.append(delta_n)#append to list of delta matrices
                    dJdW=np.dot((A[NeuralNet.hidden_size-1]).reshape(1, -1).transpose(),delta_n.reshape(1, -1))#calculate gradient
                    weights_output=weights_output-(learning_rate*dJdW)#update output weight
                elif y==NeuralNet.hidden_size:#input layer
                    if NeuralNet.hidden_size>1:#if more than one hidden layer in the network
                        delta_n=np.dot((delta.pop().reshape(1,-1)),np.delete((weights_hidden.transpose()),-1,axis=1).transpose())*F_primeZ[0]
                    else:#if a single hidden layer in the network
                        u=delta.pop()
                        delta_n=(u.reshape(1,-1)@np.delete((weights_output.transpose()),-1,axis=1))*F_primeZ[0]
                    delta.append(delta_n)
                    dJdW = np.dot(np.array(input_data[x]).reshape(1,-1).transpose(),delta_n.reshape(1,-1))#
                    weights_input = weights_input-(learning_rate*dJdW)#update input weights
                else:#hidden layers
                    delta_n=np.dot((delta.pop()).reshape(1,-1),np.delete((weights_output).transpose(),-1,axis=1))*F_primeZ[NeuralNet.hidden_size-y]
                    delta.append(delta_n)
                    dJdW=np.dot((A[NeuralNet.hidden_size-y]).reshape(1, -1).transpose(),delta_n.reshape(1, -1))

                    weights_hidden = weights_hidden-(learning_rate*dJdW)

    if weights_hidden.any():#if the the was no hidden to hidden layers
        return weights_input,weights_hidden,weights_output,Loss_vector
    else:
        return weights_input,weights_hidden,weights_output,Loss_vector
def Run(inp_dat,out_dat):#for converting classes into vectorized arrays for backpropagation
    Input_Data,Target_Data=inp_dat,out_dat#input data for backprop and target data for backprop
    Layers=NeuralNet.layer
    Weight_in=np.zeros((neurons_per_layer,13))#weights from input to hidden
    Weight_hidden=np.empty((0,0))#weights from hidden to hidden
    Weight_output=np.zeros((3,(neurons_per_layer+1)))#weights from output to hidden
    for x in range(NeuralNet.hidden_size+1):#for every layer after input layer
        #For detailed info on functionality of the inner loops, refer to update_stuff or foward_propagation
        if x==0:#input to hidden--converts weigths and bias values in the neural network class into weight_in
            for i in range(neurons_per_layer):
                for j in range(13):
                    if j==12:
                        Weight_in[i][j]=NeuralNet.bias[x][i]
                    else:
                        Weight_in[i][j]=Layers[x][i].weights[j]
        elif x==(NeuralNet.hidden_size):#hidden to output----converts weigths and bias values in the neural network class into weight_out
            for i in range(3):
                for j in range(neurons_per_layer+1):
                    if j==neurons_per_layer:
                        Weight_output[i][j]=NeuralNet.bias[x][i]
                    else:
                        Weight_output[i][j]=Layers[x][i].weights[j]
        else:#hidden to hidden----converts weigths and bias values in the neural network class into weight_hidden
            Weight_hidden=np.zeros((neurons_per_layer,(neurons_per_layer + 1)))
            for i in range(neurons_per_layer):
                for j in range(neurons_per_layer + 1):
                    if j==neurons_per_layer:
                        Weight_hidden[i][j]=NeuralNet.bias[x][i]
                    else:
                        Weight_hidden[i][j]=Layers[x][i].weights[j]

    w_i, w_h, w_o, L_v=backpropagation(Input_Data,Target_Data,Weight_in.transpose(),Weight_hidden.transpose(),Weight_output.transpose(),0.8,EPOCH,0.0001)
    return  w_i, w_h, w_o, L_v,NeuralNet
if input == '':
    NeuralNet = Layer(layers_hidden)#intialise neural network class
    history = ['X']*4
    output = np.random.choice(['R', 'P', 'S'])
    input_data = []
    target_data = []
    data=[]
    num=0
    rounds=1#current round against agent
elif history[0]!='X':
    rounds+=1#increase the rounds
    history.pop(0)
    history.append(input)
    conc_input = []
    for i in history:#normalise target data into 1-of-3 hot encoding for each segement of history... input is thus a concatention of 12
        #this will be used to evaluate loss in backprop
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
    out=np.array(forward_propagation(NeuralNet,conc_input))
    data.append(conc_input)
    conc_targ=[]
    if input=="R":#normalize desired output of 1-of-3 hot encoding based on the previous move of the agent
        conc_targ.append(0)
        conc_targ.append(1)#paper
        conc_targ.append(0)
    elif input=="P":
        conc_targ.append(0)
        conc_targ.append(0)
        conc_targ.append(1)#scissors
    elif input=="S":
        conc_targ.append(1)#rock
        conc_targ.append(0)
        conc_targ.append(0)
    else:
        print("error 1")

    move=np.argmax(out)#output is the highest of the 3 output neurons

    if move==0:#if first neuron is most activated
            output="R"
    elif move==1:#if second neuron is the most activated
            output="P"
    elif move==2:#if third neuron is the most activated
            output="S"
    else:
            print("error 2")

    if num==10:#num dictates how many rounds untill the network backpropagates
        Run(input_data,target_data)#discussed in Run
        target_data=[]
        input_data=[]
        num=0
    elif rounds>2:#after the second round the previous input is used to compare with the target data as input is the previous rounds move
        if len(data)>1:#if weve played atleast twice
            input_data.append(data[len(data)-2])#append the previous output played as our agent only gets information about the preivious round
        target_data.append(conc_targ)#append target data
        num+=1
    else:
        pass

    history.pop(0)
    history.append(output)
else:#if history has not filled completely to normalise input, play randomly
    history.pop(0)
    history.append(input)
    history.pop(0)
    history.append(output)
    output = np.random.choice(['R', 'P', 'S'])







