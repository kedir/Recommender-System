##############################################################

#  Movie Recommender System using Stacked Auto Encoder

###############################################################




# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
# creating Stack Auto Encoder using inheritance.
class SAE(nn.Module):
    def __init__(self, ): # define arcitecture of auto encoder
        super(SAE, self).__init__() # self is object of SAE class. Super is used to get the method from module class
        self.fc1 = nn.Linear(nb_movies, 20) # fc1 is first full connection . nb_movies represent input, 20 represnt no. of nurons in hidden layers.
        self.fc2 = nn.Linear(20, 10) # encoding
        self.fc3 = nn.Linear(10, 20) # decoding 
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x): # encoding and decoding take place here
        x = self.activation(self.fc1(x)) # new encoded vector in the first hidden layer.
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x)) # decoding 
        x = self.fc4(x) # decoding without activation function
        return x
sae = SAE()
criterion = nn.MSELoss() # defining mean square error
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # lr is learning rate, weight_decay is used to reduce learning rate after few epochs inorder to regulate the convergence.

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone() # simply copy input . vector of real rating
        # target.data takes all rating done by the user
        if torch.sum(target.data > 0) > 0: # to optimize the memory .i.e to look at least if the user rating 1 movie.
            output = sae(input) # vector of predicing rating.
            target.require_grad = False # don't compute gradient with respect to target that saves memory
            output[target == 0] = 0 # will not allowed in the computation of error.i.e updating weight not consider this value.
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # to ensure denominator is != zero we add 1e-10.
            loss.backward() # backword() method tells in which direction(increase or decrease) to update the weight.i.e to increase or decrease the weight
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step() # decide the amount by which update (the intensity)
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0. # represent number of user who rated at least one movie.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # in pyTorch like keras it doesn't accept a single vector so we must expand a dimention using unsqueeze.
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))
