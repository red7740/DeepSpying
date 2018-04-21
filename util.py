#Utility file of functions and imports
#Doles, Nix, Terlecky
#File includes standard imports and defined functions used in multiple project files
#
#
import random
import itertools
import numpy as np
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib

#mlp pkl file 
m = './mlp.pkl'

#Define coordinate system for the number pad
#With coords defined as dictionary of digit : position
num_pos=[[0,3],[1,3],[2,3],[0,2],[1,2],[2,2],[0,1],[1,1],[2,1],[1,0]]
nums = [1,2,3,4,5,6,7,8,9,0]
coords = {}

for i in range(len(nums)):
    coords[nums[i]] = num_pos[i]

    
def distance(start,finish):
        '''
        Returns delta_x, delta_y given starting and ending coords
        '''
        dist_x=finish[0]-start[0]
        dist_y=finish[1]-start[1]
        #before_square=(dist_x)**2+(dist_y)**2
        #final=np.sqrt(before_square)
        return(dist_x,dist_y)


def pseudo_data(trans):
    '''
    returns simulated accelerometer data for a single transition with noise
    
    usage: trans is manhattan distance btwm numbers in keypad as returned by distance()
    '''
    def noise():
        return random.random()*.1
    
    test = []
    for i in range(12):
        test.append([trans[0]*.3+noise(),trans[1]*.3+noise()])
    return test    

def make_label_map():
    label_map = {}
    loop_count = 0
    for i in range(10):
        for j in range(10):
            label_map[loop_count] = [i,j]
            loop_count+=1
    return label_map

def predictKnownPin(pin):
    label_map = make_label_map()
    seq = []
    for i in range(len(pin)-1):
        seq.append([pin[i],pin[i+1]])
    
    acc_data = []
    for s in seq:
        acc_data.append(pseudo_data(distance(coords[s[0]],coords[s[1]])))
        
    acc_data=np.array(acc_data).reshape(3,24)    
    
    mlp = joblib.load(m)
    
    probs = mlp.predict_proba(acc_data)

    possible = []
    for i in range(len(probs)):
        poss = [label_map[n] for n in np.argpartition(probs[i], -5)[-10:]]
        possible.append(poss)
    
    colapse_first = []
    for l1 in possible[0]:
        for l2 in possible[1]:
            if (l1[1] == l2[0]):
                colapse_first.append([l1[0],l2[0],l2[1]])
    
    poss_pins = []
    for l1 in colapse_first:
        for l2 in possible[2]:
            if (l1[2] == l2[0]):
                poss_pins.append([l1[0],l1[1],l2[0],l2[1]])
    return poss_pins
