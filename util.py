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
import glob

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.externals import joblib

#model file names
m = './mlp.pkl'
s = './svc.pkl'
#couldn't find an acceptable non-overfit rf model
#rf = './rf.pkl' 


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
    '''
    Generates realistic data with random noise for a given pin, then predicts 
    possible pin numbers from the generated data.
    
    Returns list of pin numbers and list of associated probabilities
    '''
    label_map = make_label_map()
    #pin number translated into number-number sequences
    seq = []
    for i in range(len(pin)-1):
        seq.append([pin[i],pin[i+1]])
    #generate pseudo-data for pin number
    acc_data = []
    for s in seq:
        acc_data.append(pseudo_data(distance(coords[s[0]],coords[s[1]])))
        
    acc_data=np.array(acc_data).reshape(3,24)    
    
    #Useful to see data, leave commented out otherwise
    #print(acc_data)
    
    #Print svc predictions first... they are garbage
    print('Support Vector Predictions : ')
    svc_pred_ind = predictSVC(acc_data)
    svc_pred = [label_map[i[0]] for i in svc_pred_ind]
    print(svc_pred)
    
    #import trained mlp
    mlp = joblib.load(m)
    
    #get probabilities from predicting on data from above
    probs = mlp.predict_proba(acc_data)
    
    #build list of possible pin numbers from probabilities
    possible = []
    prob_list = []
    
    #loop over each transition
    for i in range(len(probs)):
        #Get top probabilites and corresponding numbers
        poss = [label_map[n] for n in np.argpartition(probs[i], -10)[-10:]]
        poss_probs = [probs[i][n] for n in np.argpartition(probs[i], -10)[-10:]]
        possible.append(poss)
        prob_list.append(poss_probs)              
    
    #chain transitions based on sequence of digits (if first.last == next.first then chain and keep, otherwise drop)
    colapse_first = []
    prob_sum_first = []
    for i in range(len(possible[0])):
        for j in range(len(possible[1])):
            if (possible[0][i][1] == possible[1][j][0]):
                l1 = possible[0][i]
                l2 = possible[1][j]
                colapse_first.append([l1[0],l2[0],l2[1]])
                prob_sum_first.append(prob_list[0][i] * prob_list[1][j])
   
    #chain next level of digit transitions with first
    poss_pins = []
    prob_sums = []
    for i in range(len(colapse_first)):
        for j in range(len(possible[2])):
            if (colapse_first[i][2] == possible[2][j][0]):
                l1 = colapse_first[i]
                l2 = possible[2][j]
                poss_pins.append([l1[0],l1[1],l2[0],l2[1]])
                prob_sums.append(prob_sum_first[i] * prob_list[2][j])                      
                      
    #return possible pin numbers and model confidence (liklihood of pin)
    return poss_pins, prob_sums

def predictSVC(data):
    '''
    loads svc from pickle, returns predicted class svc model
    '''
    ret = []
    svc = joblib.load(s)
    for d in data:
        ret.append(svc.predict(d.reshape(1, -1)))
    return ret

def predictKnownPin_rf(pin, model_file):
    '''
    Generates realistic data with random noise for a given pin, then predicts 
    possible pin numbers from the generated data.
    
    Returns list of pin numbers and list of associated probabilities
    '''
    label_map = make_label_map()
    #pin number translated into number-number sequences
    seq = []
    for i in range(len(pin)-1):
        seq.append([pin[i],pin[i+1]])
    #generate pseudo-data for pin number
    acc_data = []
    for s in seq:
        acc_data.append(pseudo_data(distance(coords[s[0]],coords[s[1]])))
        
    acc_data=np.array(acc_data).reshape(3,24)    
    
    #Useful to see data, leave commented out otherwise
    #print(acc_data)
    
    #import trained mlp
    rfc = joblib.load(model_file)
    
    #get probabilities from predicting on data from above
    probs = rfc.predict_proba(acc_data)
    
    #build list of possible pin numbers from probabilities
    possible = []
    prob_list = []
    
    #loop over each transition
    for i in range(len(probs)):
        #Get top probabilites and corresponding numbers
        poss = [label_map[n] for n in np.argpartition(probs[i], -10)[-10:]]
        poss_probs = [probs[i][n] for n in np.argpartition(probs[i], -10)[-10:]]
        possible.append(poss)
        prob_list.append(poss_probs)              
    
    #chain transitions based on sequence of digits (if first.last == next.first then chain and keep, otherwise drop)
    colapse_first = []
    prob_sum_first = []
    for i in range(len(possible[0])):
        for j in range(len(possible[1])):
            if (possible[0][i][1] == possible[1][j][0]):
                l1 = possible[0][i]
                l2 = possible[1][j]
                colapse_first.append([l1[0],l2[0],l2[1]])
                prob_sum_first.append(prob_list[0][i] * prob_list[1][j])
   
    #chain next level of digit transitions with first
    poss_pins = []
    prob_sums = []
    for i in range(len(colapse_first)):
        for j in range(len(possible[2])):
            if (colapse_first[i][2] == possible[2][j][0]):
                l1 = colapse_first[i]
                l2 = possible[2][j]
                poss_pins.append([l1[0],l1[1],l2[0],l2[1]])
                prob_sums.append(prob_sum_first[i] * prob_list[2][j])                      
                      
    #return possible pin numbers and model confidence (liklihood of pin)
    return poss_pins, prob_sums

    
