#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2020)
#

# from _typeshed import _T_contra
from PIL import Image, ImageDraw, ImageFont
import sys
import copy
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

#Just used to convert list to a single string
def convert(s):
    new = ""
    for x in s:
        new += x 
    return new

#Used to read the image and write it in pixel format using '*
def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

#Used to write each character in pixel format using '*'
def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# Used to compare each character of train image list with each character of test image list by calculating  the number of 
# matched, mismatched and empty spaces.
def compare_each(given_key, given_value, train):
    matched,mismatched,empty=0,0,0
    for i in range(len(given_value)):
        if given_value[i]==' ' and train[i]==' ':
            empty=empty+1
        elif given_value[i]==train[i]:
            matched=matched+1
        elif given_value[i]!=train[i]:
            mismatched=mismatched+1
    return (given_key,matched,mismatched,empty)

# used to compare the test image list and the train image list
def compare(given_key, given_value, train):
    character,matched, mismatched,empty=given_key,0,0,0
    for i in range (len(given_value)):
        char,match,mismatch,space=compare_each(given_key,given_value[i],train[i])
        matched=match+matched
        mismatched=mismatched+mismatch
        empty=space+empty
    return(character,matched,mismatched,empty)

#This function is used to calculate the probability value by taking 0.8 probability for matched, 0.05 for mismatched and 0.15 for the empty spaces
def probability(character,given_value, train):
    char,match,mismatch,empty=compare(character, given_value, train)
    value= match*0.80+mismatch*0.05+empty*0.15
    return (char, value)

# Used to calculate probability of the values (sum=1)
def prob(emm):
    for i in emm.keys():
        sum_key = sum(emm[i].values())
        for j in emm[i].keys():
            if sum_key!=0:
                emm[i][j] = emm[i][j]/sum_key
    return emm

#This function is used to read the data using Simple Bayes Net model
def Simple(given, training):
    x=[]
    char=[]
    answer=[]
    emmission={}
    final={}   
    for i in range (len(training)):
        x=[]
        char=[]
        for j in range (len(given)):
            given_key= list(given.keys())[j]
            given_value= list(given.values())[j]
            character,value = probability(given_key, given_value, training[i]) 
            char.append(character)
            x.append(value)
            final[character]=value            
        final_index=x.index(max(x))
        z=char[final_index]
        answer.append(z)
    emmission=prob(emmission)
    return answer

#------------------------------------------------------------------------------------------------------------#

# Used to read the training text file and eleminate few characters which are not used in the image extracion
def read_data(fname):
    x=[]
    exemplars = []
    file = open(fname, 'r');

    for line in file:
        data = [w for w in line.split()]
        exemplars += [data[0::2],]
   
    for i in range(len(exemplars)):
        y=exemplars[i] 
        for j in range (len(exemplars[i])) :
            x.append(y[j]) 
    y=[]       
    y = [s.replace('`', "") for s in x]
    y = [s.replace('@', "") for s in y]
    y = [s.replace('+', "") for s in y]
    y = [s.replace('=', "") for s in y]
    y = [s.replace('_', "") for s in y]
    y = [s.replace(',', "") for s in y]
    y = [s.replace('{', "") for s in y]
    y = [s.replace('}', "") for s in y]
    y = [s.replace('+', "") for s in y]
    y = [s.replace('@', "") for s in y]
    y = [s.replace('#', "") for s in y]
    y = [s.replace('$', "") for s in y]
    y = [s.replace('&', "") for s in y]
    y = [s.replace('^', "") for s in y]
    y = [s.replace('*', "") for s in y]
    y = [s.replace(':', "") for s in y]
    y = [s.replace('~', "") for s in y]
    y = [s.replace('/', "") for s in y]
    y = [s.replace(':', "") for s in y]
    y = [s.replace('[', "") for s in y]
    y = [s.replace(']', "") for s in y]
    y = [s.replace('<', "") for s in y]
    y = [s.replace('>', "") for s in y]
    y = [s.replace(';', "") for s in y]
    y = [s.replace('%', "") for s in y]
    res = list(' '.join(y))
    
    return res

# Used to calculate the emmission probability of test image
def emmission_probability(given, training): 
    x=[]
    char=[]
    answer=[]
    emmission={}
    final={'A':{},'B':{},'C':{},'D':{},'E':{},'F':{},'G':{},'H':{},'I':{},'J':{},'K':{},'L':{},'M':{},'N':{},'O':{},'P':{},'Q':{},'R':{},'S':{},'T':{},'U':{},'V':{},'W':{},'X':{},'Y':{},'Z':{},'a':{},'b':{},'c':{},'d':{},'e':{},'f':{},'g':{},'h':{},'i':{},'j':{},'k':{},'l':{},'m':{},'n':{},'o':{},'p':{},'q':{},'r':{},'s':{},'t':{},'u':{},'v':{},'w':{},'x':{},'y':{},'z':{},'0':{},'1':{},'2':{},'3':{},'4':{},'5':{},'6':{},'7':{},'8':{},'9':{},'(':{},')':{},',':{},'.':{},'-':{},'!':{},'?':{},'\\':{},'"':{},"'":{},' ':{}}
    for i in range (len(given)):
        x=[]
        char=[]
        given_key= list(given.keys())[i]
        given_value= list(given.values())[i]
        for j in range(len(training)):
            character,value = probability(given_key, given_value, training[j]) 
            final[character][j]=value
    return final

# Used to calculate the transission probability of test image
def getTransitionProbabilities(data):
    X="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    Y=[char for char in X] #To convert to list
    transition_p={}
    d = dict(zip(X,[0 for x in range(0,len(X))]))
    for i in d.keys():
        if i not in transition_p.keys():
            transition_p[i] = copy.deepcopy(d)

    for k in range(len(data)-1):
        transition_p[data[k]][data[k+1]] = transition_p[data[k]][data[k+1]] + 1

    for i in transition_p.keys():
        sum_key = sum(transition_p[i].values())
        for j in transition_p[i].keys():
            if sum_key!=0:
                transition_p[i][j] = transition_p[i][j]/sum_key   
    return transition_p

# Used to count number of characters
def count(list,char):
    count=0
    for i in list:
        if i == char:
            count = count + 1
    return count

# This function is used to read the data using HMM Viterbi 
def viterbi(file,given,training):
    X="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    Y=[char for char in X]
    Z={}
    initial={}
    
    d = dict(zip(X,[0 for x in range(0,len(X))]))
    for i in d.keys():
        if i not in Z.keys():
            Z[i] = d   
    fil = read_data(file)

    # Calculate observation list
    observation=[]
    
    for i in range(len(training)):
        observation.append(i)
    emission_p=emmission_probability(given, training) 
    N=len(observation)
    for i in X:
        initial[i]=1/73
    i_states = tuple(X,)
    transsion_p=getTransitionProbabilities(fil)
    for i in X:
        for j in X:
            if transsion_p[i][j] != 0.0:
                transsion_p[i][j]= (math.log(transsion_p[i][j]))
            else:
                transsion_p[i][j]=0.00000000000000000001

    V_table = {i:[0] * N for i in i_states}
    table = {i:[0] * N for i in i_states}

    for s in i_states:
        V_table[s][0] = initial[s] * emission_p[s][observation[0]]

    for i in range(1, N):
        for s in i_states:
            (table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] + ((1/10)*transsion_p[s0][s])) for s0 in i_states ], key=lambda l:l[1] ) 
            V_table[s][i] *= emission_p[s][observation[i]]

    # Backtrack
    viterbi_seq = [""] * N
    temp = [(state, V_table[state][N-1]) for state in i_states]
    viterbi_seq[N-1], _ = max(temp, key=lambda array:array[1])
    for i in range(N-2, -1, -1):
        viterbi_seq[i] = table[viterbi_seq[i+1]][i+1]
        
    return viterbi_seq



#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:
print("Simple: " + convert(Simple(train_letters,test_letters)))
print("   HMM: " + convert(viterbi(train_txt_fname,train_letters,test_letters))) 
