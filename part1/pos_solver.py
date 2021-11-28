###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    # def get_file_data(filename):
    def __init__(self):
        self.corpus = {}
        self.prob=['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        self.POS_prob = {'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0}

    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "HMM":
            return -999
        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        for each_sentence in data:
            for i in range(len(each_sentence[0])):
                if not each_sentence[0][i] in self.corpus.keys():
                    self.corpus[each_sentence[0][i]] = {}
                    for j in self.prob:
                        self.corpus[each_sentence[0][i]][j] = 0
                self.corpus[each_sentence[0][i]][each_sentence[1][i]] = self.corpus[each_sentence[0][i]][each_sentence[1][i]] + 1
                self.POS_prob[each_sentence[1][i]] = self.POS_prob[each_sentence[1][i]] + 1

        self.getTEmissionProbabilities() 
        self.calcTransitionProbabilities(data)
        # print('POS_prob - ',self.POS_prob)  
        pass

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        
        POS_list = []
        for i in sentence:
            if i not in self.corpus.keys():
                #words not in corpus? - feature func - linguistic rules - first letter capital...
                POS_list.append('noun')
            else:
                max_key = max(self.corpus[i], key=self.corpus[i].get)
                POS_list.append(max_key)
           
        return POS_list

    def getTEmissionProbabilities(self):
        emiss_p = {}
       
        for i in self.corpus:
            if i not in emiss_p.keys():
                emiss_p[i] = {}
            for j in self.prob:
                emiss_p[i][j] = self.corpus[i][j]/self.POS_prob[j]
             
        return emiss_p

    def calcTransitionProbabilities(self,data):
        transition_p = {'s':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'e':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0}}
        for each_sentence in data:
            for i in range(len(each_sentence[0])-1):
                if i==0:
                    transition_p['s'][each_sentence[1][i]] = transition_p['s'][each_sentence[1][i]] + 1
                if each_sentence[1][i] not in transition_p.keys():
                    transition_p[each_sentence[1][i]] = {'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0}
                transition_p[each_sentence[1][i]][each_sentence[1][i+1]] = transition_p[each_sentence[1][i]][each_sentence[1][i+1]] + 1
                if i==len(each_sentence[0])-2:
                    transition_p['e'][each_sentence[1][i]] = transition_p['e'][each_sentence[1][i]] + 1

        for i in transition_p.keys():
            sum_key = sum(transition_p[i].values())
            for j in transition_p[i].keys():
                transition_p[i][j] = transition_p[i][j]/sum_key


    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)

    def complex_mcmc(self, sentence):
        return [ "noun" ] * len(sentence)



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

