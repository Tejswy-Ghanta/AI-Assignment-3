###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
import random
import math

import copy

random.seed(94)
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    
    # def get_file_data(filename):
    def __init__(self):
        self.corpus = {}
        self.prob=['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        self.POS_prob = {'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0}
        self.e_pr = {}
        self.t_pr  ={}
        self.grandParentWordTag = {}
        self.grandParentPOS = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        sum = 0
        total_POS = 0
        for i in self.POS_prob.values():
            total_POS = total_POS+i

        for word in list(sentence):
            for pos in label:
                if word in self.corpus.keys() and pos in self.corpus[word].keys() and self.corpus[word][pos]!=0 and self.POS_prob[pos]!=0 :
                    sum = sum + (math.log10((self.corpus[word][pos]/self.POS_prob[pos])) + math.log10(self.POS_prob[pos]/total_POS)) 
        return sum

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

        self.e_pr = self.getEmissionProbabilities() 
        self.t_pr =self.getTransitionProbabilities(data)
        self.grandParentWordTag = self.getParentWordProbability(data)
        self.grandParentPOS = self.getGrandparentPOSProbability(data)
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

    def getEmissionProbabilities(self):
        emiss_p = {}
       
        for i in self.corpus:
            if i not in emiss_p.keys():
                emiss_p[i] = {}
            for j in self.prob:
                if self.POS_prob[j] == 0:
                    emiss_p[i][j] = self.corpus[i][j]
                else:
                    emiss_p[i][j] = self.corpus[i][j]/self.POS_prob[j]
             
        return emiss_p

    def getTransitionProbabilities(self,data):
        transition_p = {'adj':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'adv':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'adp':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},
        'conj':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'det':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'noun':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'num':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'pron':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'prt':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},
        'verb':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'x':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'x':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'.':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0}}
        for each_sentence in data:
            for i in range(len(each_sentence[0])-1):
                transition_p[each_sentence[1][i]][each_sentence[1][i+1]] = transition_p[each_sentence[1][i]][each_sentence[1][i+1]] + 1

        for i in transition_p.keys():
            sum_key = sum(transition_p[i].values())
            for j in transition_p[i].keys():
                if sum_key!=0:
                    transition_p[i][j] = transition_p[i][j]/sum_key

        return transition_p

    # def dptable(self,V,observed):

    #     for state in V[0]:
    #         yield"%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
        # print("%40s: %s " % ("Observed sequence", str(observed)))
        # N=len(observed)

    def hmm_viterbi(self, sentence):
        # problem definition
        states = ('adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.')
        trans_p = self.t_pr
        emit_p = self.e_pr
        start_p = {'adj':1/12, 'adv':1/12, 'adp':1/12, 'conj':1/12, 'det':1/12, 'noun':1/12, 'num':1/12, 'pron':1/12, 'prt':1/12, 'verb':1/12, 'x':1/12, '.':1/12}
        observations = [i for i in sentence]
        
        V = [{}]
        for st in states:
            if observations[0] not in emit_p.keys():
                V[0][st] = {"prob": start_p[st], "prev":None}
            else:
                V[0][st] = {"prob": start_p[st] * emit_p[observations[0]][st], "prev":None}

        for t in range(1, len(observations)):
            V.append({})

            for st in states:
                max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
                prev_st_selected = states[0]

                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                if observations[t] not in emit_p.keys():
                    max_prob = max_tr_prob
                else:
                    max_prob = max_tr_prob * emit_p[observations[t]][st]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        # for line in self.dptable(V,observations):
        #     print(line)

        viterbi_seq = []
        max_prob = 0.0
        best_st = None
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        viterbi_seq.append(best_st)
        previous = best_st

        for t in range(len(V) - 2, -1, -1):
            viterbi_seq.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        # print("%40s: %s" % ("Most likely sequence by Viterbi:", str(viterbi_seq)))
        return viterbi_seq

    def getParentWordProbability(self,data):
        grandParentWordTags = {}
       
        for each_sentence in data:
            for i in range(1,len(each_sentence[0])):
                if not each_sentence[0][i] in grandParentWordTags.keys():
                    grandParentWordTags[each_sentence[0][i]] = {}
                    for j in self.prob:
                        grandParentWordTags[each_sentence[0][i]][j] = 0
                grandParentWordTags[each_sentence[0][i]][each_sentence[1][i-1]] = grandParentWordTags[each_sentence[0][i]][each_sentence[1][i-1]] + 1
             
        return grandParentWordTags

    # GrandparentPOSTag probability
    def getGrandparentPOSProbability(self,data):
        grandTransition_p = {'adj':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'adv':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'adp':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},
        'conj':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'det':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'noun':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'num':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'pron':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'prt':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},
        'verb':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'x':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'x':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0},'.':{'adj':0, 'adv':0, 'adp':0, 'conj':0, 'det':0, 'noun':0, 'num':0, 'pron':0, 'prt':0, 'verb':0, 'x':0, '.':0}}
        for each_sentence in data:
            for i in range(len(each_sentence[0])-2):
                grandTransition_p[each_sentence[1][i]][each_sentence[1][i+2]] = grandTransition_p[each_sentence[1][i]][each_sentence[1][i+2]] + 1

        for i in grandTransition_p.keys():
            sum_key = sum(grandTransition_p[i].values())
            for j in grandTransition_p[i].keys():
                if sum_key!=0:
                    grandTransition_p[i][j] = grandTransition_p[i][j]/sum_key

        return grandTransition_p

    def gibbsSampling(self,no_samples,sentence):
        x = [[] for _ in range(no_samples)]
        x[0]= [random.choice(self.prob) for _ in range(len(sentence))]
        y = []
        pr = []
        max_pr = 0
        p=1
        max_p = 0
        max_p_seq = copy.deepcopy(x[0])
        
        # print('Emission pr - ',self.e_pr)
        # print('Transition pr - ',self.t_pr)
        # print('Grandparent POS - ',self.grandParentPOS)
        # print('Grandparent tag - ',self.grandParentWordTag)

        for i in range(1,no_samples):#number of samples
            x[i] = x[i-1]
            for j in range(len(sentence)):#number of words in the sentence
                # print(max_p_seq)
                x[i] = copy.deepcopy(max_p_seq)
                max_p = 0
                for p in self.prob:
                    x[i][j] = p
                    # print(x[i])
                    p = 1
                    
                    # print(p)
                    # print(sentence[j])
                    # print(self.corpus.keys())
                    if sentence[j] in self.corpus.keys():
                        # print('In parent if')
                        # print(self.e_pr.keys())
                        if j ==0:
                            if sentence[0] in self.corpus.keys():
                                p = p * self.e_pr[sentence[0]][x[i][0]] 
                        else:
                            # print('In else')
                            if sentence[0] in self.corpus.keys():
                                p = p * self.e_pr[sentence[0]][x[i][0]] 
                            for q in range(1,j+1):
                                if sentence[q] in self.e_pr.keys() and  sentence[q] in self.grandParentWordTag.keys():
                                    p = p * self.e_pr[sentence[q]][x[i][q]]*self.grandParentWordTag[sentence[q]][x[i][q-1]]
                            # print('1',p)
                            if j == 1:
                                p = p * self.t_pr[x[i][0]][x[i][1]]
                            else : 
                                p = p * self.t_pr[x[i][0]][x[i][1]]
                                for k in range(2,j+1):
                                    p = p * self.t_pr[x[i][k-1]][x[i][k]] * self.grandParentPOS[x[i][k-2]][x[i][k]]
                                if j == len(sentence):
                                    p = p * self.t_pr[x[i][j-1]][x[i][j]]
                            # print('2',p)
                    # print(p)
                    #Second word
        
        return max_p_seq

    def complex_mcmc(self, sentence):
        # MCMC inference code 
        s = self.gibbsSampling(2,sentence)
        
        return s
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

