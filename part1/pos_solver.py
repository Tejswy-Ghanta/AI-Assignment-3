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
        self.e_pr = {}
        self.t_pr  ={}

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

        self.e_pr = self.getEmissionProbabilities() 
        self.t_pr =self.getTransitionProbabilities(data)
        print('e_pr - ',self.e_pr)  
        print('t_pr - ',self.t_pr)
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

    def dptable(self,V,observed):

        for state in V[0]:
            yield"%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)
        print("%40s: %s " % ("Observed sequence", str(observed)))
        N=len(observed)

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

        for line in self.dptable(V,observations):
            print(line)

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

        print("%40s: %s" % ("Most likely sequence by Viterbi:", str(viterbi_seq)))

        
                # final_POS = []
                
                # non_zero_paths = []
                # costs = {}
                # #viterbi initialization
                # key_for_tp = 's'
                # for j in self.prob:
                #     if j not in costs.keys() and self.t_pr[key_for_tp][j]!=0 and self.e_pr[sentence[0]][j]!=0:
                #     # emission probability * transition probability
                #         costs[j] =self.e_pr[sentence[0]][j]  * self.t_pr[key_for_tp][j] 
        #         non_zero_paths.append(j)

        # print('non_zero_paths - ',non_zero_paths)
        # print('cost - ',costs)
        # #viterbi chain construction
        # i= 1 
        
        # temp_c = {}

        # for i in range(1,len(sentence)):
        #     if sentence[i] not in self.e_pr.keys():
        #         #words not in corpus? 
        #         final_POS.append('noun')
        #         self.e_pr[sentence[i]]={'adj':1/12, 'adv':1/12, 'adp':1/12, 'conj':1/12, 'det':1/12, 'noun':1/12, 'num':1/12, 'pron':1/12, 'prt':1/12, 'verb':1/12, 'x':1/12, '.':1/12}
        #     for nz in non_zero_paths:
        #         for e in self.e_pr[sentence[i]].keys():
        #             if self.e_pr[sentence[i]][e]!=0:
        #                 for t in self.t_pr[e].keys():
        #                     if self.t_pr[e][t] !=0:
        #                         #don't modify costs here
        #                         #use a temp dict, to get keys and cost values, then find max and append to costs
        #                         temp_c[t] = self.e_pr[sentence[i]][e]*self.t_pr[e][t]*costs[nz]
                                
        #         print('temp_c - ',temp_c) 
        #         #find max of nz-e
                           
        return viterbi_seq

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

