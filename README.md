# Assignment 3 
a3 created for lghanta-pursurve-shrgutta

## Part 1 - Part-of-speech(POS) tagging
Parts of Speech tagging can be done in many different ways out of which the 3 methods implemented are -
1. Simple version - 

In simple version, just the emission probabilities(P(W|S)) are considered. These emission probabilities are calculated from the train data. Each word has emission probabilities w.r.to all the possible 12 Parts of speech. Based on these probabilities, each word in the sentence is assigned a Parts of speech that has the maximum emission probability. A sequence of POS is returned for the given test sentence. 

- Percent of words correctly detected in test data is :  93.95%
- Percent of sentences correctly detected in test data is : 47.5%

2. HMM version using Viterbi - 

In this version, transition probabilities (P(S[i]|S[i+1])), emission probabilities (P(W|S)), initial probabilities (P0) are considered. All the required probabilities are calculated from the train data. A Viterbi chain is formed based on all the possible POS tagging for each word using the concept of Dynamic programming. At every step ie predicting the POS for each word, we consider the maximum probabilty value of considering that route - prior probability * emission probability * tranisition probability. Maximising the probabilty at each and every step will result in a sequence of POS for given sentence. This is different from simple version in terms of - transiion probability ie there's some significance for the sequence of occurance of POS as learnt from the train data. Hence, transition probabiity also influences the final POS tagged to a word in a sequence depending on its parent word tag. The Bayes net used to solve this is the one given in question.

3. Complex Bayes Net (MCMC) using Gibbs Sampling -

In this version, there's a complex Bayes net to honor all the dependencies which include - prior/initial probabilities, emission probabilities, transition probabilities, grandChild transition probabilityies, grandParent emission probabilities.




## Part 2 - Ice Tracking
1)In the first subpart, we plotted the air ice and the ice rock boundary using Bayes net algorithm. Since if the edge strength is higher,the probability of the pixel being a point 
on the boundary is higher, we thus found 2 pixels with the maximum edge strength for each column which were minimum 10 pixels aparts. I have the used the pixels with the largest 
edge strengths in each column to plot the air-ice boundary and the pixels with the second largest edge strengths in each column to plot the ice-rock boundary.

2)The first approach does not give accurate boundary lines for few images. We can get more accurate boundary lines using the Viterbi algorithm. For the viterbi algorithm we are 
considering:

-Initial probability: We have calcuted the initial probabilities using the bayes net algorithm from subpart 1.

-Transition probability: For finding the transition probability, I have taken the taken the previous row and the current column in consideration. Since we want a smooth boundary, 
the probability of the pixels nearer to the pixel found to have higher probability would have a higher transition probability. 
Previous approach:-The transition probability is inversely proportional to
the distance between those two pixels. Thus, we calculate the reciprocal of the distances between the pixel with high probability and its neighbouring pixel within the distance of 
10 pixels above and below. The pixels which are farer than 10 pixels are assigned near to 0 probability.
Updated approach:-Here I have adjusted the transition probabilities using the above principle i.e P(si+1|si) is high if si+1 = si and is low if they are very different. However, 
when the the probabilities of the pixels closer were taken near to 1 and and the pixels that were very far were almost considered as 0, we got a more accurate ice-rock boundary

-Emission probability: We calculate the emission probability by calculating the ratio of edge strength of a particular pixel to the sum of the edge strengths of all the pixels
in that column.
We then use the above probabilities in the Viterbi algorithm for plotting the air-ice and the ice-rock boundary.


## PART 3 : Reading text

Given: 
1. An image which contains all the possible characters in the test image (train image)
2. Test image, from which we have to recognize the character and return
3. A training text file, used to train the data or generate transition probabilities for Viterbi procedure

To Do:
Recognize characters from the test image given using:
1. Simple Bayes Net
2. Viterbi (HMM)

Simple Bayes Net:
This is calculated by calculating emission probabilities by comparing each and every character of each character and taking three different counts. And have set different probability for matched, mismatched and empty and created a value by summing on these three probabilities.

Viterbi :
Start state: These are the possible 73 states given in the training image, which include alphabets capital and small, number and few special characters.
Initial Probabilities: These are the initial probabilities where we calculate the probability of each character the training test file (total number of occurrences of that character/ total number of characters in the text file)
Emission Probabilities: These are the probabilities 
Transition Probabilities: These are the probability of next possible occurrence of that character after the previous character in the test image.
Observations: I have considered the position number of each character in the test file.

Firstly, I have created viterbi tables, left to right for transition and emission probabilities and then backtracked to find the most likely state sequence or maximum probability state in a backward direction.

In this case, I have minimized Viterbi because the training data and test images are not similar. I thought of creating my own training data using the sample test images given, but I thought that would not be the appropriate solution.
