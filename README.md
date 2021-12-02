# lghanta-pursurve-shrgutta-a3
a3 created for lghanta-pursurve-shrgutta

                                                                Part 2-Ice tracking
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

