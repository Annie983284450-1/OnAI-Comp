'''
ref: https://github.com/bgalbraith/bandits/blob/master/bandits/bandit.py
ref: https://github.com/j2kun/ucb1/blob/master/ucb1.py
'''

import math
import random
import numpy as np
import arm
import sys



'''
for arm i at time/round t,
UCB_i^(t) = hat_mu_i^(t) +  sqrt( [ alpha * log (t) ] /  N_i(t)  )
where hat_mu_i^(t) is the estimated value of arm i at time t
N_i(t) : up to time t, the # of times arm i was picked
select argmax (UBC_i(t)) over K arms, observe reward x_i(t)
'''
class UCB(object):
    def __init__(self,t, hat_mu_list, N, var_list,alpha=2):
        # constant term, default = 2
        self.alpha = alpha
        self.t=t
        self.hat_mu_list=hat_mu_list
        self.N = N
        self.k = len(self.hat_mu_list)
        self.UCB = [0]*self.k # maintain a list of UCB values of all arms at the time




    def __str__(self):
        return 'UCB policy, alpha = {}'.format(self.alpha)

    '''
    input results from Thompson sampling, namely, estimated_mu for
    each arm, time t, N_i(t),
    pull arm with largest UCB,
    output: update hat_mu,  N_i(t), and ConfBound
    ouput : index of the arm pulled

    get the reward, let the reward = estimated mean + uncertainty
    '''
    def get_UCB(self):
        # suppose that we have K arms herein
        # each loop, we deal with one arm (i.e., clinical expert)]
        for i in np.arange(self.k):
        # if this arm has been selected before
            if self.N[i]!=0:
                 
                self.UCB[i]= self.hat_mu_list[i]+self.upperBound2(self.N[i])

            else:

                self.UCB[i] = sys.float_info.max
        return self.UCB

    def pull_max_arm(self):
        # suppose that we have K arms herein
        # each loop, we deal with one arm (i.e., clinical expert)]
        for i in np.arange(self.k):
        # if this arm has been selected before
            if self.N[i]!=0:
                self.UCB[i]= self.hat_mu_list[i]+self.upperBound2(self.N[i])
            else:

                self.UCB[i] = sys.float_info.max
                print(i,"-th", "upperbound value:",self.UCB[i]-self.hat_mu_list[i])

        pulled_arm_index = np.argmax(self.UCB)
        return pulled_arm_index 

    def upperBound2(self,N_it):
        return  math.sqrt(self.alpha * math.log(self.t+1) / N_it)


