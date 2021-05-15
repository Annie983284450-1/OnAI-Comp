'''
implement UCB1 policy
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
        #[0]*XX: It 'multiplies' the list elements

        self.UCB = [0]*self.k # maintain a list of UCB values of all arms at the time



        '''
        ### The following code is used in the simulated naive UCB model
            useless herein, commented
        self.var_list = var_list  #  a list of variances from beta, serve as subsitute for UCB upper bound

        '''



    def __str__(self):
        return 'UCB policy, alpha = {}'.format(self.alpha)

    '''
    input results from Thompson sampling, namely, estimated_mu for
    each arm, time t, N_i(t),
    !!!! pull arm with largest UCB,
    output: update hat_mu,  N_i(t), and ConfBound
    ouput : index of the arm pulled
    '''



    '''
    get the reward, let the reward = estimated mean + uncertainty
    '''
    def get_UCB(self):
        # max_index = 0
        # suppose that we have K arms herein
        # each loop, we deal with one arm (i.e., clinical expert)]


        for i in np.arange(self.k):
        # if this arm has been selected before
            if self.N[i]!=0:
                # self.UCB[i]= self.hat_mu_list[i]+self.upperBound(i)
                self.UCB[i]= self.hat_mu_list[i]+self.upperBound2(self.N[i])



            # Edge case:  if this arm has not been selected before, the UCB will go to infinity
            # sqrt( [ alpha * log (t) ] /  N_i(t)  )-> infinity

            else:

                self.UCB[i] = sys.float_info.max
        return self.UCB

    def pull_max_arm(self):
        # max_index = 0
        # suppose that we have K arms herein
        # each loop, we deal with one arm (i.e., clinical expert)]





        for i in np.arange(self.k):
        # if this arm has been selected before
            if self.N[i]!=0:
                # self.UCB[i]= self.hat_mu_list[i]+self.upperBound(i)
                self.UCB[i]= self.hat_mu_list[i]+self.upperBound2(self.N[i])
                #print(i,"-th", "upperbound value:",self.upperBound2(self.N[i]))


            # Edge case:  if this arm has not been selected before, the UCB will go to infinity
            # sqrt( [ alpha * log (t) ] /  N_i(t)  )-> infinity

            else:

                self.UCB[i] = sys.float_info.max
                print(i,"-th", "upperbound value:",self.UCB[i]-self.hat_mu_list[i])
            #print(i,"-th","UCB value: ", self.UCB[i])

            #print(i,"-th", "hat_mu:",self.hat_mu_list[i])
        pulled_arm_index = np.argmax(self.UCB)
        #reward = self.hat_mu_list[pulled_arm_index]
        return pulled_arm_index#, reward
        '''
        # to deal with the cold start problem, let us do this naively first ...
        if self.t <=10:
            arm_index = 1

            return arm_index, self.UCB[arm_index]
        else:
            return np.argmax(self.UCB), np.amax(self.UCB)
        '''

    '''
    sqrt( [ alpha * log (t) ] /  N_i(t)  )
    '''
    def upperBound2(self,N_it):
        #print("log(t+1)=", math.log(self.t+1) )
        return  math.sqrt(self.alpha * math.log(self.t+1) / N_it)





'''
    ### The following code is used in the simulated naive UCB model
    ### useless herein, commented
    ## use variance of beta as confidence interval

    def upperBound(self, arm):
        return self.var_list[arm]
'''
