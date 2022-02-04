import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import random
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function
    BayesNet = BayesianModel()
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("gauge","alarm")
    BayesNet.add_edge("faulty alarm","alarm")
    # raise NotImplementedError
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpd_temperature = TabularCPD("temperature",2,values=[[0.8],[0.2]])
    cpd_faultygauge = TabularCPD("faulty gauge",2,values=[[0.95,0.2],[0.05,0.8]],evidence=['temperature'], evidence_card=[2])
    cpd_faultyalarm = TabularCPD("faulty alarm",2,values=[[0.85],[0.15]])
    cpd_gague = TabularCPD("gauge",2,values=[[0.95,0.05,0.2,0.8],[0.05,0.95,0.8,0.2]],evidence=["faulty gauge","temperature"],evidence_card=[2,2])
    cpd_alarm = TabularCPD("alarm",2,values=[[0.9,0.1,0.55,0.45],[0.1,0.9,0.45,0.55]],evidence=["faulty alarm","gauge"],evidence_card=[2,2])
    bayes_net.add_cpds(cpd_temperature,cpd_faultygauge,cpd_faultyalarm,cpd_gague,cpd_alarm)

    #raise NotImplementedError
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    alarm_prob = marginal_prob['alarm'].values
    return alarm_prob[1]

#testing1.c
# BayesNet = make_power_plant_net()
# BayesNet = set_probability(BayesNet)
# print(get_alarm_prob(BayesNet))

def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    gauge_prob = marginal_prob['gauge'].values
    return gauge_prob[1]
# print(get_gauge_prob(BayesNet))


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'], evidence={'alarm': 1, 'faulty alarm': 0,'faulty gauge':0}, joint=False)
    temp_prob = conditional_prob['temperature'].values
    return temp_prob[1]
# print(get_temperature_prob(BayesNet))
#print(BayesNet.get_independencies())


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """

    # TODO: fill this out
    BayesNet = BayesianModel()

    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")


    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("A", "CvA")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("C", "BvC")

    cpd_A = TabularCPD("A", 4, [[0.15], [0.45], [0.30], [0.10]], state_names={'A':['0','1','2','3']})
    cpd_B = TabularCPD("B", 4, [[0.15], [0.45], [0.30], [0.10]], state_names={'B': ['0', '1', '2', '3']})
    cpd_C = TabularCPD("C", 4, [[0.15], [0.45], [0.30], [0.10]], state_names={'C': ['0', '1', '2', '3']})
    cpd_AvB = TabularCPD("AvB",3,[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],
                                  [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],
                                  [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],
                         evidence=['A','B'],evidence_card=[4,4])
    cpd_BvC = TabularCPD("BvC", 3,
                         [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
                          [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
                          [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]],
                         evidence=['B', 'C'], evidence_card=[4, 4])
    cpd_CvA = TabularCPD("CvA", 3,
                         [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9, 0.75, 0.6, 0.1],
                          [0.1, 0.6, 0.75, 0.9, 0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
                          [0.8, 0.2, 0.1, 0.05, 0.2, 0.8, 0.2, 0.1, 0.1, 0.2, 0.8, 0.2, 0.05, 0.1, 0.2, 0.8]],
                         evidence=['C', 'A'], evidence_card=[4, 4])

    #print(cpd_AvB)
    # # print(cpd_BvC)
    # # print(cpd_CvA)
    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    

    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'],
                                    evidence={'AvB': 0, 'CvA': 2}, joint=False)
    posterior = conditional_prob['BvC'].values
    #print(posterior)
    return posterior # list
# BayesNet = get_game_network()
# print(calculate_posterior(BayesNet))

def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    333
    """
    #Getting required tables
    A_cpd = bayes_net.get_cpds('A')
    team_table = A_cpd.values
    #print(team_table)
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    #print("match",match_table)
    if initial_state == [] or initial_state is None:
        A = random.randint(0,4,1)
        B = random.randint(0, 4, 1)
        C = random.randint(0, 4, 1)
        AvB = [0]
        BvC = random.randint(0,3,1)
        CvA = [2]
        sample = (A[0],B[0],C[0],AvB[0],BvC[0],CvA[0])
        #print(sample)
    else:
        sample = tuple(initial_state)

    #Gibbs Algorithm
    changer = random.choice([0,1,2,4])
    #print(changer)
    Pa = team_table[sample[0]]
    Pb = team_table[sample[1]]
    Pc = team_table[sample[2]]
    PAvB = match_table[sample[3]][sample[0]][sample[1]]
    PBvC = match_table[sample[4]][sample[1]][sample[2]]
    PCvA = match_table[sample[5]][sample[2]][sample[0]]

    if changer == 0:
        #case1
        Pa= team_table[0]
        Pb = team_table[sample[1]]
        Pc = team_table[sample[2]]
        PAvB = match_table[sample[3]][0][sample[1]]
        PCvA = match_table[sample[5]][sample[2]][0]
        sample_changer_0 = Pa * Pb * Pc * PAvB * PCvA
        #case2
        Pa = team_table[1]
        Pb = team_table[sample[1]]
        Pc = team_table[sample[2]]
        PAvB = match_table[sample[3]][1][sample[1]]
        PCvA = match_table[sample[5]][sample[2]][1]
        sample_changer_1 = Pa * Pb * Pc * PAvB * PCvA
        #case3
        Pa = team_table[2]
        Pb = team_table[sample[1]]
        Pc = team_table[sample[2]]
        PAvB = match_table[sample[3]][2][sample[1]]
        PCvA = match_table[sample[5]][sample[2]][2]
        sample_changer_2 = Pa * Pb * Pc * PAvB * PCvA
        #case4
        Pa = team_table[3]
        Pb = team_table[sample[1]]
        Pc = team_table[sample[2]]
        PAvB = match_table[sample[3]][3][sample[1]]
        PCvA = match_table[sample[5]][sample[2]][3]
        sample_changer_3 = Pa * Pb * Pc * PAvB * PCvA
        sum_sample_changer = sample_changer_0+sample_changer_1+sample_changer_2+sample_changer_3
        true_prob_0 = (sample_changer_0 / sum_sample_changer)
        true_prob_1 = (sample_changer_1 / sum_sample_changer)
        true_prob_2 = (sample_changer_2 / sum_sample_changer)
        true_prob_3 = (sample_changer_3 / sum_sample_changer)
        true_prob = [true_prob_0,true_prob_1,true_prob_2,true_prob_3]
        sample_changer = random.choice([0,1,2,3],p=true_prob)
        #updating the value
        sample_list = list(sample)
        sample_list[0] = sample_changer
        sample = tuple(sample_list)
        return sample
    if changer == 1:
        # case1
        Pb = team_table[0]
        Pa = team_table[sample[0]]
        Pc = team_table[sample[2]]
        PAvB = match_table[sample[3]][sample[0]][0]
        PBvC = match_table[sample[4]][0][sample[2]]
        sample_changer_0 = Pa * Pb * Pc * PBvC * PAvB
        # case2
        Pb = team_table[1]
        PAvB = match_table[sample[3]][sample[0]][1]
        PBvC = match_table[sample[4]][1][sample[2]]
        sample_changer_1 = Pa * Pb * Pc * PBvC * PAvB
        # case3
        Pb = team_table[2]
        PAvB = match_table[sample[3]][sample[0]][2]
        PBvC = match_table[sample[4]][2][sample[2]]
        sample_changer_2 = Pa * Pb * Pc * PBvC * PAvB
        # case4
        Pb = team_table[3]
        PAvB = match_table[sample[3]][sample[0]][3]
        PBvC = match_table[sample[4]][3][sample[2]]
        sample_changer_3 = Pa * Pb * Pc * PBvC * PAvB
        sum_sample_changer = sample_changer_0 + sample_changer_1 + sample_changer_2 + sample_changer_3
        true_prob_0 = (sample_changer_0 / sum_sample_changer)
        true_prob_1 = (sample_changer_1 / sum_sample_changer)
        true_prob_2 = (sample_changer_2 / sum_sample_changer)
        true_prob_3 = (sample_changer_3 / sum_sample_changer)
        true_prob = [true_prob_0, true_prob_1, true_prob_2, true_prob_3]

        sample_changer = random.choice([0, 1, 2, 3], p=true_prob)
        sample_list = list(sample)
        sample_list[1] = sample_changer
        sample = tuple(sample_list)
        return sample
    if changer == 2:
        # case1
        Pc = team_table[0]
        Pa = team_table[sample[0]]
        Pb = team_table[sample[1]]
        PBvC = match_table[sample[4]][sample[1]][0]
        PCvA = match_table[sample[5]][0][sample[0]]
        sample_changer_0 = Pa*Pb*Pc*PCvA*PBvC
        # case2
        Pc = team_table[1]
        PBvC = match_table[sample[4]][sample[1]][1]
        PCvA = match_table[sample[5]][1][sample[0]]
        sample_changer_1 =Pa*Pb*Pc*PCvA*PBvC
        # case3
        Pc = team_table[2]
        PBvC = match_table[sample[4]][sample[1]][2]
        PCvA = match_table[sample[5]][2][sample[0]]
        sample_changer_2 = Pa*Pb*Pc*PCvA*PBvC
        # case4
        Pc = team_table[3]
        PBvC = match_table[sample[4]][sample[1]][3]
        PCvA = match_table[sample[5]][3][sample[0]]
        sample_changer_3 = Pa*Pb*Pc*PCvA*PBvC
        sum_sample_changer = sample_changer_0 + sample_changer_1 + sample_changer_2 + sample_changer_3
        true_prob_0 = (sample_changer_0 / sum_sample_changer)
        true_prob_1 = (sample_changer_1 / sum_sample_changer)
        true_prob_2 = (sample_changer_2 / sum_sample_changer)
        true_prob_3 = (sample_changer_3 / sum_sample_changer)
        true_prob = [true_prob_0, true_prob_1, true_prob_2, true_prob_3]
        sample_changer = random.choice([0, 1, 2, 3], p=true_prob)
        sample_list = list(sample)
        sample_list[2] = sample_changer
        sample = tuple(sample_list)
        return sample
    if changer == 4:
        # case1
        PBvC = match_table[0][sample[1]][sample[2]]
        sample_changer_0 = Pb * Pc * PBvC
        # case2
        PBvC = match_table[1][sample[1]][sample[2]]
        sample_changer_1 = Pb * Pc * PBvC
        # case3
        PBvC = match_table[2][sample[1]][sample[2]]
        sample_changer_2 = Pb * Pc * PBvC

        sum_sample_changer = sample_changer_0 + sample_changer_1 + sample_changer_2
        true_prob_0 = (sample_changer_0 / sum_sample_changer)
        true_prob_1 = (sample_changer_1 / sum_sample_changer)
        true_prob_2 = (sample_changer_2 / sum_sample_changer)

        true_prob = [true_prob_0, true_prob_1, true_prob_2]
        sample_changer = random.choice([0,1,2],p=true_prob)
        sample_list = list(sample)
        sample_list[4] = sample_changer
        sample = tuple(sample_list)
        return sample




#print(Gibbs_sampler(BayesNet,[0, 1, 1, 0, 2, 2]))

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values

    # TODO: finish this function
    if initial_state == [] or initial_state is None:
        A = random.randint(0,4,1)
        B = random.randint(0, 4, 1)
        C = random.randint(0, 4, 1)
        AvB = [0]
        BvC = random.randint(0,3,1)
        CvA = [2]
        sample = (A[0],B[0],C[0],AvB[0],BvC[0],CvA[0])
    else:
        sample = tuple(initial_state)

    last_sample = sample
    A = random.randint(0, 4, 1)
    B = random.randint(0, 4, 1)
    C = random.randint(0, 4, 1)
    AvB = [0]
    BvC = random.randint(0, 3, 1)
    CvA = [2]
    new_sample = (A[0], B[0], C[0], AvB[0], BvC[0], CvA[0])

    #computing initial sample probability
    Pa = team_table[last_sample[0]]
    Pb = team_table[last_sample[1]]
    Pc = team_table[last_sample[2]]
    PAvB = match_table[last_sample[3]][last_sample[0]][last_sample[1]]
    PBvC = match_table[last_sample[4]][last_sample[1]][last_sample[2]]
    PCvA = match_table[last_sample[5]][last_sample[2]][last_sample[0]]
    P_last_joint = Pa*Pb*Pc*PAvB*PBvC*PCvA

    #computing new sample joint probability

    Pa = team_table[new_sample[0]]
    Pb = team_table[new_sample[1]]
    Pc = team_table[new_sample[2]]
    PAvB = match_table[new_sample[3]][new_sample[0]][new_sample[1]]
    PBvC = match_table[new_sample[4]][new_sample[1]][new_sample[2]]
    PCvA = match_table[new_sample[5]][new_sample[2]][new_sample[0]]
    P_new_joint = Pa * Pb * Pc * PAvB * PBvC * PCvA

    r = P_new_joint/P_last_joint
    u = random.uniform(0,1)

    if u<=r:
        sample = new_sample
    else:
        sample = last_sample


    return sample

#print(MH_sampler(BayesNet,[0, 2, 0, 0, 2, 2]))
def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    T=10000000
    G_BvC = [0,0,0]
    old_Gibbs = [0,0,0]
    #Gibbs Convergence
    N = 200
    delta = 0.000001
    burn_in =1000
    for i in range(burn_in):
        next_state = Gibbs_sampler(bayes_net, initial_state)
        #G_BvC[next_state[4]] += 1
        Gibbs_count +=1
        initial_state = next_state
    for i in range(T):
        next_state = Gibbs_sampler(bayes_net,initial_state)
        Gibbs_count += 1
        old_sum_BvC = sum(G_BvC)
        for i in range(len(G_BvC)):
            if old_sum_BvC != 0 :
                old_Gibbs[i] = (G_BvC[i]/old_sum_BvC)
        G_BvC[next_state[4]] += 1
        sum_BvC = sum(G_BvC)
        for i in range(len(G_BvC)):
            Gibbs_convergence[i] = (G_BvC[i]/sum_BvC)
        if abs(old_Gibbs[0] - Gibbs_convergence[0]) <= delta and abs(old_Gibbs[1] - Gibbs_convergence[1]) <= delta and abs(old_Gibbs[2] - Gibbs_convergence[2]) <= delta:
            N-=1
        else:
            N = 200
        if N== 0:
            break
        else:
            initial_state = next_state

    # MH Convergence
    T = 10000000
    M_BvC = [0, 0, 0]
    old_MH = [0, 0, 0]
    N = 200
    delta = 0.000001
    burn_in = 1000

    for i in range(burn_in):
        next_state = MH_sampler(bayes_net, initial_state)
        #M_BvC[next_state[4]] += 1
        if next_state == initial_state:
            MH_rejection_count += 1
        MH_count += 1
        initial_state = next_state
    #print(M_BvC)
    for i in range(T):
        next_state = MH_sampler(bayes_net,initial_state)
        if next_state == initial_state:
            MH_rejection_count += 1
        MH_count += 1
        old_Hsum_BvC = sum(M_BvC)
        for i in range(len(M_BvC)):
            if old_Hsum_BvC != 0 :
                old_MH[i] = (M_BvC[i]/old_Hsum_BvC)
        M_BvC[next_state[4]] += 1
        sum_MBvC = sum(M_BvC)
        for i in range(len(M_BvC)):
            MH_convergence[i] = (M_BvC[i]/sum_MBvC)
        if abs(old_MH[0] - MH_convergence[0]) <= delta and abs(old_MH[1] - MH_convergence[1]) <= delta and abs(old_MH[2] - MH_convergence[2]) <= delta:
            N-=1
        else:
            N = 200
        if N == 0:
            break
        else:
            initial_state = next_state
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

# print(compare_sampling(BayesNet,[]))
def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor

    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.11
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Surya Teja Adluri"
