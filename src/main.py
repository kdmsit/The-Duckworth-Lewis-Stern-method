# region Libraries
'''
Librarys Used.
'''
import numpy as np
import pandas as pd
import scipy.optimize as sp
import math
import matplotlib.pyplot as plt
# endregion

def data_preprocessing(match_datas):
    '''
    This Procedure fetches remaining_runs,remaining_overs,wickets_in_hand,innings_number from the data matrix "match_datas"
    :param match_datas:Data Matrix
    :return: remaining_runs,remaining_overs,wickets_in_hand,innings_number
    '''
    innings_number     = match_datas['Innings'].values
    remaining_runs     = match_datas['Innings.Total.Runs'].values - match_datas['Total.Runs'].values
    remaining_overs    = match_datas['Total.Overs'].values-match_datas['Over'].values
    wickets_in_hand    = match_datas['Wickets.in.Hand'].values
    return remaining_runs,remaining_overs,wickets_in_hand,innings_number

def sum_of_squared_errors_loss_function(parameters,args):
    '''
    This procedure defines the objective function which I have passed in scipy.optimize.minimize() function.
    It calculated all total squared error loss for all the data points for innings 1.
    :param parameters: List contains 11 parameters
    :param args: List contains innings_number,runs_scored,remaining_overs,wickets_in_hand
    :return:
    '''
    total_squared_error=0
    l_param=parameters[10]
    innings_number = args[0]
    runs_scored=args[1]
    remaining_overs=args[2]
    wickets_in_hand=args[3]
    for i in range(len(wickets_in_hand)):
        if innings_number[i] == 1:
            runscored = runs_scored[i]
            overremain = remaining_overs[i]
            wicketinhand = wickets_in_hand[i]
            Z0=parameters[wicketinhand - 1]
            if runscored > 0:
                predicted_run =  Z0 * (1 - np.exp(-1*l_param * overremain / Z0))
                total_squared_error=total_squared_error + (math.pow(predicted_run - runscored, 2))
    return total_squared_error

def fit_parameters(innings_number,runs_scored,remaining_overs,wickets_in_hand):
    '''
    This procedure will fit the curve to optimise the overall loss function against 11 parameters
    :param innings_number:
    :param runs_scored:
    :param remaining_overs:
    :param wickets_in_hand:
    :return:
    '''
    parameters = [10, 30, 40, 60, 90, 125, 150, 170, 190, 200,3]
    optimised_res = sp.minimize(sum_of_squared_errors_loss_function,parameters,
                      args=[innings_number,runs_scored,remaining_overs,wickets_in_hand],
                      method='CG')
    return optimised_res['fun'],optimised_res['x']

def plot(optparameters):
    '''
    This Procedure will plot the graphs for all parameters.
    :param optparameters:
    :return:
    '''
    plt.figure(1)
    plt.xlim((0, 50))
    plt.ylim((0, 250))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 50, 100, 150, 200, 250])
    plt.xlabel('Overs remaining')
    plt.ylabel('Expected Runs')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#555b65', '#999e45', '#777a55']
    x=np.zeros((51))
    for i in range(51):
        x[i]=i
    for i in range(len(optparameters)-1):
        y_run=optparameters[i] * (1 - np.exp(-optparameters[10] * x /optparameters[i]))
        plt.plot(x, y_run, c=colors[i], label='Z[' + str(i + 1) + ']')
        plt.legend()
    plt.savefig('parameterplot_CG.png')



if __name__ == "__main__":
    match_datas = pd.read_csv('../data/04_cricket_1999to2011.csv')
    runs_to_be_scored,remaining_overs,wickets_in_hand,innings_number=data_preprocessing(match_datas)
    totalloss,optparameters=fit_parameters(innings_number,runs_to_be_scored, remaining_overs, wickets_in_hand)
    print("TOTAL LOSS ",totalloss)
    for i in range(len(optparameters)):
        if(i == 10):
            print("L :"+str(optparameters[i]))
        else:
            print("Z["+str(i+1)+"] :"+str(optparameters[i]))
    plot(optparameters)