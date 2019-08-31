# region Libraries
'''
Librarys Used.
'''
import numpy as np
import pandas as pd
import scipy.optimize as sp
from scipy.optimize import least_squares
from scipy.optimize import leastsq
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

def sum_of_squared_errors_loss_function(parameters,innings_number,runs_scored,remaining_overs,wickets_in_hand):
    '''
    This procedure defines the objective function which I have passed in scipy.optimize.minimize() function.
    It calculated all total squared error loss for all the data points for innings 1.
    :param parameters: List contains 11 parameters
    :param args: List contains innings_number,runs_scored,remaining_overs,wickets_in_hand
    :return:
    '''
    total_squared_error=0
    l_param=parameters[10]
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
    #x0 = np.array([10, 30, 40, 60, 90, 125, 150, 170, 190, 200,3], dtype=float)
    x0 = np.array([10, 30, 40, 60, 90, 125, 150, 170, 190, 200, 10], dtype=float)
    optimised_res = least_squares(sum_of_squared_errors_loss_function,x0,args=(innings_number,runs_scored,remaining_overs,wickets_in_hand))
    return optimised_res.cost,optimised_res.x

def plotparam_expectedrunvsoverremains(optparameters):
    '''
    This Procedure will plot the graph of ExpectedRun vs OverRemaining for all parameters.
    :param optparameters:
    '''
    plt.figure(1)
    plt.title("Expected Runs vs Overs Remaininng")
    plt.xlim((0, 50))
    plt.ylim((0, 250))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 50, 100, 150, 200, 250])
    plt.xlabel('Overs remaining')
    plt.ylabel('Expected Runs')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#555b65', '#999e45', '#222a55']
    x=np.zeros((51))
    for i in range(51):
        x[i]=i
    for i in range(len(optparameters)-1):
        y_run=optparameters[i] * (1 - np.exp(-optparameters[10] * x /optparameters[i]))
        plt.plot(x, y_run, c=colors[i], label='Z[' + str(i + 1) + ']')
        plt.legend()
    plt.savefig('parameterplot_expectedrun_vs_overremain_leastsquare.png')
    plt.close()


def plotparam_resourceremainvsoverremains(optparameters):
    '''
        This Procedure will plot the graph of ResourceRemainings vs OverRemaining for all parameters.
        :param optparameters:
        '''
    plt.figure(1)
    plt.title("Resource Remaining vs Overs Remaininng")
    plt.xlim((0, 50))
    plt.ylim((0, 100))
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Overs remaining')
    plt.ylabel('percentage Of Resource Remaining')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#555b65', '#999e45', '#222a55']
    x = np.zeros((51))
    for i in range(51):
        x[i] = i
    Z5010=optparameters[9] * (1 - np.exp(-optparameters[10] * 50 /optparameters[9]))
    for i in range(len(optparameters)-1):
        y_run=optparameters[i] * (1 - np.exp(-optparameters[10] * x /optparameters[i]))
        plt.plot(x, (y_run/Z5010)*100, c=colors[i], label='Z[' + str(i + 1) + ']')
        plt.legend()
    plt.savefig('parameterplot_resourceremain_vs_overremain_leastsquare.png')
    plt.close()


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
    plotparam_expectedrunvsoverremains(optparameters)
    plotparam_resourceremainvsoverremains(optparameters)
    print("Plots are generated.Check source directory for 'parameterplot_resourceremain_vs_overremain_L-BFGS-B.png' and 'parameterplot_expectedrun_vs_overremain_L-BFGS-B.png' ")
