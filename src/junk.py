def read_features(match_datas):
    match_number                    = match_datas['Match'].values
    innings_number                  = match_datas['Innings'].values
    runs_cumulative_upto_overs      = match_datas['Total.Runs'].values
    total_runs                      = match_datas['Innings.Total.Runs'].values
    remaining_runs                  = total_runs - runs_cumulative_upto_overs
    current_over                    = match_datas['Over'].values
    total_overs                     = match_datas['Total.Overs'].values
    remaining_overs                 = total_overs-current_over
    wickets_in_hand                 = match_datas['Wickets.in.Hand'].values
    totalrun=0
    totalfirstinningsmatches=0
    run2d_matrix_peroverwck = np.zeros((50, 10), dtype=float)
    count_matrix=np.zeros((50, 10), dtype=int)
    for i in range(len(match_number)):
        if innings_number[i] == 1:
            run2d_matrix_peroverwck[remaining_overs[i] - 1][wickets_in_hand[i] - 1] += remaining_runs[i]
            count_matrix[remaining_overs[i] - 1][wickets_in_hand[i] - 1] += 1
            totalrun += total_runs[i]
            totalfirstinningsmatches += 1
    avgrun=totalrun/totalfirstinningsmatches
    for i in range(50):
        for j in range(10):
            if(count_matrix[i][j] == 0):
                count_matrix[i][j] = 1
    avgrun2d_matrix_peroverwck=np.divide(run2d_matrix_peroverwck,count_matrix)
    avgrun2d_matrix_peroverwck[49][9]=avgrun
    return avgrun2d_matrix_peroverwck,remaining_runs,remaining_overs

parameters=[10.0, 30.0, 40.0, 60.0, 90.0, 125.0, 150.0, 170.0, 190.0, 220.0, 3]


def plot():
    plt.figure(1)
    plt.xlim((0, 50))
    plt.ylim((0, 1))
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Overs remaining')
    plt.ylabel('Percentage of resource remaining')
    max_resource_list = evaluate_run_prediction_func(optparameters[9], optparameters[10], 50)
    x = np.arange(0, 51, 1)
    modified_x = np.array([50.0 - i for i in x])
    color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#234a21', '#876e34', '#a21342']
    for i in range(10):
        y = 100 * evaluate_run_prediction_func(optparameters[i], optparameters[10],
                                               modified_x) / max_resource_list
        plt.plot(x, y, c=color_list[i], label='Z[' + str(i + 1) + ']')
        plt.legend()

    y_linear = [-2 * i + 100 for i in x]
    plt.plot(x, y_linear, '#5631a2')
    plt.savefig('run_prediction_functions.pdf')


def evaluate_run_prediction_func(z, l, u):
    return z * (1 - np.exp(-l*u/z))