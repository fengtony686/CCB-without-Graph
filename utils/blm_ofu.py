import numpy as np
from utils.blm_lr import find_best_intervention


def run_blm_ofu_unknown(graph):
    payoff_list = [0]
    total_expected_payoff = 0

    # Graph Identification Process
    graphIdNum = int(np.power(graph.T, 1 / 2) * 0.1)
    correctList = [True for _ in range(graph.num_parents_y)]
    for i in range(graph.num_parents_y):
        sumvy1, sumy1 = np.zeros(graph.num_parents_y), 0
        sumvy0, sumy0 = np.zeros(graph.num_parents_y), 0
        for j in range(graphIdNum):
            vy, y = graph.simulate([i], True)
            sumvy1 += vy.T[0]
            sumy1 += y
            total_expected_payoff += graph.expect_y([i])
            if (i * graphIdNum * 2 + 2 * j) % 100 == 99:
                payoff_list.append(total_expected_payoff)
            vy, y = graph.simulate([i], False)
            sumvy0 += vy.T[0]
            sumy0 += y
            total_expected_payoff += graph.expect_y([i]) - graph.theta_y[i]
            if (i * graphIdNum * 2 + 2 * j + 1) % 100 == 99:
                payoff_list.append(total_expected_payoff)
        if sumy1 - sumy0 < 0.01 * np.power(graph.T, 3 / 10):
            correctList[i] = False

    # Original BLM-OFU algorithm
    matrix_m = np.zeros((graph.num_parents_y, graph.num_parents_y))
    for i in range(graph.num_parents_y):
        matrix_m[i][i] = 1
    matrix_mx = np.zeros(graph.num_parents_y)
    intervened_times = np.zeros(graph.num_parents_y)
    by = np.array([np.zeros(graph.num_parents_y)]).T
    for i in range(graphIdNum*graph.num_parents_y*2, graph.T):
        if i >= 0 and correctList.count(True) >= graph.k:
            inverse_m = np.linalg.inv(matrix_m)
            hat_theta_x = np.zeros(graph.num_parents_y)
            for j in range(graph.num_parents_y):
                hat_theta_x[j] = matrix_mx[j] / (i - intervened_times[j]) if intervened_times[j] < i else 0
            hat_theta_x = np.array([hat_theta_x]).T
            norm_matrix_mx = np.array([np.sqrt(i - intervened_times[j]) for j in range(graph.num_parents_y)])
            hat_theta_y = np.matmul(inverse_m, np.array([by]).T)
            rho = graph.compute_rho_ofu()
            best_intervention = find_best_intervention(graph, inverse_m, hat_theta_x, hat_theta_y, norm_matrix_mx, rho, correctList)
        else:
            best_intervention = [i if correctList[i] else -1 for i in range(graph.num_parents_y)]
            if -1 in best_intervention:
                best_intervention.remove(-1)
        vy, y = graph.simulate(best_intervention)
        total_expected_payoff += graph.expect_y(best_intervention)
        if i % 100 == 99:
            payoff_list.append(total_expected_payoff)
        matrix_m += np.matmul(vy, vy.T)
        by += y * vy
        for j in best_intervention:
            intervened_times[int(j)] += 1
        for j in range(graph.num_parents_y):
            if j not in best_intervention:
                matrix_mx[j] += vy[j]
    return payoff_list, total_expected_payoff
