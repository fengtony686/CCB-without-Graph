import numpy as np


def find_best_intervention(graph, inverse_m, hat_theta_x, hat_theta_y, norm_matrix_mx, rho, correctList):
    legalList = [i if correctList[i] else -1 for i in range(graph.num_parents_y)]
    if -1 in legalList:
        legalList.remove(-1)
    max_y = - 9999
    for i in range(graph.num_parents_y):
        hat_theta_x[i] = hat_theta_x[i] + rho / norm_matrix_mx[i] if norm_matrix_mx[i] > 0 else 1
    best_intervention = []
    for j in range(np.power(graph.num_parents_y, graph.k)):
        intervened_indexes = []
        for k in range(graph.k):
            index = int(j / np.power(graph.num_parents_y, k)) % graph.num_parents_y
            if len(intervened_indexes) == 0 or index > intervened_indexes[-1]:
                intervened_indexes.append(index)
        if len(intervened_indexes) < graph.k:
            continue
        tilde_theta_x = hat_theta_x.copy()
        for i in intervened_indexes:
            tilde_theta_x[int(i), 0] = 1
        expected_y = rho * np.sqrt(np.matmul(np.matmul(tilde_theta_x.T, inverse_m), tilde_theta_x)) + np.matmul(
            tilde_theta_x.T, hat_theta_y)[0, 0, 0]
        if expected_y > max_y and len(list(set(legalList) & set(intervened_indexes))) == graph.k:
            max_y = expected_y
            best_intervention = intervened_indexes
    return best_intervention


def pair_oracle_3(hat_theta, inverse_m, rho, parents):
    return (rho * np.sqrt(np.matmul(np.matmul(parents.T, inverse_m), parents)[0, 0]) + np.matmul(
        parents.T, hat_theta))[0, 0, 0]


def run_blm_lr_unknown(graph):
    payoff_list = [0]
    total_expected_payoff = 0

    # Graph Identification Process
    graphIdNum = int(np.power(graph.T, 2/3) * 0.1)
    correctList = [True for _ in range(graph.num_parents_y)]
    for i in range(graph.num_parents_y):
        sumvy1, sumy1 = np.zeros(graph.num_parents_y), 0
        sumvy0, sumy0 = np.zeros(graph.num_parents_y), 0
        for j in range(graphIdNum):
            vy, y = graph.simulate([i], True)
            sumvy1 += vy.T[0]
            sumy1 += y
            total_expected_payoff += graph.expect_y([i])
            if int(i*graphIdNum*2+2*j) % 100 == 99:
                payoff_list.append(total_expected_payoff)
            vy, y = graph.simulate([i], False)
            sumvy0 += vy.T[0]
            sumy0 += y
            total_expected_payoff += graph.expect_y([i]) - graph.theta_y[i]
            if int(i*graphIdNum*2+2*j+1) % 100 == 99:
                payoff_list.append(total_expected_payoff)
        if sumy1 - sumy0 < 0.01 * np.power(graph.T, 1/3) * np.log(np.square(graph.T)):
            correctList[i] = False

    # Original BLM-LR algorithm
    matrix_m = np.zeros((graph.num_parents_y, graph.num_parents_y))
    for i in range(graph.num_parents_y):
        matrix_m[i][i] = 1
    matrix_mx = np.zeros(graph.num_parents_y)
    intervened_times = np.zeros(graph.num_parents_y)
    by = np.array([np.zeros(graph.num_parents_y)]).T
    for i in range(graphIdNum * graph.num_parents_y * 2, graph.T):
        inverse_m = np.linalg.inv(matrix_m)
        hat_theta_x = np.zeros(graph.num_parents_y)
        for j in range(graph.num_parents_y):
            hat_theta_x[j] = matrix_mx[j] / (i - intervened_times[j]) if intervened_times[j] < i else 0
        hat_theta_x = np.array([hat_theta_x]).T
        norm_matrix_mx = np.array([np.sqrt(i - intervened_times[j]) for j in range(graph.num_parents_y)])
        hat_theta_y = np.matmul(inverse_m, np.array([by]).T)
        rho = graph.compute_rho_lr(i + 1)
        if correctList.count(True) >= graph.k:
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
