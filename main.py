import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from utils.parallel_graph import ParallelGraph
from utils.ucb import run_ucb
from utils.epsilon_greedy import run_eps_greedy
from utils.blm_lr import run_blm_lr_unknown
from utils.blm_ofu import run_blm_ofu_unknown
import seaborn as sns
from multiprocessing import Process, Queue

T = 10000  # round number
repeated_num = 50  # repeated times


def compute_regret_list(repeat_num, best_y, graph, algorithm, q, coef=1.):
    regret_list = np.empty((repeat_num, int(graph.T / 100 + 1)))
    for i in trange(repeat_num):
        if algorithm == 'bglm-ofu-unknown':
            payoff_list, _ = run_blm_ofu_unknown(graph)
        elif algorithm == 'blm-lr-unknown':
            payoff_list, _ = run_blm_lr_unknown(graph)
        elif algorithm == 'ucb':
            payoff_list, _ = run_ucb(graph, coef)
        else:
            payoff_list, _ = run_eps_greedy(graph, coef)
        for index, k in enumerate(payoff_list):
            payoff_list[index] = best_y * 100 * index - payoff_list[index]
        regret_list[i, ::] = payoff_list
    q.put(np.mean(regret_list, axis=0))


newGraph = ParallelGraph([.3, .3, .2, .2, .2], [.3, .3, .13, .13, .13], T, 2)  # graph parameters

lr_queue = Queue()
ofu_queue = Queue()
ucb_queue = Queue()
eps_greedy_queue = Queue()

best_expect_y = newGraph.best_expect_y()

thread1 = Process(target=compute_regret_list,
                  args=(repeated_num, best_expect_y, newGraph, "blm-lr-unknown", lr_queue))
thread2 = Process(target=compute_regret_list,
                  args=(repeated_num, best_expect_y, newGraph, "bglm-ofu-unknown", ofu_queue))
thread3 = Process(target=compute_regret_list,
                  args=(repeated_num, best_expect_y, newGraph, "ucb", ucb_queue))
thread4 = Process(target=compute_regret_list,
                  args=(repeated_num, best_expect_y, newGraph, "eps-greedy", eps_greedy_queue, 1 / 8))

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread1.join()
thread2.join()
thread3.join()
thread4.join()

lr_regret_list = lr_queue.get()
ofu_regret_list = ofu_queue.get()
ucb_regret_list = ucb_queue.get()
eps_greedy_regret_list = eps_greedy_queue.get()

plt.rcParams['figure.figsize'] = (5.2, 3.7)
clrs = sns.color_palette("husl", 7)

tmp = [ofu_regret_list[int((len(ofu_regret_list) / 5)) * (i + 1)] for i in range(5)]
tmp = [0] + tmp
plt.scatter(np.arange(6) * (T / 5), tmp, marker='+', label="BGLM-OFU-Unknown", c=clrs[0])
plt.plot(np.arange(int(T / 100 + 1)) * 100, ofu_regret_list, c=clrs[0])

tmp = [lr_regret_list[int((len(lr_regret_list) / 5)) * (i + 1)] for i in range(5)]
tmp = [0] + tmp
plt.scatter(np.arange(6) * (T / 5), tmp, marker='x', label="BLM-LR-Unknown", c=clrs[1])
plt.plot(np.arange(int(T / 100 + 1)) * 100, lr_regret_list, c=clrs[1])

tmp = [ucb_regret_list[int((len(ucb_regret_list) / 5)) * (i + 1)] for i in range(5)]
tmp = [0] + tmp
plt.scatter(np.arange(6) * (T / 5), tmp, marker='o', label="Standard UCB", c=clrs[2])
plt.plot(np.arange(int(T / 100 + 1)) * 100, ucb_regret_list, c=clrs[2])

tmp = [eps_greedy_regret_list[int((len(eps_greedy_regret_list) / 5)) * (i + 1)] for i in range(5)]
tmp = [0] + tmp
plt.scatter(np.arange(6) * (T / 5), tmp, marker='D', label=r"$\epsilon$-Greedy", c=clrs[3])
plt.plot(np.arange(int(T / 100 + 1)) * 100, eps_greedy_regret_list, c=clrs[3])

plt.xlabel("Round Number")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.savefig("./results/" + str(T) + ".png")
plt.show()
