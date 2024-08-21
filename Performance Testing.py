import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


def addEdge(adj, v, w):
    adj[v].append(w)
    adj[w].append(v)
    return adj


def greedyColoring(adj, V):
    result = [-1] * V
    result[0] = 0
    available = [False] * V


    for u in range(1, V):
        for i in adj[u]:
            if result[i] != -1:
                available[result[i]] = True


        cr = 0
        while cr < V:
            if not available[cr]:
                break
            cr += 1


        result[u] = cr


        for i in adj[u]:
            if result[i] != -1:
                available[result[i]] = False


    return result


def generate_random_instance(num_vertices, max_degree):
    graph = [[] for _ in range(num_vertices)]
    for i in range(num_vertices):
        num_edges = random.randint(1, max_degree)
        edges = random.sample(range(num_vertices), num_edges)
        for j in edges:
            if i != j and j not in graph[i]:
                graph = addEdge(graph, i, j)
    return graph


def measure_time(adj, V, num_runs=10):
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        greedyColoring(adj, V)
        end_time = time.time()
        times.append(end_time - start_time)
    return times


def compute_confidence_interval(data, confidence=0.90):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    h = std_err * t.ppf((1 + confidence) / 2., n - 1)
    return mean, h


# Experimental analysis
num_samples = 20
results = []


for num_vertices in range(10, 101, 10):
    for max_degree in range(2, 5):
        times = []
        for _ in range(num_samples):
            graph = generate_random_instance(num_vertices, max_degree)
            run_times = measure_time(graph, num_vertices)
            times.extend(run_times)


        mean_time, conf_interval = compute_confidence_interval(times)
        results.append((num_vertices, max_degree, mean_time, conf_interval))


# Plotting results
vertices = [r[0] for r in results]
mean_times = [r[2] for r in results]
conf_intervals = [r[3] for r in results]


plt.errorbar(vertices, mean_times, yerr=conf_intervals, fmt='-o', ecolor='r', capsize=5)
plt.xlabel('Number of Vertices')
plt.ylabel('Mean Execution Time (s)')
plt.title('Greedy Coloring Algorithm Performance')
plt.show()