import pandas as pd
from scipy.io import loadmat
import scipy.sparse as sparse
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
import numpy as np
from numba import jit, prange
from Gin import test_gin
from Gat import test_gat
import xgboost as xgb
import networkx as nx
import shap
from node2vec_main import main

simulations_path = "../data/wiki_vote_sim/wiki_vote_"
network_path = "../data/soc-wiki-Vote.mat"
metric = "Score"  # One of Score/Time

@jit(parallel=True, nogil=True, nopython=True)
def numba_walk_kernel(walk_matrix, node_name, sparse_pointers, sparse_neighbors, num_steps=3, num_walks=100):
    length = num_steps + 1
    for walk in prange(num_walks):
        curr = node_name
        offset = walk * length
        walk_matrix[offset] = node_name
        for step in prange(num_steps):
            num_neighs = sparse_pointers[curr+1] - sparse_pointers[curr]
            if num_neighs > 0:
                curr = sparse_neighbors[sparse_pointers[curr] + np.random.randint(num_neighs)]
            idx = offset+step+1
            walk_matrix[idx] = curr


def rank_nodes(network: sparse.csr_matrix, num_walks=1024, max_walk_length=10):
    samples = np.random.uniform(0, 1, num_walks)
    distribution = np.histogram(samples, max_walk_length)[0]
    sparse_pointers = network.indptr
    sparse_neighbors = network.indices
    hashes = []
    degree = Counter(network.nonzero()[0])
    degree = [degree[i] if i in degree else 0 for i in range(network.shape[0])]
    for i in range(network.shape[0]):
        generated_walks = []
        # Generate walks
        for j, num in enumerate(distribution):
            walk_matrix = -np.ones((num, (j + 2)), dtype=np.uint32, order='C')
            walk_matrix = np.reshape(walk_matrix, (walk_matrix.size,), order='C')
            numba_walk_kernel(walk_matrix, i, sparse_pointers, sparse_neighbors, num_steps=j + 1, num_walks=num)
            wm = walk_matrix.tolist()
            generated_walks += [np.mean([degree[node] for node in wm[k:k + num]]) for k in range(0, len(wm), num)]
        hashes.append(np.mean(generated_walks))
    return hashes

N2VEParams = {"dimensions": 128,  # Number of dimensions (features)
            "walk_length": 80,  # Length of the walk per source
            "num_walks": 10,  # Number of walks per source
            "window_size": 10,  # Context size for optimization
            "iter": 1,  # Number of epochs in SGD
            "workers": 8,  # Number of workers
            "p": 1,  # Return hyper-parameter
            "q": 1,  # Input hyper-parameter
            "is_weighted": False,  # Is graph weighted
            "is_directed": False}  # Is graph directed


def generate_feature_vector(graph_adj):
    graph = nx.from_scipy_sparse_matrix(graph_adj)
    avg_out_degree = rank_nodes(graph_adj)
    aod = [avg_out_degree[i] for i in range(graph_adj.shape[0])]
    pr_val = nx.pagerank_scipy(graph)
    pr = [pr_val[i] for i in range(len(graph.nodes))]
    deg_cent_val = nx.degree_centrality(graph)
    dc = [deg_cent_val[i] for i in range(len(graph.nodes))]
    eigh = nx.eigenvector_centrality(graph)
    haa = nx.hits(graph)
    ei = [eigh[i] for i in range(len(graph.nodes))]
    hubs = [haa[0][i] for i in range(len(graph.nodes))]
    auth = [haa[1][i] for i in range(len(graph.nodes))]
    return pd.DataFrame(data={"Average Out Degree": aod, "PageRank": pr, "Degree centrality": dc,
                              "Eigenvector centrality": ei, "Hubs": hubs, "Authorities": auth})

node = []
score = []
time = []
for i in range(5):
    with open(simulations_path+str(i)+".txt") as file:
        j = 0
        for line in file:
            node.append(j)
            j += 1
            l = line.strip().split(" ")
            time.append(int(l[0]))
            score.append(int(l[1]))

adj = loadmat(network_path)["graph"]

df = pd.DataFrame(data={"Node": node, "Time": time, "Score": score})
ef = df.copy()
df = df.groupby("Node").mean()
ff = df.reset_index()

for i in range(adj.shape[0]):
    ef.loc[ef["Node"] == i, metric] = ef[ef["Node"] == i][metric] - float(ff[ff["Node"] == i][metric])
ef.loc[:, metric] = abs(ef.loc[:, metric])
ef = ef.groupby("Node").mean()[metric].to_numpy()

nonz = np.linalg.norm(adj.toarray(), ord=0, axis=0) > 0
ef = ef[nonz]
adj = adj[nonz]
adj = adj[:, nonz]

high_score = df[metric].to_numpy()[nonz]
ef = (ef - high_score.min()) / (high_score.max() - high_score.min())
high_score = (high_score - high_score.min()) / (high_score.max() - high_score.min())
print("Simulation error (averaged)", metric, "MSE: ", mean_squared_error(ef, np.zeros(ef.shape)))

features = generate_feature_vector(adj)
features = normalize(features, axis=0)
features = features / np.max(features, axis=0)

nodes = np.array([i for i in range(adj.shape[0])])
rs = KFold(n_splits=5, shuffle=True, random_state=18)
results = []
results_ra = []
results_n2v = []
results_ga = []
results_gi = []

emb_ca = features

graph = nx.from_scipy_sparse_matrix(adj)
model = main(graph, N2VEParams)
emb_n2v = np.array([model.wv[str(node)] for node in graph.nodes])

emb_ra = np.random.random((emb_ca.shape[0], 64))

for x_test, x_train in rs.split(nodes):
    results_gi.append(test_gin(adj, features, high_score, x_train, x_test))
    results_ga.append(test_gat(adj, features, high_score, x_train, x_test))
    # XGBoost
    model = xgb.XGBRegressor()
    model.fit(emb_ca[x_train, :], high_score[x_train])
    preds = model.predict(emb_ca[x_test])
    results.append(mean_squared_error(high_score[x_test], preds))

    model = xgb.XGBRegressor()
    model.fit(emb_ra[x_train, :], high_score[x_train])
    preds = model.predict(emb_ra[x_test])
    results_ra.append(mean_squared_error(high_score[x_test], preds))

    model = xgb.XGBRegressor()
    model.fit(emb_n2v[x_train, :], high_score[x_train])
    preds = model.predict(emb_n2v[x_test])
    results_n2v.append(mean_squared_error(high_score[x_test], preds))

print("")
print("-----------------------------")
print("")
print("CABoost", "MSE:", np.mean(results), "VAR:", np.var(results), "STD:", np.std(results))
print("Random", "MSE:", np.mean(results_ra), "VAR:", np.var(results_ra), "STD:", np.std(results_ra))
print("node2vec", "MSE:", np.mean(results_n2v), "VAR:", np.var(results_n2v), "STD:", np.std(results_n2v))
print("GIN: ", "MSE:", np.mean(results_gi), "VAR:", np.var(results_gi), "STD:", np.std(results_gi))
print("GAT: ", "MSE:", np.mean(results_ga), "VAR:", np.var(results_ga), "STD:", np.std(results_ga))
print("")
print("-----------------------------")
print("")

# SHAP
data = xgb.DMatrix(data=emb_ca, label=high_score,
                   feature_names=["Average out degree", "PageRank", "Degree centrality", "Eigenvector centrality",
                                  "Hubs", "Authorities"])
model = xgb.train({},
                  data,
                  num_boost_round=200,
                  evals=[(data, 'train')])

index = np.random.randint(0, emb_ca.shape[0])

model_barr = model.save_raw()[4:]
model.save_raw = lambda: model_barr

explainer = shap.TreeExplainer(model)
dt = pd.DataFrame(data=emb_ca, columns=["Avg out", "PR", "Deg", "Eigen",
                                        "Hubs", "Auth"])
shap_values = explainer.shap_values(dt)
shap.waterfall_plot(explainer.expected_value, shap_values[index, :], dt.iloc[index, :])



