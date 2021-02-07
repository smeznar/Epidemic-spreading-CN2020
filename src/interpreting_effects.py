import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from evaluate_epidemics import generate_feature_vector
import shap

simulations_path = "../data/wiki_vote_sim/wiki_vote_"
network_path = "../data/soc-wiki-Vote.mat"
metric = "Score"  # One of Score/Time

node = []
score = []
time = []
for i in range(10):
    with open(simulations_path + str(i) + ".txt") as file:
        for line in file:
            l = line.strip().split(" ")
            node.append(int(l[0]))
            time.append(int(l[1]))
            score.append(int(l[2]))

adj = loadmat(network_path)["graph"]
df = pd.DataFrame(data={"Node": node, "Time": time, "Score": score})
df = df.groupby("Node").mean()

nonz = np.linalg.norm(adj.toarray(), ord=0, axis=0) > 0
high_score = df[metric].to_numpy()[nonz]
high_score = (high_score - high_score.min()) / (high_score.max() - high_score.min())

features = generate_feature_vector(adj)
features = normalize(features, axis=0)
features = features / np.max(features, axis=0)

data = xgb.DMatrix(data=features, label=high_score,
                   feature_names=["Average out degree", "PageRank", "Degree centrality", "Eigenvector centrality",
                                  "Hubs", "Authorities"])
model = xgb.train({},
                  data,
                  num_boost_round=200,
                  evals=[(data, 'train')])

index = np.random.randint(0, features.shape[0])

model_barr = model.save_raw()[4:]
model.save_raw = lambda: model_barr

explainer = shap.TreeExplainer(model)
dt = pd.DataFrame(data=features, columns=["Avg out", "PR", "Deg", "Eigen",
                                        "Hubs", "Auth"])
shap_values = explainer(dt)
shap.plots.waterfall(shap_values[index])
