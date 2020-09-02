from scipy.io import loadmat
import argparse
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser("scoring")
    parser.add_argument("--dataset", help='Dataset path')
    parser.add_argument("--num_sims", type=int, default=5, help="Simulations per node")
    parser.add_argument("--iters", type=int, default=200, help="Steps of simulation")
    parser.add_argument("--beta", type=float, default=0.05, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=0.005, help="Death rate")
    parser.add_argument("--out_file", help="Out file filename")
    args = parser.parse_args()

    network = loadmat(args.dataset)["graph"]
    graph = nx.from_scipy_sparse_matrix(network)
    model = ep.SIRModel(graph)

    cfg = mc.Configuration()
    cfg.add_model_parameter('beta', args.beta)
    cfg.add_model_parameter('gamma', args.gamma)
    model.set_initial_status(cfg)

    with open(args.out_file, "w") as file:
        for i in range(network.shape[0]):
            for j in range(args.num_sims):
                model.reset([i])
                iterations = model.iteration_bunch(args.iters)
                trends = model.build_trends(iterations)
                data = trends[0]['trends']['node_count'][1]
                max_point = np.argmax(data)
                file.write(str(max_point) + " " + str(data[max_point]) + "\n")
