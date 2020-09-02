import numpy as np
import torch
import time
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error


def get_evaluation(adj, Y, graph_features, NN, train_indices, test_indices):
    y = Y
    assert adj.shape[1] == len(y)
    cx = adj.tocoo()
    pairs = []
    
    for i, j, v in zip(cx.row, cx.col, cx.data):
        pairs.append([i, j])
        
    coolists = np.array(pairs).T
    graph_features = torch.from_numpy(graph_features)
    device = "cpu"
    y = torch.from_numpy(y).to(device).type(torch.FloatTensor)

    mask = np.zeros(len(y), dtype=bool)
    val = int(len(train_indices)*0.1)
    val_indices = train_indices[0:val]
    train_indices = train_indices[val:]
    train_mask = mask.copy()
    train_mask[train_indices] = True
    val_mask = mask.copy()
    val_mask[val_indices] = True
    test_mask = mask.copy()
    test_mask[test_indices] = True
    test_mask = torch.from_numpy(test_mask)
    train_mask = torch.from_numpy(train_mask)
    val_mask = torch.from_numpy(val_mask)
    edge_index = torch.from_numpy(coolists)

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    edge_index = edge_index.to(device).long()
    graph_features = graph_features.to(device).float()
    model = NN(edge_index, graph_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    def train():
        model.train()
        model.zero_grad()
        otp = model()[train_mask]
        out = y[train_mask]
        F.mse_loss(otp, out).backward()
        optimizer.step()

    def test():
        model.eval()
        logits, accs = model(), []
        true_labs = y[test_mask]
        pred = logits[test_mask]

        return mean_squared_error(true_labs, pred.detach().numpy())

    def testv():
        model.eval()
        logits, accs = model(), []
        true_labs = y[val_mask]
        pred = logits[val_mask]

        return mean_squared_error(true_labs, pred.detach().numpy())

    start = time.time()
    best_val = 1000
    best_mse = 1000
    for epoch in range(1, 200):
        train()
        mse = test()
        vls = testv()
        if vls < best_val:
            best_mse = mse
            best_val = vls
        print("Iteration:", epoch, " Test mse: ", mse, " Val mse: ", vls, " Time: ", time.time()-start)
    return best_mse
