from trainUtil import zero_gradients
from graphUtil import nodes_to_network, network_to_edge, network_to_nodes

import torch
from torch import nn as nn


loss_fn = nn.CrossEntropyLoss()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: Make this depend on the network
edge_index = network_to_edge([28 * 28, 64, 64, 10]).to(device)

def update_weights_gnn(model, update_fn, hidden):
    with torch.no_grad():
        # This parts create update dictionary using gradients and gnn l2o optimizer
        graph_x = network_to_nodes(model, device)
        update, hidden = update_fn(graph_x, hidden, edge_index)
        update = nodes_to_network(model, update)

        for m_key in model._modules:
            m1 = model._modules[m_key]
            update_module = update[m_key]
            for p_key in m1._parameters:
                m1._parameters[p_key] -= update_module[p_key]
    
    return hidden

def train_batch(dataloader, model, loss_fn, gnn_optimizer, history_step, print_step):
    h = None
    size = len(dataloader.dataset)
    history = []
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Preprocessing
        X, y = X.to(device), y.to(device)

        # Forward Pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        zero_gradients(model)
        loss.backward()
        
        h = update_weights_gnn(model, gnn_optimizer, h)

        loss = loss.item()

        if batch % print_step == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

        if batch % history_step == 0:
            history.append(loss)

    return history

def train_with_GNN(optimizer, optimizee_model, train_dataloader, num_epochs, history_step=15, print_step=100):
    gnn_hist = []
    for i in range(num_epochs):
        print(f"Epoch: {i + 1}")
        loss = train_batch(
            train_dataloader,
            optimizee_model,
            loss_fn,
            optimizer,
            history_step,
            print_step)

        gnn_hist.append(loss)

    return gnn_hist
