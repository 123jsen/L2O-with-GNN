# This file contains code that actually trains the GNN model
# Allows the training to be executed as a function, instead of a notebook

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

from models.optim_nets import gnn_l2o_optimizer
from models.mnist_nets import class_net
from graphUtil import network_to_edge, network_to_nodes, nodes_to_network
from trainUtil import init_sequence, reset_model_computational_graph, zero_gradients

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

from math import ceil

# Constants
in_size = 28 * 28
out_size = 10

## Meta-Hyperparameters
unroll_len = 30
batch_size = 128
num_epochs = 2

## Optimizer Model
update_fn = gnn_l2o_optimizer().to(device)
meta_optimizer = torch.optim.Adam(update_fn.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
num_batches = ceil(len(training_data) / batch_size)

edge_index = network_to_edge([in_size, 32, out_size]).to(device)

def init_hidden():
    # Since we don't do updates individually for every parameter,
    # hidden weights become a simple vector tensor.
    return None

def update_optimizee_and_copy(old_model, new_model, hidden):
    # This function is specific to GNN

    # This parts create update dictionary using gradients and gnn l2o optimizer
    graph_x = network_to_nodes(old_model, device)
    update, hidden = update_fn(graph_x, hidden, edge_index)
    update = nodes_to_network(old_model, update)

    for m_key in old_model._modules:
        m1, m2 = old_model._modules[m_key], new_model._modules[m_key]
        update_m = update[m_key]
        for p_key in m1._parameters:
            m2._parameters[p_key] = m1._parameters[p_key].detach() - update_m[p_key] 
            m2._parameters[p_key].requires_grad_()
            m2._parameters[p_key].retain_grad()
    
    return hidden

def backprop_on_optimizer(total_loss):
    # Outer Loop Backprop
    meta_optimizer.zero_grad()
    total_loss.backward()
    meta_optimizer.step()

def inner_pass(h, models_t):
    total_loss = 0
    for i, (X, y) in enumerate(train_dataloader):
        iter = i % unroll_len

        if (num_batches - i) < unroll_len:
            # End prematurely if remaining batches not enough for unroll length
            reset_model_computational_graph(models_t, class_net(in_size, out_size).to(device))

            h = (h[0].detach(), h[1].detach())
            h[0].requires_grad_()
            h[1].requires_grad_()

            break

        # Preprocessing
        X = X.reshape(-1, 28 * 28)
        X, y = X.to(device), y.to(device)

        # Forward Pass
        pred = models_t[iter](X)
        loss = loss_fn(pred, y)
        total_loss = total_loss + loss

        if (i % 125) == 0:
            print(f"Batch {i} / {num_batches}, Model loss: {loss}")

        if iter == unroll_len - 1:
            backprop_on_optimizer(total_loss)
            reset_model_computational_graph(models_t, class_net(in_size, out_size).to(device))

            h = (h[0].detach(), h[1].detach())
            h[0].requires_grad_()
            h[1].requires_grad_()

            total_loss = 0

        else:
            # Backprop
            zero_gradients(models_t[iter])
            loss.backward(retain_graph=True)
            models_t[iter+1] = class_net(28*28, 10).to(device)  # Initialize a new model
            h = update_optimizee_and_copy(
                old_model=models_t[iter], new_model=models_t[iter+1], hidden=h)

def outer_pass():
    models_t = init_sequence(class_net(in_size, out_size).to(device), unroll_len)
    h = init_hidden()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        inner_pass(h, models_t)

def trainGNN(num_optimizee, out_path):
    for count in range(num_optimizee):
        print(f"{count}-th optimizee")
        outer_pass()

    torch.save(update_fn.state_dict(), out_path)