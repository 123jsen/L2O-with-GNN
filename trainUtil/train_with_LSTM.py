import torch
from torch import nn as nn

from trainUtil import init_hidden, zero_gradients

loss_fn = nn.CrossEntropyLoss()

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def update_weights(model, update_fn, hidden):
    with torch.no_grad():
        for m_key in model._modules:
            m1 = model._modules[m_key]
            h_module = hidden[m_key]
            for p_key in m1._parameters:
                
                grad_in = m1._parameters[p_key].grad.reshape(1, -1, 1)

                update, h_module[p_key] = update_fn(grad_in, h_module[p_key])        
                update = update.reshape(m1._parameters[p_key].shape)
                
                m1._parameters[p_key] -= update 

def train_batch(dataloader, model, loss_fn, l2o_optimizer, history_step, print_step):
    h = init_hidden(model)
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
        update_weights(model, l2o_optimizer, h)

        loss = loss.item()

        if batch % print_step == 0:
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

        if batch % history_step == 0:
            history.append(loss)

    return history

def train_with_lstm(optimizer, optimizee_model, train_dataloader, num_epochs, history_step=15, print_step=100):
    l2o_hist = []
    for i in range(num_epochs):
        print(f"Epoch: {i + 1}")
        loss = train_batch(
            train_dataloader,
            optimizee_model,
            loss_fn,
            optimizer,
            history_step,
            print_step)

        l2o_hist.append(loss)
    
    return l2o_hist
