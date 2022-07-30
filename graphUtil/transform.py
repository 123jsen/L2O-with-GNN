import torch

def network_to_nodes(model):
    '''Pass in a neural network model with gradients included'''
    out = torch.empty(0)

    for m_key in model._modules:
        m1 = model._modules[m_key]
        for p_key in m1._parameters:
            out = torch.cat((out, m1._parameters[p_key].grad.reshape(-1)))

    return out
    

def network_to_edge(model):
    pass

def nodes_to_network(model, node_y):
    '''Pass in model structure and predicted updates to change update back into same shape as parameter, as a update step dictionary'''

    out = dict()

    for m_key in model._modules:
        m1 = model._modules[m_key]
        m_dict = dict()     # initialize empty dictionary
        for p_key in m1._parameters:
            num = torch.numel(m1._parameters[p_key])
            m_dict[p_key] = node_y[:num]
            node_y = node_y[num:]

        out[m_key] = m_dict

    return out

