import torch

def network_to_nodes(model):
    '''Pass in a neural network model with gradients included'''
    out = torch.empty(0)

    for m_key in model._modules:
        m1 = model._modules[m_key]
        for p_key in m1._parameters:
            out = torch.cat((out, m1._parameters[p_key].grad.reshape(-1)))

    return out
    

def make_offsets(num_nodes_list):
    '''For example if the number of nodes per layer is (2, 3, 4), then the offsets are (0, 6, 9, 21)'''

    # Note: if there are (n + 1) layers, then there are 2n offsets
    
    list = [0]
    base = 0

    for i in range(1, len(num_nodes_list)):
        base += num_nodes_list[i-1] * num_nodes_list[i]
        list.append(base)

        if (i != len(num_nodes_list) - 1):
            base += num_nodes_list[i]
            list.append(base)
        
    return list

def network_to_edge(nlist, transpose=True):
    '''Produces PyG adjacency list from list of neuron count, see documentation for shape'''
    offsets = make_offsets(nlist)
    edge_index = torch.empty(0)

    # Loop through matrix to create edges forward
    for i in range(1, len(nlist)):
        for j in range(nlist[i]): # count row
            for k in range(nlist[i-1]): # counts index / column
                # Connect each row to corresponding entry of next bias vector
                new_edge = torch.tensor([[offsets[2 * i - 2] + k + j * nlist[i-1], offsets[2 * i - 1] + j]])
                edge_index = torch.cat((edge_index, new_edge))


    # Create edges backward
    for i in range(2, len(nlist)):
        for k in range(nlist[i-1]): # counts column
            for j in range(nlist[i]): # count row
                # Connect each column to corresponding entry of previous bias vector
                new_edge = torch.tensor([[offsets[2 * i - 2] + k + j * nlist[i-1], offsets[2 * i - 3] + k]])
                edge_index = torch.cat((edge_index, new_edge))


    if transpose:
        edge_index = edge_index.T
        
    return edge_index

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

