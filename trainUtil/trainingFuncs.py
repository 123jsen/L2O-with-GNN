def init_hidden(model):
    # Initializes a hidden state dictionary for every parameter value in the model.
    h = {}

    for m_key in model._modules:
        modules = model._modules[m_key]
        h_module = {}
        for p_key in modules._parameters:
            h_module[p_key] = None
        h[m_key] = h_module
    
    return h

def init_sequence(new_model, unroll_len):
    models_t = [None for _ in range(unroll_len)]
    models_t[0] = new_model
    return models_t

def reset_model_computational_graph(models_t, new_model):
    # Resets model sequence after training iterations
    # Assigns the starting model to be the prev ending model

    model_end = models_t[-1]
    model_new_start = new_model

    for m_key in model_end._modules:
        m1, m2 = model_end._modules[m_key], model_new_start._modules[m_key]
        for p_key in m1._parameters:
            m2._parameters[p_key] = m1._parameters[p_key].detach()
            m2.requires_grad_()

    models_t[0] = model_new_start

def reset_h_computational_graph(h_dict):
    # Resets computational graph of hidden state
    for m_key in h_dict:
        h_mod = h_dict[m_key]
        for p_key in h_mod:
            # Every h has two values, short term and long term memory
            h_mod[p_key] = (h_mod[p_key][0].detach(), h_mod[p_key][1].detach())
            
            h_mod[p_key][0].requires_grad_()
            h_mod[p_key][1].requires_grad_()

def zero_gradients(model):
    for m_key in model._modules:
        m1 = model._modules[m_key]
        for p_key in m1._parameters:
            # Shape for Batch input: (1, Num, 1)
            # Shape for Hidden State: (1, Num, 24)
            
            if m1._parameters[p_key].grad is not None:
                m1._parameters[p_key].grad.zero_()