import torch

@torch.no_grad()
def batch_eval(gru, test_data, cutoff=[20], batch_size=512, mode='conservative', item_key='ItemId', session_key='SessionId', time_key='Time'):
    if gru.error_during_train: 
        raise Exception('Attempting to evaluate a model that wasn\'t trained properly (error_during_train=True)')
    recall = dict()
    mrr = dict()
    for c in cutoff:
        recall[c] = 0
        mrr[c] = 0
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))
    n = 0
    ii = 0
    n_valid = batch_size
    for d in gru.data_iterator.iter_data(test_data, batch_size, item_key=item_key, session_key=session_key, time_key=time_key, session_order=None):
        if len(d) == 2:
            for h in H: h.detach_()
            O = gru.model.forward(d[0], H, None, training=False)
            oscores = O.T
            tscores = torch.diag(oscores[d[1]])
            if mode == 'standard': ranks = (oscores > tscores).sum(dim=0) + 1
            elif mode == 'conservative': ranks = (oscores >= tscores).sum(dim=0)
            elif mode == 'median':  ranks = (oscores > tscores).sum(dim=0) + 0.5*((oscores == tscores).dim(axis=0) - 1) + 1
            else: raise NotImplementedError
            for c in cutoff:
                recall[c] += (ranks <= c).sum().cpu().numpy()
                mrr[c] += ((ranks <= c) / ranks.float()).sum().cpu().numpy()
            n += n_valid
            ii += 1
        else:
            n_valid, finished_mask, valid_mask = d
            for i in range(len(gru.layers)):
                H[i][finished_mask] = 0
            if n_valid < len(valid_mask):
                for i in range(len(H)):
                    H[i] = H[i][valid_mask]
    for c in cutoff:
        recall[c] /= n
        mrr[c] /= n
    return recall, mrr
