from unirec import unirecError

def one_train_sample(model, uid, nxt_idx, dataset, seqlen, n_items,
                     **kwargs):
    if model == 'unirec':
        from unirec.data.samplers.unirec_sampler import train_sample
    else:
        raise unirecError(f'Not support train sampler for {model} model')
    return train_sample(uid, nxt_idx, dataset, seqlen, n_items,
                        **kwargs)


def one_test_sample(model, uid, dataset, seqlen, n_items, **kwargs):
    if model == 'unirec':
        from unirec.data.samplers.unirec_sampler import test_sample
    else:
        raise unirecError(f'Not support test sampler for {model} model')
    return test_sample(uid, dataset, seqlen, n_items, **kwargs)
