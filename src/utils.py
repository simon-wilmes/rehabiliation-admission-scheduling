def slice_dict(d, *conditions):
    return {
        k: v
        for k, v in d.items()
        if all(cond is None or k[i] == cond for i, cond in enumerate(conditions))
    }
