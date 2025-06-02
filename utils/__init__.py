
def get_exp_name(
    loss,
    backbone,
    weights,
    seg_dist,
    dataset,
):  
    exp_name = f'{dataset}_{loss}_{backbone}_{seg_dist}_{",".join(list(map(str, weights)))}'
    print(exp_name)
    return exp_name