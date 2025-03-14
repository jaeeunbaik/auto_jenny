import os

import torch

def last_checkpoints(path):
    states = torch.load(path, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
    return states

def average_checkpoints(last):
    avg = None
    for path in last:
        states = torch.load(path, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
        states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(last)
            else:
                avg[k] //= len(last)
    return avg


def ensemble():
    path = ['/home/nas4/user/yh/auto_avsr/outputs/2023-09-06/20-14-46/lrs3/testaudio/epoch=70.ckpt']
    #model_path = os.path.join(
    #    args.exp_dir, args.exp_name, f"model_avg_{args.checkpoint.save_top_k}.pth"
    #)

    model_path_last = os.path.join(
        '/home/nas4/user/yh/auto_avsr/outputs/2023-09-06/20-14-46/lrs3/testaudio', f"model_last.pth"
    )

    #torch.save(average_checkpoints(last), model_path)
    torch.save(average_checkpoints(path), model_path_last)
    return model_path_last

print("start")

temp = ensemble()
print(temp)