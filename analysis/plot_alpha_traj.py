import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import seaborn
seaborn.set_style('whitegrid')

def plot_alpha_model_training(model_name: str, 
                              color: str = 'k', ls: str = '-', marker: str = 'o'):
    res = np.load(os.path.join('results',f'results_{model_name}.npy'),allow_pickle=True).item()
    step_nums = list(res.keys())
    ys = [res[s]['alpha'] for s in step_nums]
    plt.plot(step_nums, ys, marker=marker, color=color, ls=ls, label=f'{model_name}')

def plot_alpha_model_training_flops(model_name: str, 
                              color: str = 'k', ls: str = '-', marker: str = 'o'):
    res = np.load(os.path.join('results',f'results_{model_name}.npy'),allow_pickle=True).item()
    step_nums = list(res.keys())
    num_params_str = model_name.split('pythia-')[-1].split('-deduped')[0]
    num_params_val = float(num_params_str[:-1])
    num_params_order = 1e6 if num_params_str[-1]=='m' else 1e9
    flops_per_step = num_params_val*num_params_order
    ys = [res[s]['alpha'] for s in step_nums]
    xs = [flops_per_step*s for s in step_nums]
    plt.plot(xs, ys, marker=marker, color=color, ls=ls, lw=3, label=f'{model_name}')

def plot_alpha_model_training_isoflops(model_name: str, 
                              marker: str = 'o', max_flops: int = 2e15):
    res = np.load(os.path.join('results',f'results_{model_name}.npy'),allow_pickle=True).item()
    step_nums = list(res.keys())
    num_params_str = model_name.split('pythia-')[-1].split('-deduped')[0]
    num_params_val = float(num_params_str[:-1])
    num_params_order = 1e6 if num_params_str[-1]=='m' else 1e9
    flops_per_step = num_params_val*num_params_order
    # ys = [res[s]['alpha'] for s in step_nums]
    # xs = [flops_per_step]*len(ys)
    # cmap = cm.get_cmap(name='plasma', lut=256)
    # flops_mapper = [np.log(flops_per_step*(s+1))/np.log(max_flops) for s in step_nums]
    # plot_colors = [cmap(f) for f in flops_mapper]
    isoflops = np.array([1e13, 5e13, 1e14])
    isoflops_steps = (isoflops/flops_per_step//1000*1000).astype(int)
    plot_steps = [s for s in isoflops_steps if s<max(step_nums) and s>min(step_nums)]
    ys = [res[s]['alpha'] for s in plot_steps]
    xs = [flops_per_step]*len(ys)
    plot_colors = colors[:len(plot_steps)]

    # tqdm.write(f'{model_name} {flops_per_step} {flops_mapper[-1]} {flops_mapper[0]}')
    plt.scatter(xs, ys, marker=marker, color=plot_colors) #, label=f'{model_name}')

model_names = ['pythia-14m', 'pythia-31m', 
               'pythia-70m', 'pythia-70m-deduped', 
               'pythia-160m', 'pythia-160m-deduped', 
               'pythia-410m', 'pythia-410m-deduped',
               'pythia-1b', 'pythia-1b-deduped',
               'pythia-1.4b', 'pythia-1.4b-deduped',
               'pythia-2.8b', 'pythia-2.8b-deduped',
               'pythia-6.9b', 'pythia-6.9b-deduped',
               'pythia-12b', 'pythia-12b-deduped',
               ]

colors = ['turquoise', 'cornflowerblue', 
          'dodgerblue', 'dodgerblue',
          'gold', 'gold', 
          'lime', 'lime',
          'darkgreen', 'darkgreen',
          'magenta', 'magenta',
          'deeppink', 'deeppink',
          'purple', 'purple',
          'brown', 'brown',
          ]

filter_model_names = [
                    # 'pythia-410m', 'pythia-410m-deduped',
                    # 'pythia-1.4b', 'pythia-1.4b-deduped',
                    # 'pythia-2.8b', 'pythia-2.8b-deduped',
                    # 'pythia-6.9b', 'pythia-6.9b-deduped',
                    # 'pythia-12b', 'pythia-12b-deduped',
    ]

def main(xvar: str = 'steps'):
    for midx, model_name in enumerate(tqdm(model_names)):
        if len(filter_model_names) and model_name not in filter_model_names: continue
        color = colors[midx]
        ls = '--' if 'deduped' in model_name else '-'
        marker = '*' if 'deduped' in model_name else 's'
        try:
            if xvar=='steps':
                plot_alpha_model_training(model_name, color=color, ls=ls, marker='')
            elif xvar=='flops':
                plot_alpha_model_training_flops(model_name, color=color, ls=ls, marker='')
            elif xvar=='isoflops':
                plot_alpha_model_training_isoflops(model_name, marker=marker)
            else:
                raise NotImplementedError
        except:
            continue

    plt.xscale('log')
    # plt.xlim([-1,5000])
    if xvar=='steps':
        plt.xlabel('Steps', fontsize=14)
    elif xvar=='flops':
        plt.xlabel('Flops', fontsize=14)
    elif xvar=='isoflops':
        plt.xlabel('Parameters', fontsize=14)
        # plt.colorbar()
    else:
        raise NotImplementedError
    plt.ylabel(r'$\alpha$', fontsize=14)
    plt.legend()
    plt.show()

if __name__=='__main__':
    from jsonargparse import CLI

    CLI(main)
    # To run (e.g.): 
    #  python analysis/plot_alpha_traj.py --xvar flops