from transformers import GPTNeoXForCausalLM, AutoTokenizer
import datasets
import torch.nn as nn
from tqdm import tqdm
from data import wikitext_loader
import numpy as np
from utils import powerlaw
import torch
import os

def prepare_dataset(dataset: datasets.arrow_dataset.Dataset,
                    tokenizer: AutoTokenizer,
                    content_key: str = "text",
                    min_length: int = 10,
                    max_length: int = 128,
                    ) -> list:
    encoded_dataset = []
    for seq in dataset:
        encoded_seq = tokenizer(seq[content_key], max_length=max_length,
                                       return_tensors="pt", truncation=True)
        try:
            if encoded_seq['input_ids'].shape[-1] > min_length:
                encoded_dataset.append(encoded_seq)
        except:
            continue
    
    print(f"Number of valid sequences tokenized: {len(encoded_dataset)}/{len(dataset)}")
    return encoded_dataset

def get_alpha(model_name: str, step_num: int, 
              dataset: datasets.arrow_dataset.Dataset) -> dict:    

    model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{model_name}",
                                            revision=f"step{step_num}")
    model.embed_out = nn.Identity()
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}",
                                            revision=f"step{step_num}")

    tokenized_dataset = prepare_dataset(dataset, tokenizer)

    activations_arr = []
    with torch.no_grad():
        for seq in tokenized_dataset:
            for k,v in seq.items():
                seq[k] = v.cuda()
            out = model(**seq)
            # print(out.logits.shape)
            activations_arr.append(out.logits[0,-1].cpu().numpy())

    activations_arr = np.array(activations_arr)
    eigen = powerlaw.get_eigenspectrum(activations_arr)
    alpha, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11,100))
    return {'eigenspectrum': eigen,
            'ypred': ypred, 
            'alpha': alpha,
            'r2': fit_r2, 
            'r2_100': fit_r2_100}


def main(model_name: str = "pythia-70m-deduped",
         dataset_name: str = "wikitext"):
    print(model_name, dataset_name)
    assert dataset_name in ['wikitext'], NotImplementedError

    dataset = wikitext_loader.get_dataset()

    # step_nums = [0,1,2,4,8,16,32,64,128,256,512] + list(np.arange(1000,143000+1,10000))
    step_nums = [0,8,16,32,64,128,256,512] + list(np.arange(1000,143000+1,1000))
    # step_nums = [0, 141000]

    tmp_save_fname = os.path.join('results',f'results_{model_name}_temp.npy')
    res_dict = {}
    for step_num in tqdm(step_nums):
        try:
            res_dict[step_num] = get_alpha(model_name, step_num, dataset)
            tqdm.write(f"Step {step_num}: alpha = {res_dict[step_num]['alpha']:.3f}, \
                    r2_100 = {res_dict[step_num]['r2_100']:.3f}")
        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")

        np.save(tmp_save_fname, res_dict)

    save_fname = os.path.join('results',f'results_{model_name}.npy')
    print(f"Saving results to {save_fname}")
    np.save(save_fname, res_dict)
    os.system(f'rm {tmp_save_fname}')   # removing tmp result file

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)