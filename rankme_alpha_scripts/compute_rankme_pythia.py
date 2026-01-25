from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch.nn as nn
from tqdm import tqdm
from data import fineweb_loader
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


def get_rankme(model_path: str, step_num: int,
               dataset: datasets.arrow_dataset.Dataset) -> dict:
    base_dir = '/network/scratch/z/zixuan.li/160m-v2/'
    model_path = f"{base_dir}checkpoint-{step_num}"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.embed_out = nn.Identity()
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_dataset = prepare_dataset(dataset, tokenizer)

    activations_arr = []
    with torch.no_grad():
        for seq in tokenized_dataset:
            for k, v in seq.items():
                seq[k] = v.cuda()
            out = model(**seq)
            activations_arr.append(out.logits[0, -1].cpu().numpy())

    activations_arr = np.array(activations_arr)
    eigen = powerlaw.get_eigenspectrum(activations_arr)
    rankme = powerlaw.rankme(eigen)
    return {'eigenspectrum': eigen,
            'rankme': rankme}


def main(model_name: str = "pythia-70m-deduped",
         dataset_name: str = "fineweb"):
    assert dataset_name in ['fineweb'], NotImplementedError

    dataset = fineweb_loader.get_dataset()
    step_nums = list(range(5000, 1000000, 5000))
    tmp_save_fname = os.path.join('/network/scratch/z/zixuan.li/160m-v2/rankme_results', f'results_gpt2_temp.npy')

    # Load existing results if they exist
    if os.path.exists(tmp_save_fname):
        print(f"Loading existing results from {tmp_save_fname}")
        res_dict = np.load(tmp_save_fname, allow_pickle=True).item()
    else:
        res_dict = {}

    # Skip already computed checkpoints
    completed_steps = set(res_dict.keys())
    step_nums = [step for step in step_nums if step not in completed_steps]

    print(f"Remaining steps to compute: {len(step_nums)}")

    for step_num in tqdm(step_nums):
        try:
            res_dict[step_num] = get_rankme(model_name, step_num, dataset)
            tqdm.write(f"Step {step_num}: rankme = {res_dict[step_num]['rankme']:.3f}")
        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")

        np.save(tmp_save_fname, res_dict)

    save_fname = os.path.join('/network/scratch/z/zixuan.li/160m-v2/rankme_results', f'results_gpt2.npy')
    print(f"Saving final results to {save_fname}")
    np.save(save_fname, res_dict)
    os.system(f'rm {tmp_save_fname}')


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
