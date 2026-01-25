import os, sys, torch, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader
from utils import powerlaw
from data import wikitext_loader,sciq_loader,lam_loader


def compute_metrics_for_checkpoint(model_repo, model_name, revision, dataset_name, dataset_content_key, min_length, max_length, batch_size):
    model_path = f"{model_repo}/{model_name}"
    print(f"Running dataset {dataset_name} for model {model_path}/{revision}")

    try:
        step_num = int(revision.split('-tokens')[0].split('step')[-1])
    except ValueError as e:
        print(f"Could not parse revision {revision} for model {model_path}\n{e}")
        return None

    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision,
        torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, revision=revision,
        trust_remote_code=True
    )

    dataset = fineweb_loader.get_dataset()
    filtered_dataset = [s for s in tqdm(dataset[dataset_content_key])
                        if tokenizer(s, return_tensors="pt").input_ids.shape[-1] > min_length]

    activations_arr = []
    for bidx in tqdm(range(0, len(filtered_dataset), batch_size)):
        batch_prompts = filtered_dataset[bidx : bidx+batch_size]
        tokenized = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                              max_length=max_length, truncation=True)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        if model.device.type == "cuda":
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = np.arange(attention_mask.shape[0])
        activations = outputs.hidden_states[-1][batch_indices, last_indices, ...]
        activations_arr.append(activations.cpu().numpy())

    if not activations_arr:
        print("No valid features extracted.")
        return None

    all_activations = np.vstack(activations_arr)
    eigen = powerlaw.get_eigenspectrum(all_activations)
    rankme = powerlaw.rankme(eigen)
    alpha, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11, 100))

    return {
        'step': step_num,
        'checkpoint': revision,
        'eigenspectrum': eigen,
        'rankme': rankme,
        'alpha': alpha,
        'ypred': ypred,
        'r2': fit_r2,
        'r2_100': fit_r2_100
    }

def get_available_checkpoints(filepath: str) -> dict:
    print(f"🔍 Opening checkpoint file: {filepath}")
    checkpoint_map = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()  # Read all lines at once for debugging
            for line in lines:
                line = line.strip()
                if line.startswith("step") and "-tokens" in line:
                    step_str = line.split("-")[0].replace("step", "")
                    try:
                        step = int(step_str)
                        checkpoint_map[step] = line
                        #print(f"✔ Found checkpoint: step {step} -> {line}")
                    except ValueError:
                        print(f"✖ Skipping line (invalid step): {line}")
                        continue
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("Final checkpoint map:", checkpoint_map)
    return checkpoint_map

def run_all_checkpoints(model_repo="allenai", model_name="OLMo-1B", dataset_name="fineweb",
                        dataset_content_key="text", min_length=32, max_length=512, batch_size=32):
    model_path = f"{model_repo}/{model_name}"
    os.makedirs('olmo_metrics_results/mmlu', exist_ok=True)

    checkpoint_map = get_available_checkpoints("1b_revisions.txt")

    for step, rev in sorted(checkpoint_map.items()):
        print(f" - Step {step}: {rev}")

    all_steps = sorted(checkpoint_map.keys())
    step_nums = []
    for s in all_steps:
        if s <= 10000:
            step_nums.append(s)
        elif s % 10000 == 0:
            step_nums.append(s)
            
    step_nums = sorted(step_nums)

    tmp_save = f'olmo_metrics_results/mmlu/results_{model_name}_temp.npy'
    final_save = f'olmo_metrics_results/mmlu/results_{model_name}.npy'

    res_dict = {}
    if os.path.exists(final_save):
        try:
            res_dict = np.load(final_save, allow_pickle=True).item()
        except:
            pass

    if os.path.exists(tmp_save):
        try:
            temp = np.load(tmp_save, allow_pickle=True).item()
            res_dict.update(temp)
        except:
            pass

    completed = set(res_dict.keys())
    to_process = [s for s in step_nums if s not in completed]

    print(f"Processing {len(to_process)} remaining checkpoints...")
    for step in tqdm(to_process, desc="Calculating metrics"):
        revision = checkpoint_map[step]
        try:
            result = compute_metrics_for_checkpoint(model_repo, model_name, revision, dataset_name,
                                                    dataset_content_key, min_length, max_length, batch_size)
            if result:
                res_dict[step] = result
                tqdm.write(f"Step {step}: rankme={result['rankme']:.3f}, alpha={result['alpha']:.3f}, r2_100={result['r2_100']:.3f}")
                np.save(tmp_save, res_dict)
        except Exception as e:
            tqdm.write(f"Error at step {step}: {str(e)}")
            continue

    np.save(final_save, res_dict)
    if os.path.exists(tmp_save):
        os.remove(tmp_save)

    print(f"Completed! Results saved to {final_save}")

if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    CLI(run_all_checkpoints)

