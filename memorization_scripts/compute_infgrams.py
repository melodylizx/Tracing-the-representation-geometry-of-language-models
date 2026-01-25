import requests
import time, os, glob
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

save_dir = os.path.join(os.environ['SCRATCH'], 'llm_dynamics','results','infgram')

def main():
    tokenizer = AutoTokenizer.from_pretrained("shauray/Llava-Llama-2-13B-hf", add_bos_token=False, add_eos_token=False)

    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    try:
        files = glob.glob(os.path.join(save_dir,'triviaqa_*.npy'))
        files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('samples')[0]))
        fname = files[-1]
        print(f"Trying to resume from {fname}")
        res_dict = np.load(fname, allow_pickle=True).item()
    except: #add error
        res_dict = {}

    for idx, sample in enumerate(tqdm(ds['validation'])):
        if idx in res_dict: continue
        question_x = sample['question']
        answer_y = sample['answer']['value']

        # question_x = "Feel Like Making Love and The First Time Ever I Saw Your Face were hit singles for which female artist?"
        # answer_y = "Roberta Flack"
        if question_x[-1] == '?':
            instruction_text = f"Question: {question_x} Answer: {answer_y}"
        else:
            instruction_text = f"Question: {question_x}? Answer: {answer_y}"

        instruction_tokens = tokenizer.encode(instruction_text)
        pre_answer_tokens = tokenizer.encode(instruction_text.split(f' {answer_y}')[0])
        payload = {
            'index': 'v4_piletrain_llama',
            # 'query_type': 'count',
            'query_type': 'infgram_prob',
            'query_ids': None,
        }

        start_idx = len(pre_answer_tokens)+1
        end_idx = len(instruction_tokens)+1

        # start_time = time.time()
        infgram_probs = []
        for i in range(start_idx, end_idx):
            payload['query_ids'] = instruction_tokens[:i]
            for api_try_count in range(1,1+5): # try 5 times to avoid API time out errors
                try:
                    result = requests.post('https://api.infini-gram.io/', json=payload).json()
                    assert 'message' not in result.keys() # API time out error
                    
                except AssertionError:
                    tqdm.write(f"Retrying index {i} for repeat {api_try_count}")
                    time.sleep(2)
                    
                if 'message' not in result:
                    # success in using API
                    break
            
            if api_try_count == 5:
                break
            infgram_probs.append(result['prob'])
        if api_try_count == 5:
            tqdm.write(f"Skipping sequence {idx} because of API errors!!!")
            continue
        # end_time = time.time()
        # print(infgram_probs)
        answer_probs = np.array(infgram_probs)
        answer_probs[answer_probs < 0] = 0
        if np.sum(answer_probs) == 0.0:
            logprob_result = np.nan
            prob_result = 0.0
            
        else:
            answer_probs[answer_probs == 0.0] = 1.0
            logprob_result = np.sum(np.log(answer_probs))
            prob_result = np.exp(logprob_result)

        # print(answer_probs)
        # print(f"loglikelihood: {logprob_result:.6f}, prob: {prob_result:.6f}")
        # print(f"time taken = {end_time-start_time}")
        res_dict[idx] = {
            "Question": question_x,
            "Answer": answer_y,
            "infgram_prob": infgram_probs,
            "logprob_result": logprob_result,
            "prob_result": prob_result
        }

        if (1+idx)%100 == 0 or idx==len(ds['validation'])-1:
            np.save(os.path.join(save_dir, f'triviaqa_{1+idx}samples.npy'), res_dict)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)