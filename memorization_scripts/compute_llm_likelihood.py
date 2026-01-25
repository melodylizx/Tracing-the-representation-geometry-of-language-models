from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Generator, Dict, Any
import math, os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


save_dir = os.path.join(os.environ['SCRATCH'], 'llm_dynamics','results','llm_likelihood')

class TokenProbabilityAnalyzer:
    def __init__(self, model_name: str = "EleutherAI/pythia-70m", revision: int = 10000):
        """
        Initialize with a Pythia model from HuggingFace.
        
        Args:
            model_name: Name of the Pythia model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                    revision=f"step{revision}").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_token_probabilities(
        self, 
        input_text: str,
        prompt_text: str = None,
        top_k: int = 5
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Get token probabilities for the input text.
        
        Args:
            input_text: The text to analyze
            top_k: Number of top alternative tokens to return
            
        Yields:
            Dictionary containing token and probability information
        """
        # Tokenize the input
        # breakpoint()
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        if prompt_text is not None:
            prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt")
            start_idx = prompt_tokens.input_ids.shape[1]-1
            end_idx = inputs.input_ids.shape[1]-1
        else:
            start_idx = 0
            end_idx = inputs.input_ids.shape[1]-1
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get logits (raw scores) for each position
        logits = outputs.logits[0]  # Remove batch dimension
        
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process each token position
        for pos in range(start_idx, end_idx):
            # Get current token
            token_id = inputs.input_ids[0][pos].item()
            next_token_id = inputs.input_ids[0][pos+1].item()
            token_text = self.tokenizer.decode([token_id])
            
            # Get probability for the actual token
            token_prob = probs[pos][next_token_id].item()
            token_logprob = math.log(token_prob)
            
            # Get top-k alternative tokens
            top_probs, top_indices = torch.topk(probs[pos], top_k)
            
            alternatives = []
            for alt_prob, alt_idx in zip(top_probs, top_indices):
                alt_text = self.tokenizer.decode([alt_idx])
                alternatives.append({
                    "text": alt_text,
                    "probability": alt_prob.item(),
                    "logprob": math.log(alt_prob.item())
                })
            
            yield {
                "position": pos,
                "token": token_text,
                "probability": token_prob,
                "logprob": token_logprob,
                "alternatives": alternatives
            }

def main(model_name="EleutherAI/pythia-410m", step=10000):
    print(model_name, step)
    # Example usage with the specific input
    # analyzer = TokenProbabilityAnalyzer("EleutherAI/pythia-410m")
    # input_text = "what is capital of america? washington DC"
    analyzer = TokenProbabilityAnalyzer(model_name=model_name, revision=step)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")

    res_dict = {}
    for idx, sample in enumerate(tqdm(ds['validation'])):
        question_x = sample['question']
        answer_y = sample['answer']['value']
        # question_x = "Feel Like Making Love and The First Time Ever I Saw Your Face were hit singles for which female artist?"
        # answer_y = "Roberta Flack"
        if question_x[-1] == '?':
            instruction_text = f"Question: {question_x} Answer: {answer_y}"
        else:
            instruction_text = f"Question: {question_x}? Answer: {answer_y}"
        input_text = instruction_text
        prompt_text = instruction_text.split(f' {answer_y}')[0]
        
        # print(f"\nAnalyzing token probabilities for: '{input_text}'\n")
        
        # for prob_info in analyzer.get_token_probabilities(input_text, prompt_text):
        #     print(f"\nPosition {prob_info['position']}:")
        #     print(f"Token: '{prob_info['token']}'")
        #     print(f"Probability: {prob_info['probability']:.6f}")
        #     print(f"Log Probability: {prob_info['logprob']:.6f}")
        #     print("Top Alternative Tokens:")
        #     for alt in prob_info['alternatives']:
        #         print(f"  '{alt['text']}' (prob: {alt['probability']:.4f}, logprob: {alt['logprob']:.4f})")
        # breakpoint()
        llm_probs = []
        llm_logprobs = []
        for prob_info in analyzer.get_token_probabilities(input_text=input_text, 
                                                        prompt_text=prompt_text):
            llm_probs.append(prob_info['probability'])
            llm_logprobs.append(prob_info['logprob'])
        
        logprob_result = np.sum(llm_logprobs)

        res_dict[idx] = {
            "Question": question_x,
            "Answer": answer_y,
            "llm_prob": llm_probs,
            "llm_logprob": llm_logprobs,
            "logprob_result": logprob_result,
        }

    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    np.save(os.path.join(model_save_dir, f'triviaqa_step_{step}.npy'), res_dict)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)