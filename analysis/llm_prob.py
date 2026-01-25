from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from typing import Generator, Dict, Any
import math


class TokenProbabilityAnalyzer:
    def __init__(self, model_name: str = "EleutherAI/pythia-410m"):
        """
        Initialize with a Pythia model from HuggingFace.
        
        Args:
            model_name: Name of the Pythia model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def get_token_probabilities(
        self, 
        input_text: str,
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
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get logits (raw scores) for each position
        logits = outputs.logits[0]  # Remove batch dimension
        
        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process each token position
        for pos in range(inputs.input_ids.shape[1]):
            # Get current token
            token_id = inputs.input_ids[0][pos].item()
            token_text = self.tokenizer.decode([token_id])
            
            # Get probability for the actual token
            token_prob = probs[pos][token_id].item()
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


import pickle
import os

from tqdm import tqdm

def process_triviaqa(analyzer, dataset_split="validation", output_file="triviaqa_probabilities.pkl"):
    """
    Process all examples from the TriviaQA dataset and save results to a pickle file.
    
    Args:
        analyzer: Instance of TokenProbabilityAnalyzer
        dataset_split: Which split of TriviaQA to use ("train", "validation")
        output_file: Path to the pickle file for saving results
    """
    # Load the TriviaQA dataset
    dataset = load_dataset("trivia_qa", "unfiltered")
    trivia_split = dataset[dataset_split]
    
    print(f"Processing the entire TriviaQA {dataset_split} split...\n")
    
    results = []  # Store all results in a list
    
    # Iterate over all samples with tqdm for progress tracking
    for idx, sample in enumerate(tqdm(trivia_split, desc="Processing Examples", unit="example")):
        question = sample["question"]
       
        # Safely extract the answer from the answer dictionary
        answers = sample.get("answer", {})
        if isinstance(answers, dict) and "value" in answers:
            answer = answers["value"]
        else:
            answer = "No answer available"
        
        input_text = f"{question} {answer}"
        
        example_data = {
            "example_id": idx + 1,
            "question": question,
            "answer": answer,
            "tokens": []
        }
        
        for prob_info in analyzer.get_token_probabilities(input_text):
            # Save token probability details in the result dictionary
            example_data["tokens"].append(prob_info)
        
        # Append to results
        results.append(example_data)
        
        # Save intermediate results every 10 examples
        if (idx + 1) % 10 == 0:
            with open(output_file, "wb") as f:
                pickle.dump(results, f)
    
    # Save the final results
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Processing complete. Results saved to {output_file}.\n")





def main():
    # Initialize the analyzer
    analyzer = TokenProbabilityAnalyzer("EleutherAI/pythia-410m")
    
    # Process TriviaQA dataset and save results
    output_file = "triviaqa_probabilities.pkl"
    process_triviaqa(analyzer, dataset_split="validation", output_file=output_file)



if __name__ == "__main__":
    main()
