import argparse
import tiktoken
import torch
import torch.nn.functional as F

from models.gpt2 import GPT, GPTConfig
from src.utils.utils import get_device


class GPTInference:
    def __init__(
        self,
        checkpoint_path: str,
        device: str | None | None = None,
        temperature: float = 0.8,
        max_length: int = 100,
        top_k: int = 50,
    ):
        self.device = device or get_device()
        self.temperature = temperature
        self.max_length = max_length
        self.top_k = top_k
        
        # Initialize model and load checkpoint
        self.model = GPT(GPTConfig())
        self.model.to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
    def generate(
        self,
        prompt: str,
        max_length: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> str:
        # Use provided parameters or defaults
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_k = top_k or self.top_k
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate tokens
        while x.size(1) < max_length:
            with torch.no_grad():
                # Get predictions
                logits, loss = self.model(x)
                logits = logits[:, -1, :] / temperature
                
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                xcol = torch.gather(topk_indices, dim=-1, index=ix)
                x = torch.cat((x, xcol), dim=1)
        
        # Decode the generated sequence
        generated_ids = x[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained GPT-2 model')
    parser.add_argument('checkpoint_path', type=str, help='Path to the model checkpoint file')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Input prompt for text generation')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (default: 0.8)')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text (default: 100)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter (default: 50)')
    
    args = parser.parse_args()
    
    # Initialize inference
    gpt_inference = GPTInference(
        checkpoint_path=args.checkpoint_path,
        temperature=args.temperature,
        max_length=args.max_length,
        top_k=args.top_k,
    )
    
    # Generate text
    generated_text = gpt_inference.generate(args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Generated text: {generated_text}")


if __name__ == '__main__':
    main() 
