from transformers import AutoTokenizer
import torch
def decode(pathvar: str, seq: torch.Tensor)->str:
    tokenizer = AutoTokenizer.from_pretrained(pathvar)
    for i in range(len(seq)):
        tokens = tokenizer.convert_ids_to_tokens(seq[i])
        print(f"joined tokens are {tokens}")
    return "".join(tokens)


if __name__ == "__main__":
    pass