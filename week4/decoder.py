import transformers
import os, importlib.metadata as im
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def main():
    # export HF_HUB_ENABLE_HF_TRANSFER=1 in shell before running this first time
    print(
        "transfer?",
        os.getenv("HF_HUB_ENABLE_HF_TRANSFER"),
        "hub-ver",
        im.version("huggingface_hub"),
        "hf_transfer",
        im.version("hf_transfer"),
    )
    c = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    v = transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    ds = load_dataset("nlphuji/flickr30k", num_proc=8)
    # Download latest version
    # path = kagglehub.dataset_download("adityajn105/flickr30k")
    # print("Path to dataset files:", path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    main()
