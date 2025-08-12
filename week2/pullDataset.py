from datasets import load_dataset

def pull_ms_marco_dataset():
    """
    Pulls the MSMARCO dataset from Hugging Face and saves it as a Parquet file.
    """
    print("Pulling MSMARCO dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    dataset.to_parquet("ms_marco_train.parquet")
    print("MSMARCO dataset saved as 'msmarco_train.parquet'.")

pull_ms_marco_dataset()