import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import load_data, load_embeddings, get_device
from dataloader import PostDataset
from model import BigModel
import argparse
import os

# Hyperparameters (keep same as training)
BATCH_SIZE = 64
TIME_DECAY = 0.02

DEVICE = get_device()
EMBEDDING_FILE = "skipgram_models/silvery200.pt"
ITEMS_FILE = "data/items.parquet"
USER_FILE = "data/users.parquet"

def inverse_transform(log_pred):
    return np.exp(log_pred) - 1

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint['input_dim']
    
    model = BigModel(vector_size=input_dim, scale=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    w2i = checkpoint['w2i']
    embedding_matrix = checkpoint['embedding_matrix']
    
    return model, w2i, embedding_matrix

def prepare_validation_data(w2i, embedding_matrix):
    df = load_data(ITEMS_FILE, USER_FILE)
    df = df.drop(["by", "url","text", "created", "karma", 'max_score', 'min_score','mean_score', 'max_descendants', 'min_descendants', 'mean_descendants'], axis=1)
    df = df.sort_values(by="time").reset_index(drop=True)
    df = df.dropna()

    train_size = int(0.9 * len(df))
    val_df = df.iloc[train_size:]
    train_df = df.iloc[:train_size]
    T_ref = train_df['time'].max()

    val_dataset = PostDataset(val_df, embedding_matrix, w2i, ref_time=T_ref, lambda_=TIME_DECAY)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return val_dataloader, val_df

def run_inference(model, dataloader):
    predictions = []
    
    with torch.no_grad():
        for batch_x, _, _ in dataloader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            preds = outputs.cpu().numpy().flatten()
            predictions.extend(preds)
    
    predictions = np.array(predictions)
    scores = inverse_transform(predictions)
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint')
    parser.add_argument('--output_csv', type=str, default=None, help='Optional path to save predictions as CSV')
    args = parser.parse_args()

    model, w2i, embedding_matrix = load_model(args.model_path, DEVICE)
    val_dataloader, val_df = prepare_validation_data(w2i, embedding_matrix)
    
    preds = run_inference(model, val_dataloader)
    
    val_df = val_df.reset_index(drop=True)
    val_df['predicted_score'] = preds

    print(val_df[['score', 'predicted_score']].head(10))

    if args.output_csv:
        val_df.to_csv(args.output_csv, index=False)
        print(f"Predictions saved to {args.output_csv}")
