import torch
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
# These imports assume the project root is in PYTHONPATH
# You can add it by running: export PYTHONPATH=$PYTHONPATH:/path/to/recurrent-rebels
from big_model.utils import load_embeddings, extract_features, get_device, time_transform, log_transform_plus1
from big_model.model import FullModel
import json

def extract_domain_info(url):
    """Extract domain and TLD from URL"""
    parsed = urlparse(url)
    domain = parsed.netloc
    parts = domain.split('.')
    if len(parts) > 1:
        tld = parts[-1]
    else:
        tld = 'com'  # default
    return domain, tld

def process_title(title, w2i, embedding_matrix):
    """Process title into embedding"""
    words = title.lower().split()
    word_indices = [w2i.get(word, 0) for word in words]  # 0 is UNK token
    if not word_indices:
        word_indices = [0]
    
    word_embeddings = embedding_matrix[word_indices]
    title_embedding = torch.mean(word_embeddings, dim=0)
    return title_embedding

def prepare_features(post_dict, features_vec, w2i, embedding_matrix, domain_to_idx, tld_to_idx, user_to_idx):
    """Prepare all features for a single post"""
    # Process title
    title_emb = process_title(post_dict['title'], w2i, embedding_matrix)
    
    # Process time
    timestamp = datetime.fromtimestamp(post_dict['time'])
    year, hour_angle, dow_angle, day_angle = time_transform(timestamp)
    
    # Get domain and TLD indices
    domain, tld = extract_domain_info(post_dict['url'])
    domain_idx = domain_to_idx.get(domain, 0)  # 0 for unknown
    tld_idx = tld_to_idx.get(tld, 0)  # 0 for unknown
    
    # Get user index
    user_idx = user_to_idx.get(post_dict['by'], 0)  # 0 for unknown

    # Create numerical features tensor (empty but with correct shape for concatenation)
    features_num = torch.tensor(features_vec, dtype=torch.float32)  # Shape: [batch_size=1, features=0]

    # Reshape title embedding to 2D
    title_emb = title_emb.unsqueeze(0)  # Shape: [batch_size=1, embedding_dim]
    
    # Reshape indices to 2D
    domain_idx = torch.tensor([domain_idx], dtype=torch.long)  # Shape: [batch_size=1]
    tld_idx = torch.tensor([tld_idx], dtype=torch.long)  # Shape: [batch_size=1]
    user_idx = torch.tensor([user_idx], dtype=torch.long)  # Shape: [batch_size=1]
    
    return features_num, title_emb, domain_idx, tld_idx, user_idx

def predict_single_post(post_dict, model_path):
    """Make prediction for a single HN post"""
    device = get_device()
    
    # Load vocab sizes from vocab file
    with open("data/train_vocab.json", 'r') as f:
        vocabs = json.load(f)
    
    domain_vocab_size = len(vocabs['domain_vocab'])
    tld_vocab_size = len(vocabs['tld_vocab'])
    user_vocab_size = len(vocabs['user_vocab'])
    
    # Load model
    model = FullModel(
        vector_size_num=0,  # Set to 0 to match checkpoint dimensions
        vector_size_title=200,
        scale=3,
        domain_vocab_size=domain_vocab_size,
        tld_vocab_size=tld_vocab_size,
        user_vocab_size=user_vocab_size
    ).to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load embeddings and mappings
    w2i, embedding_matrix = load_embeddings("skipgram_models/silvery200.pt")
    domain_to_idx = vocabs['domain_vocab']
    tld_to_idx = vocabs['tld_vocab']
    user_to_idx = vocabs['user_vocab']
    
    # Prepare features
    features_num, title_emb, domain_idx, tld_idx, user_idx = prepare_features(
        post_dict, w2i, embedding_matrix, domain_to_idx, tld_to_idx, user_to_idx
    )
    
    # Move tensors to device
    features_num = features_num.to(device)
    title_emb = title_emb.to(device)
    domain_idx = domain_idx.to(device)
    tld_idx = tld_idx.to(device)
    user_idx = user_idx.to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(features_num, title_emb, domain_idx, tld_idx, user_idx)
    
    # Convert prediction back to original scale
    prediction = 10 ** prediction.item() - 1
    return prediction

if __name__ == '__main__':
    # Example usage
    example_post = {
        'by': 'example_user',
        'title': 'Example Hacker News Post',
        'url': 'https://example.com/article',
        'time': int(datetime.now().timestamp())
    }
    
    prediction = predict_single_post(example_post, 'big_model/models/20250612_191343/best_model_1.pth')
    print(f"Predicted score: {prediction:.2f}") 