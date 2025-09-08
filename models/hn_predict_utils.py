import numpy as np
import pandas as pd
import tldextract
import torch
from datetime import datetime

TRAINING_VOCAB_PATH = "data/train_vocab.json"

global global_Tmin
global global_Tmax

global global_domain_vocab
global global_tld_vocab
global global_user_vocab
global global_embedding_matrix
global global_w2i

def load_data(items_file, users_file):
    raw_items = pd.read_parquet(items_file)
    raw_users = pd.read_parquet(users_file)
    merged = pd.merge(raw_items, raw_users, on="by", how="left", suffixes=("", "_user"))
    has_score = merged.dropna(subset=["score"])
    has_title = has_score[has_score["title"].notnull()]
    has_title = has_title[has_title["title"].str.strip().astype(bool)]  # drop empty or whitespace-only
    return has_title.drop(columns=["id"])


def load_embeddings(embeddings_file):
    edict = torch.load(embeddings_file, weights_only=True)
    embeddings = edict["embeddings"]
    word_to_ix = edict["word_to_ix"]
    word_to_ix['UNK'] = 0
    return word_to_ix, embeddings

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def log_transform_plus1(x):
    if x <= 0:
        return x
    else:
        return np.log10(x+1)

def time_transform(time):
    if isinstance(time, (int, float)):
        timestamp = datetime.fromtimestamp(time)
    else:
        timestamp = time
            
    year = timestamp.year    
    hour_angle = 2 * np.pi * timestamp.hour / 24
    dow_angle = 2 * np.pi * timestamp.weekday() / 7
    day_angle = 2 * np.pi * (timestamp.timetuple().tm_yday - 1) / 365
    return year, hour_angle, dow_angle, day_angle

def extract_features(row):
    # Time features
    year, hour_angle, dow_angle, day_angle = time_transform(row['time'])
    year_norm = (year - global_Tmin.year) / (global_Tmax.year - global_Tmin.year)

    time_feats = [
        year_norm,
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(dow_angle),
        np.cos(dow_angle),
        np.sin(day_angle),
        np.cos(day_angle),
        log_transform_plus1(row['num_posts'])
    ]

    # Collect all remaining user features (everything except those handled above
    user_feature_names = [
        col for col in row.keys()
        if col not in ['by', 'time', 'title', 'score', 'url', 'num_posts']
    ]

    user_feats = [row[col] for col in user_feature_names]

    all_features = np.array(time_feats + user_feats, dtype=np.float32)
    return all_features


def process_row(row):

    feats = extract_features(row)
    url = normalize_url(row['url'])
    domain = tldextract.extract(url).domain or ''
    tld = tldextract.extract(url).suffix or ''
    user = row['by'] or ''

    domain_idx = global_domain_vocab.get(domain, 0)
    tld_idx = global_tld_vocab.get(tld, 0)
    user_idx = global_user_vocab.get(user, 0)

    tokens = tokenize_title(row['title'])
    emb = embed_title(tokens)

    retval =  {
        "features_num": feats,
        "embedding": emb,
        "domain_idx": domain_idx,
        "tld_idx": tld_idx,
        "user_idx": user_idx,
    }
    if 'score' in row:
        retval['target'] = np.clip(row['score'], 0, 500)
    return retval


def normalize_url(url):
    if url is None or not str(url).strip():
        return 'http://empty'
    url = str(url).strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    return url


def tokenize_title(title_text):
    tokens = title_text.lower().split()
    token_indices = [global_w2i.get(token, 0) for token in tokens]
    return token_indices


def embed_title(token_indices):
    if len(token_indices) == 0:
        return torch.zeros(global_embedding_matrix.shape[1])
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    embedded = global_embedding_matrix[token_indices]
    avg_embedding = embedded.mean(dim=0)
    return avg_embedding.numpy()

