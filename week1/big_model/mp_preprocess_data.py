import os
import numpy as np
import pandas as pd
import torch

import utils
from tqdm import tqdm
import multiprocessing as mp
import json
from collections import Counter
import tldextract

UNK_TOKEN = '<unk>'


def init_worker(embedding_matrix, w2i_dict, Tmin, Tmax, domain_vocab, tld_vocab, user_vocab):
    utils.global_embedding_matrix = torch.as_tensor(embedding_matrix).clone().detach()
    utils.global_w2i = w2i_dict
    utils.global_Tmin = Tmin
    utils.global_Tmax = Tmax
    utils.global_domain_vocab = domain_vocab
    utils.global_tld_vocab = tld_vocab
    utils.global_user_vocab = user_vocab



def build_vocab(values, min_freq=1, topk=None):
    counter = Counter(values)
    items = [(v, count) for v, count in counter.items() if count >= min_freq]
    if topk is not None:
        items = sorted(items, key=lambda x: -x[1])[:topk]
    
    vocab = {UNK_TOKEN: 0}  # reserve index 0 for unknown token
    for idx, (v, count) in enumerate(items, start=1):
        vocab[v] = idx
    return vocab


def precompute_parallel(df, embedding_matrix, w2i_dict, domain_vocab, tld_vocab, user_vocab,
                         Tmin=None, Tmax=None, ref_time=None, compute_delta_t=False, num_workers=None):
    df = df.reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])

    if compute_delta_t:
        delta_t = (ref_time - df['time']) / np.timedelta64(30, 'D')
        delta_t = delta_t.to_numpy().astype(np.float32)
    else:
        delta_t = None

    with mp.Pool(processes=num_workers, initializer=init_worker,
                 initargs=(embedding_matrix, w2i_dict, Tmin, Tmax, domain_vocab, tld_vocab, user_vocab)) as pool:
        results = list(tqdm(pool.imap(utils.process_row, df.to_dict(orient='records')),
                            total=len(df), desc="Precomputing"))

    all_features_num = np.stack([r['features_num'] for r in results]).astype(np.float32)
    all_embeddings = np.stack([r['embedding'] for r in results]).astype(np.float32)
    all_targets = np.array([r['target'] for r in results], dtype=np.float32)
    all_domain_idx = np.array([r['domain_idx'] for r in results], dtype=np.int32)
    all_tld_idx = np.array([r['tld_idx'] for r in results], dtype=np.int32)
    all_user_idx = np.array([r['user_idx'] for r in results], dtype=np.int32)

    return all_features_num, all_embeddings, all_domain_idx, all_tld_idx, all_user_idx, all_targets, delta_t

    
if __name__ == '__main__':
    EMBEDDING_FILE = "skipgram_models/silvery200.pt"
    FILEPATH = "data/posts.parquet"
    OUTPUT_DIR = "data"
    NUM_WORKERS = 20

    w2i, embedding_matrix = utils.load_embeddings(EMBEDDING_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(FILEPATH)
    df = df.drop(["id"], axis=1)
    df = df.sort_values(by="time").reset_index(drop=True)
    df = df.dropna()

    train_size = int(0.9 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    T_ref = train_df['time'].max()
    Tmin = train_df['time'].min()
    Tmax = train_df['time'].max()

    domains = [tldextract.extract(utils.normalize_url(url)).domain or '' for url in train_df['url']]
    tlds = [tldextract.extract(utils.normalize_url(url)).suffix or '' for url in train_df['url']]
    users = train_df['by'].fillna('')

    domain_vocab = build_vocab(domains)
    tld_vocab = build_vocab(tlds, topk=5)
    user_vocab = build_vocab(users)

    # save vocab with unknown token
    all_vocabs = {
        "domain_vocab": domain_vocab,
        "tld_vocab": tld_vocab,
        "user_vocab": user_vocab
    }

    with open(utils.TRAINING_VOCAB_PATH, "w") as f:
        json.dump(all_vocabs, f)

    train_features_num, train_embeddings, train_domain_idx, train_tld_idx, train_user_idx, train_targets, train_delta_t = precompute_parallel(
        train_df, embedding_matrix, w2i, domain_vocab, tld_vocab, user_vocab,
        Tmin=Tmin, Tmax=Tmax, ref_time=T_ref, compute_delta_t=True, num_workers=NUM_WORKERS)

    np.savez(os.path.join(OUTPUT_DIR, 'train.npz'),
            features_num=train_features_num,
             title_embeddings=train_embeddings,
             domain_index=train_domain_idx,
             tld_index=train_tld_idx,
             user_index=train_user_idx,
             delta_t=train_delta_t,
             targets=train_targets)

    val_features_num, val_embeddings, val_domain_idx, val_tld_idx, val_user_idx, val_targets, val_delta_t = precompute_parallel(
        val_df, embedding_matrix, w2i, domain_vocab, tld_vocab, user_vocab,
        Tmin=Tmin, Tmax=Tmax, compute_delta_t=False, num_workers=NUM_WORKERS)

    np.savez(os.path.join(OUTPUT_DIR, 'val.npz'),
            features_num=val_features_num,
             title_embeddings=val_embeddings,
             domain_index=val_domain_idx,
             tld_index=val_tld_idx,
             user_index=val_user_idx,
             delta_t=val_delta_t,
             targets=val_targets)

    print("Precomputation finished!")
