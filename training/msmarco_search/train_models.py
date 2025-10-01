import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import wandb
from tqdm import tqdm
from common import utils
from torch.utils.data import DataLoader
import numpy as np
from dataset import DATASET_FILE
from models import msmarco_search
from tokenizer import Word2VecTokenizer

# ------------- additional imports -------------
import random

hyperparameters = {
    "embed_dim": 300,
    "margin": 0.3,
    "query_learning_rate": 2e-5,
    "doc_learning_rate": 1e-5,
    "batch_size": 256,
    "epochs": 50,
    "dropout_rate": 0.1,
    "patience": 5,
    "temperature": 0.05,
}


class TripletDataLoader(DataLoader):
    def __init__(self, dataset, device):
        num_workers = 8 if device.type == "cuda" else 0 if device.type == "mps" else 4
        super().__init__(
            dataset,
            batch_size=hyperparameters["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )


# -------------------- HARD NEGATIVE MINING UTILITY ---------------------
def mine_hard_negatives(
    dataset,
    query_tower,
    doc_tower,
    tokenizer,
    device,
    pool_size=10000,
    top_k=1,
    batch_enc=512,
):
    """
    For every query, pick the most similar *non‑positive* passage from a
    sampled pool of all documents (excluding positives for that query).
    """
    # ---- resolve triplet source --------------------------------------------
    triplets = dataset.triplets if hasattr(dataset, "triplets") else dataset

    # ---- build candidate pool from ALL documents (not just positives) ------
    all_doc_texts = []
    for t in triplets:
        all_doc_texts.append(t["positive_text"])
        all_doc_texts.append(t["negative_text"])

    # Remove duplicates
    all_doc_texts = list(set(all_doc_texts))
    if len(all_doc_texts) > pool_size:
        all_doc_texts = random.sample(all_doc_texts, pool_size)

    # encode pool once
    doc_tower.eval()
    cand_embs = []
    with torch.no_grad():
        for i in range(0, len(all_doc_texts), batch_enc):
            toks = tokenizer(all_doc_texts[i : i + batch_enc])["input_ids"].to(device)
            cand_embs.append(doc_tower(toks))
    cand_embs = torch.cat(cand_embs, dim=0)  # [N,D]
    cand_embs = F.normalize(cand_embs, p=2, dim=1)  # unit vectors

    # ---- mine hardest for each query ----------------------------------------
    query_tower.eval()
    mapping = {}
    with torch.no_grad():
        for i in range(0, len(triplets), batch_enc):
            sub = triplets[i : i + batch_enc]
            q_texts = [t["query_text"] for t in sub]
            pos_texts = [t["positive_text"] for t in sub]

            toks = tokenizer(q_texts)["input_ids"].to(device)
            q_embs = F.normalize(query_tower(toks), p=2, dim=1)  # [b,D]

            sims = torch.matmul(q_embs, cand_embs.T)  # [b,N]

            # For each query, find the hardest negative (highest similarity)
            # that is NOT the positive for that query
            for j, (q_text, pos_text) in enumerate(zip(q_texts, pos_texts)):
                # Get similarities for this query
                q_sims = sims[j]  # [N]

                # Find indices of documents that are NOT the positive
                non_pos_mask = torch.tensor(
                    [all_doc_texts[k] != pos_text for k in range(len(all_doc_texts))],
                    device=device,
                )

                if non_pos_mask.any():
                    # Mask out positive documents and find hardest negative
                    masked_sims = q_sims.masked_fill(~non_pos_mask, float("-inf"))
                    top_idx = masked_sims.argmax()
                    mapping[q_text] = all_doc_texts[top_idx.item()]
                else:
                    # Fallback: use random negative if no non-positive docs
                    mapping[q_text] = random.choice(
                        [t for t in all_doc_texts if t != pos_text]
                    )

    # ---- apply --------------------------------------------------------------
    dataset.replace_negatives(mapping)
    query_tower.train()
    doc_tower.train()


def analyze_embedding_diversity(run, embeddings, name, epoch):
    """Check if embeddings are collapsing to similar values"""
    # Calculate pairwise similarities between embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    similarities = torch.mm(embeddings_norm, embeddings_norm.t())

    # Remove diagonal (self-similarities)
    similarities.fill_diagonal_(0)

    mean_sim = similarities.mean().item()
    std_sim = similarities.std().item()

    run.log(
        {
            f"{name.lower().replace(' ', '_')}_mean_inter_embedding_similarity": mean_sim,
            f"{name.lower().replace(' ', '_')}_std_inter_embedding_similarity": std_sim,
        },
        step=epoch,
    )
    return mean_sim, std_sim


def validate_model(run, query_tower, doc_tower, validation_dataloader, epoch, device):
    query_tower.eval()
    doc_tower.eval()

    total_loss = 0.0
    num_batches = 0

    all_margins = []
    pos_similarities = []
    neg_similarities = []

    all_queries = []
    all_positives = []
    all_negatives = []

    correct_at_k = 0
    total_queries = 0

    with torch.no_grad():  # Important: no gradients during validation
        for i, batch in tqdm(enumerate(validation_dataloader), desc="Validation"):
            # move only tensor items to the target device
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }

            # Forward pass (shared with training)
            (
                loss,
                logits,
                labels,
                q,
                pos,
                neg,
            ) = _contrastive_forward(
                batch, query_tower, doc_tower, hyperparameters["temperature"]
            )

            if i == 0:  # Only for first batch
                analyze_embedding_diversity(run, q, "query", epoch)
                analyze_embedding_diversity(run, pos, "positive doc", epoch)
                analyze_embedding_diversity(run, neg, "negative doc", epoch)

                # Check if embeddings are too similar to each other
                q_mean = q.mean(dim=0)
                pos_mean = pos.mean(dim=0)
                neg_mean = neg.mean(dim=0)

                # Check variance across dimensions
                q_var = q.var(dim=0).mean()
                pos_var = pos.var(dim=0).mean()
                neg_var = neg.var(dim=0).mean()

                all_queries.extend(q.cpu())
                all_positives.extend(pos.cpu())
                all_negatives.extend(neg.cpu())

                run.log(
                    {
                        "embedding_means_query_norm": q_mean.norm().item(),
                        "embedding_means_pos_norm": pos_mean.norm().item(),
                        "embedding_means_neg_norm": neg_mean.norm().item(),
                        "embedding_variance_query": q_var.item(),
                        "embedding_variance_pos": pos_var.item(),
                        "embedding_variance_neg": neg_var.item(),
                    },
                    step=epoch,
                )

            dst_pos = F.cosine_similarity(q, pos)
            dst_neg = F.cosine_similarity(q, neg)

            pos_similarities.extend(dst_pos.cpu().numpy())
            neg_similarities.extend(dst_neg.cpu().numpy())
            margins = (dst_pos - dst_neg).cpu().numpy()
            all_margins.extend(margins)

            # Check if positive document ranks higher than negative
            correct_at_k += (dst_pos > dst_neg).sum().item()
            total_queries += dst_pos.size(0)

            # loss already computed by _contrastive_forward
            total_loss += loss.item()
            num_batches += 1

    # Calculate retrieval metrics
    mrr_scores = []
    for q, pos, neg in zip(all_queries, all_positives, all_negatives):
        pos_sim = F.cosine_similarity(q.unsqueeze(0), pos.unsqueeze(0))
        neg_sim = F.cosine_similarity(q.unsqueeze(0), neg.unsqueeze(0))

        if pos_sim > neg_sim:
            mrr_scores.append(1.0)  # Positive ranked first
        else:
            mrr_scores.append(0.0)  # Positive not ranked first

    run.log(
        {
            "validation_pos_similarities_mean": np.mean(pos_similarities),
            "validation_pos_similarities_std": np.std(pos_similarities),
            "validation_neg_similarities_mean": np.mean(neg_similarities),
            "validation_neg_similarities_std": np.std(neg_similarities),
            "validation_margins_mean": np.mean(all_margins),
            "validation_margins_std": np.std(all_margins),
            "validation_positive_margins_count": (np.array(all_margins) > 0).sum(),
            "validation_positive_margins_ratio": (np.array(all_margins) > 0).mean()
            * 100,
            "mean_mrr": np.mean(mrr_scores),
        },
        step=epoch,
    )
    recall_at_k = correct_at_k / total_queries if total_queries > 0 else 0.0

    query_tower.train()  # Set back to training mode
    doc_tower.train()

    return total_loss / num_batches, recall_at_k


# ------------------------------------------------------------------ #
# Shared forward pass + in‑batch contrastive loss
# ------------------------------------------------------------------ #
def _contrastive_forward(batch, query_tower, doc_tower, temperature):
    """
    Returns:
        loss     – scalar
        logits   – [B, 2B] similarity scores (cosine / temperature)
        labels   – ground‑truth class indices 0 … B‑1
        q, pos, neg – the three embedding blocks (for diagnostics)
    """
    q = query_tower(batch["query"])  # [B,D]
    pos = doc_tower(batch["positive"])  # [B,D]
    neg = doc_tower(batch["negative"])  # [B,D]

    docs = torch.cat([pos, neg], dim=0)  # [2B,D]
    logits = torch.matmul(q, docs.T) / temperature  # [B,2B]
    labels = torch.arange(q.size(0), device=q.device)

    loss = F.cross_entropy(logits, labels)
    return loss, logits, labels, q, pos, neg


def training_loop_core(batch, query_tower, doc_tower, all_train_margins):
    """
    Multi‑negative contrastive loss that **explicitly includes the mined
    hard negative** for each query in addition to the in‑batch negatives.

    For a batch of size B:
      • docs[:B]   == positives aligned with queries
      • docs[B:]   == per‑query hard negatives
    The logits matrix is [B  ×  2B].
    The correct class for query *i* is column *i*.
    """
    loss, logits, labels, q, pos, neg = _contrastive_forward(
        batch, query_tower, doc_tower, hyperparameters["temperature"]
    )

    # --- monitor margin (pos − best other) -----------------------------
    with torch.no_grad():
        pos_scores = logits[torch.arange(q.size(0)), labels]  # diag
        mask = F.one_hot(labels, num_classes=logits.size(1)).bool()
        hardest = logits.masked_fill(mask, float("-inf")).max(dim=1).values
        all_train_margins.extend((pos_scores - hardest).cpu().tolist())

        preds = logits.argmax(dim=1)
        num_correct = (preds == labels).sum().item()

    return loss, num_correct


def main():
    utils.setup_logging()
    device = utils.get_device()

    tokenizer = Word2VecTokenizer()
    vocab_size = tokenizer.vocab_size

    query_tower = msmarco_search.QueryTower(
        tokenizer.embeddings,
        hyperparameters["embed_dim"],
        hyperparameters["dropout_rate"],
    ).to(device)
    doc_tower = msmarco_search.DocTower(
        tokenizer.embeddings,
        hyperparameters["embed_dim"],
        hyperparameters["dropout_rate"],
    ).to(device)

    config = {
        **hyperparameters,
        "vocab_size": vocab_size,
        "query_trained": query_tower.embedding.weight.requires_grad,
        "doc_trained": doc_tower.embedding.weight.requires_grad,
    }
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="mlx-institute",
        # Set the wandb project where this run will be logged.
        project="TwoTowers",
        # Track hyperparameters and run metadata.
        config=config,
    )

    datasets = torch.load(DATASET_FILE, weights_only=False)
    train_dataset = datasets["train"]
    validation_dataset = datasets["validation"]
    test_dataset = datasets["test"]
    logging.info(
        f"Dataset sizes: training {len(train_dataset)} validation: {len(validation_dataset)} test: {len(test_dataset)}"
    )
    validation_dataloader = TripletDataLoader(validation_dataset, device)
    test_dataloader = TripletDataLoader(test_dataset, device)

    scaler = GradScaler()

    params = [
        {
            "params": query_tower.parameters(),
            "lr": hyperparameters["query_learning_rate"],
        },
        {"params": doc_tower.parameters(), "lr": hyperparameters["doc_learning_rate"]},
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0
    all_params = list(query_tower.parameters()) + list(doc_tower.parameters())
    grad_norm = 0
    for epoch in range(hyperparameters["epochs"]):
        # refresh dataloader (dataset may have grown new negatives last epoch)
        training_dataloader = TripletDataLoader(train_dataset, device)

        # mine hard negatives every 5 epochs (skip epoch 0)
        if epoch > 0 and epoch % 5 == 0:
            logging.info("🔍 Mining hard negatives with current model...")
            mine_hard_negatives(
                train_dataset, query_tower, doc_tower, tokenizer, device
            )

        total_train_loss = 0.0
        total_correct = 0
        num_train_batches = 0
        all_train_margins = []
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            if device.type == "mps":
                loss, correct = training_loop_core(
                    batch, query_tower, doc_tower, all_train_margins
                )
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
            else:
                with autocast():
                    loss, correct = training_loop_core(
                        batch, query_tower, doc_tower, all_train_margins
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total_train_loss += loss.item()
            num_train_batches += 1
            total_correct += correct
            grad_norm = total_norm.item()

        logging.info(f"Epoch {epoch + 1}/{hyperparameters['epochs']}")
        margins_tensor = torch.tensor(all_train_margins)
        avg_train_loss = total_train_loss / num_train_batches
        train_top1_acc = total_correct / (
            num_train_batches * hyperparameters["batch_size"]
        )
        avg_val_loss, retrieval_acc = validate_model(
            run, query_tower, doc_tower, validation_dataloader, epoch, device
        )
        scheduler.step(avg_val_loss)

        run.log(
            {
                "query_learning_rate": optimizer.param_groups[0]["lr"],
                "doc_learning_rate": optimizer.param_groups[1]["lr"],
                "train_margins_avg": margins_tensor.mean().item(),
                "train_margins_min": margins_tensor.min().item(),
                "train_margins_max": margins_tensor.max().item(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_retrieval_accuracy": retrieval_acc,
                "grad_norm": grad_norm,
                "train_top1_acc": train_top1_acc,
            },
            step=epoch,
        )
        last_epoch += 1
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "query_tower": query_tower.state_dict(),
                    "doc_tower": doc_tower.state_dict(),
                    "parameters": params,
                    "vocab_size": vocab_size,
                    "embed_dim": hyperparameters["embed_dim"],
                    "dropout_rate": hyperparameters["dropout_rate"],
                },
                msmarco_search.MODEL_FILE,
            )
        else:
            patience_counter += 1
            if patience_counter >= hyperparameters["patience"]:
                run.log({"early_stopping_epochs": epoch + 1})
                break
    checkpoint = torch.load(msmarco_search.MODEL_FILE)
    query_tower.load_state_dict(checkpoint["query_tower"])
    doc_tower.load_state_dict(checkpoint["doc_tower"])
    test_loss, retrieval_acc = validate_model(
        run,
        query_tower,
        doc_tower,
        test_dataloader,
        last_epoch + 1,
        device,
    )
    run.log(
        {"test_loss": test_loss, "test_retrieval_accuracy": retrieval_acc},
        step=last_epoch + 1,
    )
    artifact = wandb.Artifact(name="two_tower_model", type="model")
    artifact.add_file(msmarco_search.MODEL_FILE)
    run.log_artifact(artifact)
    run.finish(0)


if __name__ == "__main__":
    main()
