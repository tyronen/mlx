import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import wandb
from tqdm import tqdm
import utils
import models
import subprocess
import bitsandbytes.optim as bnb_optim
from utils import CustomDataLoader

parser = argparse.ArgumentParser(description="Train simple model")
parser.add_argument("--entity", help="W and B entity", default="mlx-institute")
parser.add_argument("--base", help="Whether to use base decoder", action="store_true")
parser.add_argument("--sweep", help="Run a sweep", action="store_true")
parser.add_argument("--check", help="Make sure it works", action="store_true")
args = parser.parse_args()


hyperparameters = {
    "accumulation_steps": 128,
    "batch_size": 32,
    "model_dim": 512,
    "ffn_dim": 1536,
    "num_heads": 8,
    "num_decoders": 4,
    "learning_rate": 5e-4,
    "epochs": 50,
    "dropout": 0.1,
    "patience": 1,
    "label_smoothing": 0.1,
    "use_custom_decoder": not args.base,
}

sweep_config = {
    "method": "random",  # can be 'grid', 'random', or 'bayes'
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "accumulation_steps": {"values": [128, 192]},
        "batch_size": {"values": [16, 32]},
        "model_dim": {"values": [384, 512]},
        "ffn_dim": {"values": [1536, 2048]},
        "num_heads": {"values": [8]},
        "num_decoders": {"values": [4]},
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 5e-5,
            "max": 5e-4,
        },
        "epochs": {"values": [20]},
        "dropout": {"values": [0.0, 0.1, 0.2]},
        "patience": {"values": [3, 5, 10]},
        "label_smoothing": {"values": [0.0, 0.05, 0.1]},
        "use_custom_decoder": {"values": [not args.base]},
    },
}


def validate_model(model, validation_dataloader, epoch, device, config):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Important: no gradients during validation
        for i, batch in tqdm(enumerate(validation_dataloader), desc="Validation"):
            # move only tensor items to the target device
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }

            # Forward pass (shared with training)
            loss = loss_fn(batch, model, config["label_smoothing"])

            # loss already computed by loss_fn
            total_loss += loss.item()
            num_batches += 1

    model.train()  # Set back to training mode
    logging.info(f"validation: total loss {total_loss} batches {num_batches}")
    return total_loss / num_batches


def loss_fn(batch, model, label_smoothing):
    input_ids = batch["input_ids"]
    logits = model(batch["images"], input_ids)

    # Standard causal LM alignment:
    # Shift logits to the left and labels to the right.
    # Prediction for token i+1 is at logits[:, i, :]
    # The ground truth for token i+1 is at input_ids[:, i+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1), 
        ignore_index=model.tokenizer.pad_token_id,
        label_smoothing=label_smoothing,
    )
    return loss


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main():
    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=args.project)
        wandb.agent(sweep_id, function=run_training)
    else:
        config = dict(hyperparameters)  # makes a shallow copy
        config["git_commit"] = get_git_commit()
        run_training(config)


def run_training(config=None, **_):
    utils.setup_logging()
    device = utils.get_device()

    model_file = (
        utils.CUSTOM_MODEL_FILE
        if config["use_custom_decoder"]
        else utils.BASE_MODEL_FILE
    )

    project = "custom-decoder" if config["use_custom_decoder"] else "base-decoder"
    run = wandb.init(
        entity=args.entity,
        project=project,
        # Track hyperparameters and run metadata.
        config=config,
    )

    if config is None:
        config = dict(wandb.config)

    train_dataset = models.Flickr30kDataset(split="train")
    validation_dataset = models.Flickr30kDataset(split="val")
    test_dataset = models.Flickr30kDataset(split="test")
    logging.info(
        f"Dataset sizes: training {len(train_dataset)} validation: {len(validation_dataset)} test: {len(test_dataset)}"
    )
    training_dataloader = CustomDataLoader(
        train_dataset, device, batch_size=config["batch_size"], train=True
    )
    validation_dataloader = CustomDataLoader(
        validation_dataset, device, batch_size=config["batch_size"]
    )
    test_dataloader = CustomDataLoader(
        test_dataset, device, batch_size=config["batch_size"]
    )

    # Total optimizer steps = batches per epoch × epochs
    total_steps = len(training_dataloader) * config["epochs"]

    maybe_autocast, scaler = utils.amp_components(device, True)
    model = models.CombinedTransformer(
        model_dim=config["model_dim"],
        ffn_dim=config["ffn_dim"],
        num_heads=config["num_heads"],
        num_decoders=config["num_decoders"],
        dropout=config["dropout"],
        use_custom_decoder=config["use_custom_decoder"],
    ).to(device)
    wandb.watch(model, log="all", log_freq=100)
    wandb.define_metric("val_loss", summary="min")
    params = [
        {
            "params": model.parameters(),
            "lr": config["learning_rate"],
        },
    ]
    optimizer = bnb_optim.Adam8bit(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=config["learning_rate"]
)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0
    grad_norm = 0
    for epoch in range(config["epochs"]):

        total_train_loss = 0.0
        num_train_batches = 0
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            with maybe_autocast:
                loss = loss_fn(batch, model, config["label_smoothing"])
                loss = loss / config["accumulation_steps"]

            scaler.scale(loss).backward()
            if (i + 1) % config["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                total_train_loss += loss.item()
                num_train_batches += 1
                grad_norm = total_norm.item()
                del loss, batch

        logging.info(f"Epoch {epoch + 1}/{config['epochs']}")
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else total_train_loss
        avg_val_loss = validate_model(
            model, validation_dataloader, epoch, device, config
        )
        scheduler.step()

        run.log(
            {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "grad_norm": grad_norm,
            },
        )
        last_epoch += 1
        logging.info(f"train {avg_train_loss} avg {avg_val_loss} best {best_val_loss}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_dim": config["model_dim"],
                    "ffn_dim": config["ffn_dim"],
                    "num_heads": config["num_heads"],
                    "num_decoders": config["num_decoders"],
                    "dropout": config["dropout"],
                },
                model_file,
            )
            logging.info("Saved in epoch {last_epoch}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                run.log({"early_stopping_epochs": epoch + 1})
                break
        if args.check:
            break
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss = validate_model(model, test_dataloader, last_epoch + 1, device, config)
    run.log(
        {"test_loss": test_loss},
    )
    if not args.check:
        name = (
            "basic-decoder-model"
            if hyperparameters["use_custom_decoder"]
            else "qwen-decoder-model"
        )
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(model_file)
        run.log_artifact(artifact)
    run.finish(0, timeout=0)


if __name__ == "__main__":
    main()
