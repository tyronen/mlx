import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import torch._dynamo
import torch._inductor.config
from tqdm import tqdm
from common import arguments, utils
from models import image_caption, image_caption_utils

parser = arguments.get_parser(description="Train simple model")
parser.add_argument(
    "--finetune_from",
    type=str,
    default=None,
    help="Path to a pretrained model checkpoint to fine-tune from",
)
args = parser.parse_args()


hyperparameters = {
    "accumulation_steps": 16,
    "batch_size": 256,
    "model_dim": 512,
    "ffn_dim": 1536,
    "num_heads": 8,
    "num_decoders": 4,
    "learning_rate": 1e-4,
    "epochs": int(args.epochs),
    "dropout": 0.3,
    "patience": 3,
    "weight_decay": 1e-3,
    "label_smoothing": 0.1,
    "use_custom_decoder": args.custom,
    "dataset": args.dataset,
    "use_official_captions": args.official_captions,
    "finetune_from": args.finetune_from,
}

sweep_config = {
    "method": "random",  # can be 'grid', 'random', or 'bayes'
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "accumulation_steps": {"values": [8, 16]},
        "batch_size": {"values": [192, 256, 320]},
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
        "use_custom_decoder": {"values": [args.custom]},
        "dataset": {"values": [args.dataset]},
        "use_official_captions": {"values": [args.official_captions]},
        "finetune_from": {"values": [args.finetune_from]},
    },
}


def validate_model(model, validation_dataloader, epoch, device, config, pad_token_id):
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
            loss = loss_fn(batch, model, config["label_smoothing"], pad_token_id)

            # loss already computed by loss_fn
            total_loss += loss.item()
            num_batches += 1

    model.train()  # Set back to training mode
    logging.info(f"validation: total loss {total_loss:.3f} batches {num_batches}")
    return total_loss / num_batches


def loss_fn(batch, model, label_smoothing, pad_token_id):
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
        ignore_index=pad_token_id,
        label_smoothing=label_smoothing,
    )
    return loss


def main():
    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=args.project)
        wandb.agent(sweep_id, function=run_training)
    else:
        config = dict(hyperparameters)  # makes a shallow copy
        run_training(config)


def run_training(config, **_):
    utils.setup_logging()
    device = utils.get_device()

    model_file = (
        image_caption_utils.CUSTOM_FLICKR_MODEL_FILE
        if config["dataset"] == "flickr" and config["use_custom_decoder"]
        else (
            image_caption_utils.CUSTOM_COCO_MODEL_FILE
            if config["dataset"] == "coco" and config["use_custom_decoder"]
            else (
                image_caption_utils.BASE_FLICKR_MODEL_FILE
                if config["dataset"] == "flickr"
                else (
                    image_caption_utils.OFFICIAL_COCO_MODEL_FILE
                    if config.get("use_official_captions")
                    else image_caption_utils.BASE_COCO_MODEL_FILE
                )
            )
        )
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

    if config["dataset"] == "flickr":
        train_dataset = image_caption.Flickr30kDataset(split="train")
        validation_dataset = image_caption.Flickr30kDataset(split="val")
        test_dataset = image_caption.Flickr30kDataset(split="test")
    elif config["dataset"] == "coco":
        train_dataset = image_caption.CocoDataset(
            split="train",
            use_official_captions=config.get("use_official_captions", False),
        )
        validation_dataset = image_caption.CocoDataset(
            split="val",
            use_official_captions=config.get("use_official_captions", False),
        )
        test_dataset = image_caption.CocoDataset(
            split="test",
            use_official_captions=config.get("use_official_captions", False),
        )
    else:
        raise ValueError(
            f"Unknown dataset: {config['dataset']}. Choose 'coco' or 'flickr'."
        )

    logging.info(
        f"Dataset sizes: training {len(train_dataset)} validation: {len(validation_dataset)} test: {len(test_dataset)}"
    )
    training_dataloader = image_caption_utils.CustomDataLoader(
        train_dataset, device, batch_size=config["batch_size"], train=True
    )
    validation_dataloader = image_caption_utils.CustomDataLoader(
        validation_dataset, device, batch_size=config["batch_size"]
    )
    test_dataloader = image_caption_utils.CustomDataLoader(
        test_dataset, device, batch_size=config["batch_size"]
    )

    # Total optimizer steps = (batches per epoch / accumulation steps) × epochs
    steps_per_epoch = len(training_dataloader) // config["accumulation_steps"]
    total_steps = steps_per_epoch * config["epochs"]

    maybe_autocast, scaler = utils.amp_components(device, True)
    model = image_caption.CombinedTransformer(
        model_dim=config["model_dim"],
        ffn_dim=config["ffn_dim"],
        num_heads=config["num_heads"],
        num_decoders=config["num_decoders"],
        dropout=config["dropout"],
        use_custom_decoder=config["use_custom_decoder"],
    ).to(device)

    # If fine-tuning from a checkpoint, load it now
    if config.get("finetune_from"):
        logging.info(f"Loading pretrained model from {config['finetune_from']}...")
        checkpoint = torch.load(config["finetune_from"], map_location=device)
        # Load state dict but allow for missing keys if architecture slightly differs (e.g. compiled vs not)
        # strict=False is safer when moving between compiled/uncompiled or if LoRA keys shift
        try:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logging.info("Successfully loaded pretrained state dict")
        except Exception as e:
            logging.warning(f"Error loading state dict (trying strict=False): {e}")

    wandb.define_metric("val_loss", summary="min")

    # Store pad_token_id before compilation
    pad_token_id = model.tokenizer.pad_token_id

    # Enable TF32 for better performance on RTX 5090
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.allow_tf32 = True  # type: ignore

    # Compile model with CUDA Graphs disabled (CUDA Graphs conflicts with gradient accumulation)
    # Still get kernel fusion and graph optimization from torch.compile()

    torch._dynamo.config.suppress_errors = True
    # Multiple ways to disable CUDA Graphs
    torch._inductor.config.triton.cudagraphs = False  # type: ignore
    torch._inductor.config.triton.cudagraph_trees = False  # type: ignore

    # Use default mode instead of reduce-overhead to avoid CUDA Graphs entirely
    model = torch.compile(model)
    logging.info(
        "Model compiled with torch.compile() (CUDA Graphs disabled for gradient accumulation)"
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0
    grad_norm = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config["epochs"]):

        total_train_loss = 0.0
        num_train_batches = 0
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            with maybe_autocast:
                loss = loss_fn(batch, model, config["label_smoothing"], pad_token_id)
                loss = loss / config["accumulation_steps"]

            scaler.scale(loss).backward()
            if (i + 1) % config["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                total_train_loss += loss.item()
                num_train_batches += 1
                grad_norm = total_norm.item()

        logging.info(f"Epoch {epoch + 1}/{config['epochs']}")
        avg_train_loss = (
            total_train_loss / num_train_batches
            if num_train_batches > 0
            else total_train_loss
        )
        avg_val_loss = validate_model(
            model, validation_dataloader, epoch, device, config, pad_token_id
        )

        run.log(
            {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "grad_norm": grad_norm,
            },
        )
        last_epoch += 1
        logging.info(
            f"train loss {avg_train_loss:.3f} avg val loss {avg_val_loss:.3f} best val loss {best_val_loss:.3f}"
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save underlying model state dict (not compiled wrapper)
            state_dict = (
                model._orig_mod.state_dict()
                if hasattr(model, "_orig_mod")
                else model.state_dict()
            )
            torch.save(
                {
                    "state_dict": state_dict,
                    "model_dim": config["model_dim"],
                    "ffn_dim": config["ffn_dim"],
                    "num_heads": config["num_heads"],
                    "num_decoders": config["num_decoders"],
                    "dropout": config["dropout"],
                },
                model_file,
            )
            logging.info(f"Saved in epoch {last_epoch}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                run.log({"early_stopping_epochs": epoch + 1})
                break
        if args.check:
            break
    checkpoint = torch.load(model_file)
    # Load state dict into the underlying (non-compiled) model
    if hasattr(model, "_orig_mod"):
        model._orig_mod.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    test_loss = validate_model(
        model, test_dataloader, last_epoch + 1, device, config, pad_token_id
    )
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
    run.finish(0)


if __name__ == "__main__":
    main()
