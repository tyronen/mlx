import logging
import os
import time
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
parser.add_argument(
    "--accumulation_steps",
    type=int,
    default=8,
    help="Gradient accumulation steps (controls effective batch size)",
)
parser.add_argument(
    "--max_batch_size",
    type=int,
    default=512,
    help="Max batch size",
)
parser.add_argument(
    "--profile_timings",
    action="store_true",
    help="Profile timing of the model",
)
parser.add_argument(
    "--use_mlp_projector",
    action="store_true",
    help="Use MLP for image projection instead of Linear",
)
args = parser.parse_args()


hyperparameters = {
    "accumulation_steps": int(args.accumulation_steps),
    "batch_size": int(args.max_batch_size),
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
    "dataset": args.dataset,
    "use_official_captions": args.official_captions,
    "finetune_from": args.finetune_from,
    "max_vision_tokens": args.max_vision_tokens,
    "use_mlp_projector": args.use_mlp_projector,
}

sweep_config = {
    "method": "random",  # can be 'grid', 'random', or 'bayes'
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "accumulation_steps": {"values": [int(args.accumulation_steps)]},
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
        "dataset": {"values": [args.dataset]},
        "use_official_captions": {"values": [args.official_captions]},
        "finetune_from": {"values": [args.finetune_from]},
        "max_vision_tokens": {"values": [args.max_vision_tokens]},
        "use_mlp_projector": {"values": [args.use_mlp_projector]},
    },
}


def is_cuda_oom_error(error: Exception) -> bool:
    """Best-effort check for CUDA OOM without pattern matching every backend."""
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def prepare_batch(raw_batch, device, vision_encoder, use_precomputed: bool):
    """
    Move a dataloader batch to the target device, run the vision encoder,
    and expand grouped captions so the decoder sees one feature per caption.
    """
    batch = {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in raw_batch.items()
    }
    if use_precomputed:
        images = batch["images"].to(device, dtype=torch.bfloat16)
    else:
        if vision_encoder is None:
            raise RuntimeError(
                "Vision encoder is None but precomputed features disabled."
            )
        with torch.no_grad():
            images = vision_encoder(batch["images"])
    if "num_captions_per_image" in batch:
        repeats = torch.tensor(
            batch["num_captions_per_image"],
            device=images.device,
        )
        images = images.repeat_interleave(repeats, dim=0)
    return {"images": images, "input_ids": batch["input_ids"]}


def resolve_chunk_size(total_captions: int, micro_batch_size: int) -> int:
    if micro_batch_size is None or micro_batch_size <= 0:
        if total_captions <= 1:
            return total_captions
        # Largest power of two strictly less than total_captions
        highest_pow2 = 1 << (total_captions.bit_length() - 1)
        if highest_pow2 >= total_captions:
            highest_pow2 //= 2
        return max(1, highest_pow2)
    return min(micro_batch_size, total_captions)


def default_precomputed_metadata_path(
    dataset: str, use_official_captions: bool, max_tokens: int
) -> str:
    meta_path, _ = image_caption_utils.feature_paths(
        dataset, use_official_captions if dataset == "coco" else False, max_tokens
    )
    return meta_path


def backward_micro_batches(
    batch,
    micro_batch_size,
    model,
    pad_token_id,
    label_smoothing,
    maybe_autocast,
    scaler,
    accumulation_steps,
):
    input_ids = batch["input_ids"]
    images = batch["images"]
    total = input_ids.size(0)
    chunk_size = resolve_chunk_size(total, micro_batch_size)

    # Pre-calculate fractions
    if chunk_size >= total:
        chunks = [(0, total, 1.0)]
    else:
        chunks = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunks.append((start, end, (end - start) / total))

    batch_loss_value = 0.0

    # Manual loop to avoid generator overhead
    for start, end, fraction in chunks:
        # Slice directly
        chunk_images = images[start:end]
        chunk_input_ids = input_ids[start:end]

        with maybe_autocast:
            # Re-implement loss_fn inline to avoid function call overhead
            logits = model(chunk_images, chunk_input_ids)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_input_ids[:, 1:].contiguous()

            vocab_size = shift_logits.size(-1)
            chunk_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=pad_token_id,
                label_smoothing=label_smoothing,
            )
            chunk_loss = chunk_loss * fraction

        scaled_loss = chunk_loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        batch_loss_value += chunk_loss.detach().item()

    return batch_loss_value


def evaluate_micro_batches(
    batch, micro_batch_size, model, label_smoothing, pad_token_id
):
    chunk_size = resolve_chunk_size(batch["input_ids"].size(0), micro_batch_size)
    total_loss = 0.0
    for chunk, fraction in iter_micro_batches(batch, chunk_size):
        loss = loss_fn(chunk, model, label_smoothing, pad_token_id)
        total_loss += loss.item() * fraction
    return total_loss


def autotune_batching(
    config,
    build_dataloaders,
    device,
    vision_encoder,
    model,
    pad_token_id,
    maybe_autocast,
    scaler,
    min_batch_size,
    use_precomputed: bool,
):
    logging.info(
        "Auto-tuning batch size and micro batch size to avoid CUDA OOM for the selected max vision tokens."
    )

    training_dataloader, _, _ = build_dataloaders()

    while True:
        data_iter = iter(training_dataloader)
        try:
            raw_batch = next(data_iter)
        except StopIteration:
            raise RuntimeError(
                "Training dataloader yielded no samples; unable to tune batch size."
            )

        try:
            prepared = prepare_batch(raw_batch, device, vision_encoder, use_precomputed)
        except RuntimeError as err:
            if not is_cuda_oom_error(err):
                raise
            if config["batch_size"] <= min_batch_size:
                raise RuntimeError(
                    "CUDA OOM while encoding images even at the minimum batch size."
                ) from err
            config["batch_size"] = max(min_batch_size, config["batch_size"] // 2)
            logging.warning(
                f"OOM during warmup image encoding. Reducing training batch size to {config['batch_size']}."
            )
            torch.cuda.empty_cache()
            training_dataloader, _, _ = build_dataloaders()
            continue

        try:
            backward_micro_batches(
                prepared,
                config.get("micro_batch_size", 0),
                model,
                pad_token_id,
                config["label_smoothing"],
                maybe_autocast,
                scaler,
                config["accumulation_steps"],
            )
            model.zero_grad(set_to_none=True)
            logging.info(
                "Auto-tune settled on batch_size=%s and micro_batch_size=%s captions.",
                config["batch_size"],
                config.get("micro_batch_size", 0) or "full",
            )
            break
        except RuntimeError as err:
            model.zero_grad(set_to_none=True)
            if not is_cuda_oom_error(err):
                raise
            torch.cuda.empty_cache()
            total_captions = prepared["input_ids"].size(0)
            chunk_size = resolve_chunk_size(
                total_captions, config.get("micro_batch_size", 0)
            )

            # If chunk_size is currently "full batch" (or close to it), find the largest power of 2 below it.
            if chunk_size >= total_captions:
                # Start search at next lower power of 2 from total_captions
                new_micro = 1 << (total_captions.bit_length() - 1)
                if (
                    new_micro >= total_captions
                ):  # handle exact power of 2 case or slight overshot
                    new_micro //= 2
            else:
                # Otherwise just halve it, keeping it a power of 2 if it started as one
                new_micro = max(1, chunk_size // 2)

            if new_micro < 1:
                # Fallback logic for extremely small batches
                if config["batch_size"] <= min_batch_size:
                    raise RuntimeError(
                        "Unable to fit even with micro_batch_size=1 caption."
                    ) from err
                config["batch_size"] = max(min_batch_size, config["batch_size"] // 2)
                logging.warning(
                    f"OOM persists with micro batch size {chunk_size}. Reducing training batch size to {config['batch_size']}."
                )
                training_dataloader, _, _ = build_dataloaders()
            else:
                config["micro_batch_size"] = new_micro
                logging.warning(
                    f"OOM during warmup. Reducing micro batch size to {new_micro} captions (power-of-2 preferred)."
                )
            continue

    # Rebuild loaders so the actual training epoch starts from the first batch.
    return build_dataloaders()


def iter_micro_batches(batch, chunk_size: int):
    """
    Yield smaller (images, input_ids) batches so we can trade compute for memory.

    Args:
        batch: dictionary containing tensors keyed by 'images' and 'input_ids'.
        chunk_size: desired captions per micro batch. <=0 means no chunking.
    """
    images = batch["images"]
    input_ids = batch["input_ids"]
    total = input_ids.size(0)
    if total == 0:
        return
    if chunk_size is None or chunk_size <= 0 or chunk_size >= total:
        chunk_size = total
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        fraction = (end - start) / total
        yield {
            "images": images[start:end],
            "input_ids": input_ids[start:end],
        }, fraction


def validate_model(
    model,
    vision_encoder,
    validation_dataloader,
    device,
    config,
    pad_token_id,
    maybe_autocast,
    use_precomputed: bool,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        with maybe_autocast:
            # Important: no gradients during validation
            for i, batch in tqdm(enumerate(validation_dataloader), desc="Validation"):
                prepared = prepare_batch(batch, device, vision_encoder, use_precomputed)
                batch_loss = evaluate_micro_batches(
                    prepared,
                    config.get("micro_batch_size", 0),
                    model,
                    config["label_smoothing"],
                    pad_token_id,
                )
                total_loss += batch_loss
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


def maybe_synchronize():
    if args.profile_timings:
        torch.cuda.synchronize()


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
    if device.type != "cuda" or not torch.cuda.is_available():
        logging.warning("Running on non-CUDA device. This is not supported.")
        return

    meta_candidate = default_precomputed_metadata_path(
        config["dataset"],
        config.get("use_official_captions", False),
        config["max_vision_tokens"],
    )
    if not os.path.exists(meta_candidate):
        raise FileNotFoundError(
            f"Precomputed feature metadata not found at {meta_candidate}. "
            "Run `python -m training.image_caption.precompute_images` with matching "
            f"--dataset/--official_captions/--max_vision_tokens before starting training."
        )
    logging.info("Using precomputed vision features at %s", meta_candidate)
    precomputed_store = image_caption_utils.PrecomputedFeatureStore(meta_candidate)
    if (
        precomputed_store.dataset is not None
        and config["dataset"] != precomputed_store.dataset
    ):
        logging.warning(
            "Precomputed features were generated for dataset '%s' but training is using '%s'. "
            "Proceeding, but this may lead to missing images.",
            precomputed_store.dataset,
            config["dataset"],
        )
    if (
        precomputed_store.use_official_captions is not None
        and config.get("use_official_captions", False)
        != precomputed_store.use_official_captions
    ):
        logging.warning(
            "Precomputed features were generated with use_official_captions=%s "
            "but training requested %s. Ensure these match.",
            precomputed_store.use_official_captions,
            config.get("use_official_captions", False),
        )
    if config["max_vision_tokens"] != precomputed_store.max_vision_tokens:
        logging.info(
            "Overriding max_vision_tokens (%s) with value from precomputed features (%s).",
            config["max_vision_tokens"],
            precomputed_store.max_vision_tokens,
        )
        config["max_vision_tokens"] = precomputed_store.max_vision_tokens

    using_precomputed = True

    model_file = (
        image_caption_utils.BASE_FLICKR_MODEL_FILE
        if config["dataset"] == "flickr"
        else (
            image_caption_utils.OFFICIAL_COCO_MODEL_FILE
            if config.get("use_official_captions")
            else image_caption_utils.BASE_COCO_MODEL_FILE
        )
    )

    project = "base-decoder"
    run = wandb.init(
        entity=args.entity,
        project=project,
        # Track hyperparameters and run metadata.
        config=config,
    )

    if config is None:
        config = dict(wandb.config)

    use_official = False
    if config["dataset"] == "flickr":
        train_dataset = image_caption.Flickr30kDataset(
            split="train", precomputed_store=precomputed_store
        )
        validation_dataset = image_caption.Flickr30kDataset(
            split="val", precomputed_store=precomputed_store
        )
        test_dataset = image_caption.Flickr30kDataset(
            split="test", precomputed_store=precomputed_store
        )
    elif config["dataset"] == "coco":
        use_official = config.get("use_official_captions", False)

        train_dataset = image_caption.CocoDataset(
            split="train",
            use_official_captions=use_official,
            precomputed_store=precomputed_store,
        )
        validation_dataset = image_caption.CocoDataset(
            split="val",
            use_official_captions=use_official,
            precomputed_store=precomputed_store,
        )
        test_dataset = image_caption.CocoDataset(
            split="test",
            use_official_captions=use_official,
            precomputed_store=precomputed_store,
        )
    else:
        raise ValueError(
            f"Unknown dataset: {config['dataset']}. Choose 'coco' or 'flickr'."
        )

    logging.info(
        f"Dataset sizes: training {len(train_dataset)} validation: {len(validation_dataset)} test: {len(test_dataset)}"
    )

    logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logging.info(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    maybe_autocast, scaler = utils.amp_components(device, True)

    # Precomputed features are mandatory; no online encoding path
    vision_encoder = None

    model = image_caption.CombinedTransformer(
        model_dim=config["model_dim"],
        ffn_dim=config["ffn_dim"],
        num_heads=config["num_heads"],
        num_decoders=config["num_decoders"],
        dropout=config["dropout"],
        use_mlp_projector=config.get("use_mlp_projector", False),
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

    def build_dataloaders():
        training_loader = image_caption_utils.CustomDataLoader(
            train_dataset, device, batch_size=config["batch_size"], train=True
        )
        validation_loader = image_caption_utils.CustomDataLoader(
            validation_dataset, device, batch_size=2 * config["batch_size"]
        )
        test_loader = image_caption_utils.CustomDataLoader(
            test_dataset, device, batch_size=2 * config["batch_size"]
        )
        return training_loader, validation_loader, test_loader

    min_batch_size = 4 if (config["dataset"] == "coco" and use_official) else 2
    (
        training_dataloader,
        validation_dataloader,
        test_dataloader,
    ) = autotune_batching(
        config,
        build_dataloaders,
        device,
        vision_encoder,
        model,
        pad_token_id,
        maybe_autocast,
        scaler,
        min_batch_size,
        using_precomputed,
    )

    micro_batch_size = config.get("micro_batch_size", 0)
    if micro_batch_size and micro_batch_size > 0:
        logging.info(
            f"Processing captions in micro batches of {micro_batch_size} to reduce peak memory."
        )

    if run:
        run.config.update(
            {
                "batch_size": config["batch_size"],
                "micro_batch_size": config.get("micro_batch_size", 0),
            },
            allow_val_change=True,
        )

    # Total optimizer steps = (batches per epoch / accumulation steps) × epochs
    steps_per_epoch = max(1, len(training_dataloader) // config["accumulation_steps"])
    total_steps = steps_per_epoch * config["epochs"]

    # Compile model with CUDA Graphs disabled (CUDA Graphs conflicts with gradient accumulation)
    # Still get kernel fusion and graph optimization from torch.compile()

    torch._dynamo.config.suppress_errors = True
    # Multiple ways to disable CUDA Graphs
    torch._inductor.config.triton.cudagraphs = False  # type: ignore
    torch._inductor.config.triton.cudagraph_trees = False  # type: ignore

    # Actually compile the model
    logging.info(
        "Compiling model with torch.compile(mode='max-autotune-no-cudagraphs')..."
    )
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

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
        epoch_start = time.perf_counter()
        total_train_loss = 0.0
        num_train_batches = 0
        accumulation_loss = 0.0
        data_time = 0.0
        prepare_time = 0.0
        compute_time = 0.0
        step_time = 0.0
        captions_processed = 0
        progress = tqdm(total=len(training_dataloader), desc=f"Epoch {epoch + 1}")
        data_iter = iter(training_dataloader)
        for i in range(len(training_dataloader)):
            step_wall_start = time.perf_counter()
            fetch_start = step_wall_start
            try:
                raw_batch = next(data_iter)
            except StopIteration:
                break
            fetch_end = time.perf_counter()
            data_time += fetch_end - fetch_start
            maybe_synchronize()
            prepare_start = time.perf_counter()
            prepared_batch = prepare_batch(
                raw_batch, device, vision_encoder, using_precomputed
            )
            maybe_synchronize()
            prepare_end = time.perf_counter()
            prepare_time += prepare_end - prepare_start
            maybe_synchronize()
            compute_start = time.perf_counter()
            batch_loss_value = backward_micro_batches(
                prepared_batch,
                config.get("micro_batch_size", 0),
                model,
                pad_token_id,
                config["label_smoothing"],
                maybe_autocast,
                scaler,
                config["accumulation_steps"],
            )
            maybe_synchronize()
            compute_end = time.perf_counter()
            compute_time += compute_end - compute_start
            if args.profile_timings:
                # Log batch-level timing every 5 steps
                if i % 5 == 1:
                    load_dur = fetch_end - fetch_start
                    prep_dur = prepare_end - prepare_start
                    comp_dur = compute_end - compute_start
                    logging.info(
                        "Batch %d/%d (s): load %.4f | prep %.4f | comp %.4f",
                        i,
                        len(training_dataloader),
                        load_dur,
                        prep_dur,
                        comp_dur,
                    )

            captions_processed += prepared_batch["input_ids"].size(0)
            step_time += compute_end - step_wall_start
            accumulation_loss += batch_loss_value
            if (i + 1) % config["accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                avg_accumulated = accumulation_loss / config["accumulation_steps"]
                total_train_loss += avg_accumulated
                num_train_batches += 1
                grad_norm = total_norm.item()
                accumulation_loss = 0.0
            progress.update(1)
        progress.close()
        epoch_duration = time.perf_counter() - epoch_start
        tracked_sections = data_time + prepare_time + compute_time
        if tracked_sections > 0:
            logging.info(
                "Epoch %s timing (s): load %.2f (%.1f%%) | prepare %.2f (%.1f%%) | compute %.2f (%.1f%%) | step %.2f | epoch %.2f",
                epoch + 1,
                data_time,
                (data_time / tracked_sections) * 100.0,
                prepare_time,
                (prepare_time / tracked_sections) * 100.0,
                compute_time,
                (compute_time / tracked_sections) * 100.0,
                step_time,
                epoch_duration,
            )
        if epoch_duration > 0 and captions_processed > 0:
            logging.info(
                "Epoch %s throughput: %.1f captions/sec (total %d captions)",
                epoch + 1,
                captions_processed / epoch_duration,
                captions_processed,
            )
        timing_metrics = {
            "timing/load_seconds": data_time,
            "timing/prepare_seconds": prepare_time,
            "timing/compute_seconds": compute_time,
            "timing/step_seconds": step_time,
            "timing/epoch_seconds": epoch_duration,
            "timing/captions_per_sec": (
                captions_processed / epoch_duration if epoch_duration > 0 else 0.0
            ),
            "timing/captions_processed": captions_processed,
        }
        if run:
            run.log(timing_metrics)

        logging.info(f"Epoch {epoch + 1}/{config['epochs']}")
        avg_train_loss = (
            total_train_loss / num_train_batches
            if num_train_batches > 0
            else total_train_loss
        )
        avg_val_loss = validate_model(
            model,
            vision_encoder,
            validation_dataloader,
            device,
            config,
            pad_token_id,
            maybe_autocast,
            using_precomputed,
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
                    "use_mlp_projector": config.get("use_mlp_projector", False),
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
        model,
        vision_encoder,
        test_dataloader,
        device,
        config,
        pad_token_id,
        maybe_autocast,
        using_precomputed,
    )
    run.log(
        {"test_loss": test_loss},
    )
    if not args.check:
        name = "qwen-decoder-model"
    run.finish(0)


if __name__ == "__main__":
    main()
