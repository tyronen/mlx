import time
import logging
from contextlib import asynccontextmanager
import torch
from transformers import (
    pipeline,
    Pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTimeStampLogitsProcessor,
)
from transformers.utils import is_flash_attn_2_available
import uvicorn

from typing import Optional

import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect
import utils


@asynccontextmanager
async def lifespan(_: FastAPI):
    utils.setup_logging()
    # 1. Load your model object (so you can grab its generation_config)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    _load_pipeline(model)
    yield


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Whisper model (loaded once at startup)
# ---------------------------------------------------------------------------


# Model guide:
# openai/whisper-tiny.en: 39m
# openai/whisper-base.en: 74m
# distil-whisper/distil-small.en: 166m, short-form wer 12.1
# openai/whisper-small.en: 244m
# distil-whisper/distil-medium.en: 394m, 11.1
# distil-whisper/distil-large-v3: 756m, 9.7
# openai/whisper-medium: 769m
# openai/whisper-large-v3-turbo: 809m
# openai/whisper-large-v3: 1550m, 8.4

MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-large-v3")
DEVICE_PREFERENCE = utils.get_device().type

asr_pipe: Optional[Pipeline] = None


@app.post("/model")
async def model(model_name: str = Form(...)):
    """
    Switch to a new model by name.
    """
    global MODEL_NAME, asr_pipe
    try:
        MODEL_NAME = model_name
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        _load_pipeline(model)
        return {"success": True, "model": MODEL_NAME}
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=400, detail=str(e))


def _load_pipeline(model) -> None:
    """
    Load the insanely‑fast‑whisper checkpoint via Hugging Face Transformers.
    Uses Flash‑Attention‑2 if available, otherwise SDPA attention.
    """
    global asr_pipe

    # Decide device string
    if DEVICE_PREFERENCE.startswith("cuda") and torch.cuda.is_available():
        device_str = DEVICE_PREFERENCE
    elif DEVICE_PREFERENCE == "mps" and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    attn_impl = (
        {"attn_implementation": "flash_attention_2"}
        if is_flash_attn_2_available()
        else {"attn_implementation": "sdpa"}
    )

    config = model.config
    # Set alignment_heads for specific models
    if "openai/whisper-tiny" in MODEL_NAME:
        config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
    elif "openai/whisper-tiny.en" in MODEL_NAME:
        config.alignment_heads = [
            [1, 0],
            [2, 0],
            [2, 5],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
        ]
    elif "openai/whisper-base" in MODEL_NAME:
        config.alignment_heads = [
            [3, 1],
            [4, 2],
            [4, 3],
            [4, 7],
            [5, 1],
            [5, 2],
            [5, 4],
            [5, 6],
        ]
    elif "openai/whisper-base.en" in MODEL_NAME:
        config.alignment_heads = [[3, 3], [4, 7], [5, 1], [5, 5], [5, 7]]
    elif "openai/whisper-medium" in MODEL_NAME:
        config.alignment_heads = [
            [13, 15],
            [15, 4],
            [15, 15],
            [16, 1],
            [20, 0],
            [23, 4],
        ]
    elif "openai/whisper-medium.en" in MODEL_NAME:
        # alignment heads for medium.en
        config.alignment_heads = [
            [11, 4],
            [14, 1],
            [14, 12],
            [14, 14],
            [15, 4],
            [16, 0],
            [16, 4],
            [16, 9],
            [17, 12],
            [17, 14],
            [18, 7],
            [18, 10],
            [18, 15],
            [20, 0],
            [20, 3],
            [20, 9],
            [20, 14],
            [21, 12],
        ]
    elif "openai/whisper-large-v3" in MODEL_NAME:
        config.alignment_heads = [
            [24, 3],
            [22, 18],
            [25, 6],
            [25, 17],
            [22, 14],
            [12, 16],
            [26, 7],
            [26, 6],
            [23, 12],
            [23, 5],
        ]

    # 2. Load your processor as before
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    # 3. Compute the “begin_index” as the number of forced_decoder_ids the model is already using:
    forced = getattr(model.generation_config, "forced_decoder_ids", [])
    begin_index = len(forced)

    # 4. Instantiate the logits‐processor with the model’s GenerationConfig and that index:
    ts_processor = WhisperTimeStampLogitsProcessor(
        model.generation_config,
        begin_index,
        _detect_timestamp_from_logprob=True,  # you can omit or set False if you like
    )

    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
        device=device_str,
        model_kwargs=attn_impl,
        chunk_length_s=ROLLING_WINDOW_SEC,
        stride_length_s=STRIDE_INTERVAL_SEC,
        return_timestamps="word",
        config=config,
        processor=processor,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        generate_kwargs={
            "task": "transcribe",
            "logits_processor": [ts_processor],
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
        },
        ignore_warning=True,
    )

    Path("chunks").mkdir(exist_ok=True)
    logging.info(f"Model {MODEL_NAME} loaded")


# ---------------------------------------------------------------------------
# WebSocket endpoint for raw 16-kHz PCM streaming
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
ROLLING_WINDOW_SEC = 29

# Initial exponential send intervals (seconds): then full window
INITIAL_SEND = 3
# After initial intervals, slide window
SLIDE_INTERVAL_SEC = ROLLING_WINDOW_SEC / 3
STRIDE_INTERVAL_SEC = ROLLING_WINDOW_SEC / 3
rolling_window_samples = ROLLING_WINDOW_SEC * SAMPLE_RATE


def transcribe_pipeline(rolling_buffer, buffer_duration):
    global asr_pipe
    result = asr_pipe(rolling_buffer)
    text = result["text"].strip()
    chunks = result.get("chunks", [])
    tokens = [c["text"] for c in chunks]
    word_timestamps = [c["timestamp"] for c in chunks]
    # Detect incomplete timestamps and fall back if needed
    if any(len(ts) != 2 for ts in word_timestamps):
        logging.warning(
            "Incomplete timestamp detected; falling back to uniform timing."
        )
        raw = asr_pipe(rolling_buffer, return_timestamps=False)
        text = raw["text"].strip()
        tokens = text.split()
        word_timestamps = []
        for i in range(len(tokens)):
            s = i * buffer_duration / len(tokens)
            t = (i + 1) * buffer_duration / len(tokens)
            word_timestamps.append((s, t))
    return text, tokens, word_timestamps


def transcribe(buf, chunk_id, received_samples, buffer_duration):
    global asr_pipe, asr_model
    silence = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.int16)
    buf = np.concatenate((buf, silence))
    rolling_buffer = buf.astype(np.float32) / 32768.0
    end_sec = received_samples / SAMPLE_RATE
    start_sec = max(0.0, end_sec - buffer_duration)
    start_trans = time.time()
    text, tokens, word_timestamps = transcribe_pipeline(rolling_buffer, buffer_duration)
    end_trans = time.time()
    # Include tokens and timestamps in the JSON you send back
    logging.info(
        {
            "transcribe_time": f"{end_trans - start_trans:.1f}",
            "start_sec": f"{start_sec:.0f}",
            "end_sec": f"{end_sec:.0f}",
            "chunk_id": chunk_id,
            "text": text,
        }
    )
    return {
        "chunk_id": chunk_id,
        "text": text,
        "tokens": tokens,
        "word_timestamps": word_timestamps,
        "start_sec": start_sec,
        "duration": buffer_duration,
    }


def transcribe_window(ring, chunk_id, received_samples):
    # Always use the last 30s (or less, if not enough)
    if ring.size >= rolling_window_samples:
        buf = ring[-rolling_window_samples:]
        buffer_duration = rolling_window_samples / SAMPLE_RATE
    else:
        buf = ring
        buffer_duration = ring.size / SAMPLE_RATE

    return transcribe(buf, chunk_id, received_samples, buffer_duration)


@app.websocket("/ws")
async def websocket_pcm(websocket: WebSocket):
    """
    Client sends binary Int16 PCM frames (little-endian, 16 kHz, mono).
    We keep a simple ring-buffer; every 32 000 samples (~2 s) we run the
    Whisper pipeline and push the text back.

    Outgoing message shape:
        {"chunk_id": <int>, "text": "<transcript>"}
    """
    await websocket.accept()

    # Make sure the ASR model is loaded
    global asr_pipe
    if asr_pipe is None and asr_model is None:
        await websocket.close(code=1011, reason="ASR model not ready")
        return

    ring = np.empty((0,), dtype=np.int16)  # rolling PCM buffer
    received_samples = 0
    chunk_id = 0

    # Emission scheduling state
    next_emit_sec = INITIAL_SEND
    next_emit_samples = int(next_emit_sec * SAMPLE_RATE)

    try:
        stream_ended = False
        while True:
            # Receive raw PCM bytes
            frame = await websocket.receive_bytes()
            if len(frame) == 0:
                stream_ended = True
                # fall through to flush logic below
            else:
                samples = np.frombuffer(frame, dtype=np.int16)
                # Append to ring-buffer
                ring = np.concatenate((ring, samples))
                received_samples += samples.size

                if received_samples >= next_emit_samples:
                    # if we’re more than one full window behind, jump straight to “now”
                    if received_samples - next_emit_samples > rolling_window_samples:
                        # reset the ring to just the last window
                        ring = ring[-rolling_window_samples:]

                    # transcribe the most recent window only
                    retval = transcribe_window(ring, chunk_id, received_samples)
                    await websocket.send_text(json.dumps(retval))
                    chunk_id += 1

                    # schedule the next emit exactly SLIDE_INTERVAL_SEC from now
                    next_emit_sec = SLIDE_INTERVAL_SEC
                    next_emit_samples = received_samples + int(
                        next_emit_sec * SAMPLE_RATE
                    )

            if ring.size > rolling_window_samples:
                ring = ring[-rolling_window_samples:]
            if stream_ended and ring.size > 0:
                # do one final pass on whatever is left
                buf = ring
                buffer_duration = buf.size / SAMPLE_RATE

                retval = transcribe(buf, chunk_id, received_samples, buffer_duration)
                await websocket.send_text(json.dumps(retval))
                # clear the ring so we don’t send again
                ring = np.empty((0,), dtype=np.int16)

            # Finally, once client stopped *and* ring is empty *and*
            # we’re not going to emit any more intervals, break and close
            if stream_ended and ring.size == 0 and received_samples < next_emit_samples:
                await websocket.close(code=1000, reason="flushed")
                return

    except WebSocketDisconnect as e:
        logging.error(f"WebSocket disconnected: {e}")
        # Client hung up – just exit the coroutine
        return


@app.post("/train")
async def train(audio: UploadFile = File(...), transcript: str = Form(...)):
    """
    Accepts an audio file and a corrected transcript. Fine-tunes the ASR model on this pair.
    """
    global asr_pipe

    # Only support insanely-fast-whisper for now (training requires PyTorch)
    if asr_pipe is None:
        raise HTTPException(
            status_code=503, detail="ASR model is not loaded or not trainable."
        )

    # Read the raw audio file into numpy
    audio_bytes = await audio.read()
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Prepare the model for training (single step)
    model = asr_pipe.model
    # Ensure training runs in full precision to avoid NaN issues
    model.float()
    original_param_states = {
        name: p.requires_grad for name, p in model.named_parameters()
    }

    processor = asr_pipe.processor

    # Encode audio and target
    input_features = asr_pipe.feature_extractor(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(model.device)

    # Convert dtype if model expects float16 (half precision)
    if model.dtype == torch.float16:
        input_features = input_features.half()
    elif model.dtype == torch.bfloat16:
        input_features = input_features.bfloat16()
    else:
        input_features = input_features.float()

    labels = (
        processor(text=transcript, return_tensors="pt")
        .input_ids.to(model.device)
        .long()
    )

    # Standard training loop for a single step
    model.train()
    # Freeze only the encoder; allow decoder and LM head to adapt
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    # Ensure all other parameters remain trainable
    for p in model.model.decoder.parameters():
        p.requires_grad = True
    # Unfreeze the model's output embeddings (lm_head), if present
    lm_head = model.get_output_embeddings()
    if lm_head is not None:
        for p in lm_head.parameters():
            p.requires_grad = True
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-6
    )

    # Compute pre-update loss
    outputs = model(input_features=input_features, labels=labels)
    pre_loss = float(outputs.loss.detach().cpu().item())

    # Single optimization step
    optimizer.zero_grad()
    outputs.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Restore original training mode and parameter states
    for name, p in model.named_parameters():
        if name in original_param_states:
            p.requires_grad = original_param_states[name]

    # Evaluate post-update loss on the same inputs
    model.eval()
    with torch.no_grad():
        post_outputs = model(input_features=input_features, labels=labels)
    post_loss = float(post_outputs.loss.detach().cpu().item())

    summary = f"{transcript[:30]}..." if len(transcript) > 30 else transcript
    logging.info(
        f"Pre-training loss {pre_loss:.3f}, Post-training loss {post_loss:.3f}: {summary}"
    )
    _load_pipeline(model)
    return {
        "success": True,
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# If run as main, start Uvicorn on all interfaces, honoring proxy headers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8888")),
        proxy_headers=True,
    )
