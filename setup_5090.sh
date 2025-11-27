cd /workspace/flash-attention




# May be needed on RTX 5090 - install PyTorch FIRST before compiling flash-attn
# The below should only be done if nvidia-smi shows we're actually running CUDA 12.9
# pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
# Only needed for synthetic-generation (image-captioning), api server (audio) and SFT (fine-tuning)
# Must be installed AFTER PyTorch to compile against the correct version
# Uninstall first, then compile from source using the CUDA arch env vars set above
# --no-cache-dir bypasses pip's wheel cache to force source build
pip uninstall -y flash-attn
pip install flash-attn --no-build-isolation --no-binary flash-attn --no-cache-dir