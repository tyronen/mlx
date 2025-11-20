# mlx

Portfolio of projects written during the MLX Institute course

## Build environment

This was built running on a Mac (Silicon) as the dev machine and a RunPod Ubuntu instance with GPU.

Because RunPods are rented for only a few hours at a time, every time you start a new instance, run `./setup_dev.sh`. This assumes you have only one RunPod instance, it queries for that and creates an entry in your `~/.ssh/config` pointing to it. You are also assumed to have a Network Volume on it mounted at `/workspace/`.

We do not set up a Git repo directly on the pod; merely use `rsync` to copy code and data between dev machine and pod.

We use RunPod's ramdisk (in `/dev/shm/`) as the cache for HuggingFace and other downloads to save time. Back this up to permanent storage before terminating your pod to the day:

```
rsync -vrt /dev/shm /workspace/ramdiskbackup/
```

## Week 0 - CNN over MNist data

Week 0's project (written before the course started) was to use CNN to do single-digit recognition on the MNIST data set.

You can train the model with:

```
pip install -r training/requirements.txt -c training/constraints.txt
python -m training.train_mnist_cnn --model_path data/mnist_cnn.pth
```

## Week 1 - Hacker News Predictor

### On the dev machine

`python -m training.hn_predict.prepare_indices` prepares preliminary word indices used to train title embeddings.

`python -m training.hn_predict.process_items` takes the OpenPipe/HackerNews dataset and backfills a lot of user data - at post time! into it and creates a new file, `data/posts.parquet`. This needs large RAM and disk space but no GPU.

`python -m training.hn_predict.build_cache` needs to run to prepare a cache used both at the next step and at inference.

`python -m training.hn_predict.mp_preprocess_data` takes the input data and cache and preprocesses it into large tensor files in both training and test sets.

These scripts produce two final input files: `train.pt` and `val.pt`. Upload these to your GPU machine.

### On the GPU

Run `python -m training.hn_predict.train_hurdle_model` to actually train the model. The best output will be in `data/best_quantile_epoch_n.pth` choose the highest value of `n` from the most recent run. Download this to your deployment machine for inference.

### Inference

A simple server environment like AWS Lightsail will do the job.

1. Install Docker Engine on your server. Docker should be runnable by someone other than root. Follow the instructions on the Docker website.
2. Docker Desktop on your dev machine must have containerd enabled.
3. Debug locally.

```
export ENV=./.env.dev
./ui_dev.sh
```

4. Build full image on your dev machine:

```
./build.sh
./ui_staging.sh
```

5. Start on your server:

```
./deploy.sh
```

## Week 2 - MS Marco Search

1. On your dev machine, run `python -m training.msmarco_search.word2vec_prereq`.
2. Upload the directory `data/msmarco_search` to your GPU.
3. On the GPU run `python -m training.msmarco_search.word2vec`
4. Download `data/word2vec_skipgram.pth` back to dev
5. On dev, run `python -m training.msmarco_search.preprocess_data`
6. Upload `data/ms_marco_data.pt` back to GPU.
7. On GPU run `python -m training.msmarco_search.train_models`
8. Download `data/twin_towers.pth` from GPU.
9. On the UX machine run `./ui_dev.sh`
10. While Redis and the webserver are running, run `python -m training.store_documents`. This builds the index.
11. Test the Docker setup with `./ui_staging.sh`
12. If it works, push it with `./deploy.sh`

## Week 3 - 4-way image recogniser

```
# dev machine
python -m training.complex_mnist.create_composite_images
rsync data/composite*.pt <GPU>:/<path>/data/
# GPU
python -m training.complex_mnist.train_complex_model
# dev machine
rsync <GPU>L:/path/data/complex_mnist.pth data/
./ui_dev.sh
```

## Week 4 - Image captioner

On some hardwares, CUDA registers as unavailable. You can run `check_gpu.py` to sometimes correct this.

```
# Generate synthetic dataset from COCO
python -m training.image_caption.synthetic_generator
python -m training.image_caption.fix_metadata
# Precompute image features
python -m training.image_caption.precompute_images
python -m training.image_caption.precompute_images --dataset=flickr
python -m training.image_caption.train_models --dataset=flickr
python -m training.image_caption.train_models
```

## Week 5a - Urban sounds

```
# GPU
python -m training.urban_sounds.audio_data
python -m traiing.urban_sounds.train_model
```

This was demonstration exercise which merely attempted to train the model, not use it.

## Week 5b - Audio transcode
