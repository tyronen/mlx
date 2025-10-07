# mlx

Portfolio of projects written during the MLX Institute course

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

## UX server

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
scp ui_prod.sh <server>:ui_prod.sh
scp docker-compose.yml <server>:docker-compose.yml
# on server
./ui_prod.sh
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
9. On the UX machine run `python -m training.store_documents`
10. Run inference - TODO may have to replace dockerfile with docker compose, need Redis on Lightsail
