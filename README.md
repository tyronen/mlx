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

`tokenizer.py` will tokenize comments and titles into `hn_corpus.txt`.

`process_items.py` takes the large file downloaded (`items.parquet`) and backfills a lot of user data - at post time! into it and creates a new file, `posts.parquet`.

To run the word2vec code, run `python -m training.hn_predict.word2vec` from the command line.

We can evaluate models side by side with `python -m training.hn_predict.word2vec_eval model1 model2`

`python training/hn_predict/mp_preprocess_data.py` takes the input data (`posts.parquet`) and preprocesses it into large tensor files in both training and test sets.

Then you can run `python training/hn_predict/train_hurden_model.py` to actually train the model.

## UX server

A simple server environment like AWS Lightsail will do the job.

1. Install Docker Engine on your server. Docker should be runnable by someone other than root. Follow the instructions on the Docker website.
2. Docker Desktop on your dev machine must have containerd enabled.
2. Create a file `.env.prod` on the server containing:

```
RUNPOD_API_KEY=<your RunPod api key>
RUNPOD_ENDPOINT="https://api.runpod.ai/v2/<runpod-serverless-endpoint-id>/runsync"
```

3. Debug locally. You need to override the port. Copy `.env.prod` to `.env.dev` and add the line `PORT=8000`. Then:

```
export ENV=./.env.dev
./ui_dev.sh
```

4. Build on your dev machine:

```
./build.sh ui
scp ui_prod.sh <server>:ui_prod.sh
```


5. Start on your server:

```
./ui_prod.sh
```

