# mlx

Portfolio of projects written during the MLX Institute course

## Week 0 - CNN over MNist data

Week 0's project (written before the course started) was to use CNN to do single-digit recognition on the MNIST data set.

You can train the model with:

```
pip install -r training/requirements.txt -c training/constraints.txt
python -m training.train_mnist_cnn --model_path data/mnist_cnn.pth
```

Build inference:

```
docker build --platform linux/amd64 -t docker.io/$DOCKER_USERNAME/inference:v$VERSION -f inference/Dockerfile .
```

To run inference locally, `test_inference.sh`.

Deploy to serverless as follows (from laptop):

```
docker push $DOCKER_USERNAME/inference:v$VERSION
```

Create a serverless endpoint on RunPod using the Serverless console. Point it at `docker.io/$DOCKER_USERNAME/inference:v$VERSION`. You must update the version after each push.

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
```


5. Start on your server:

```
# dev machine
docker push docker.io/$DOCKER_USERNAME/ui:latest
# server
export ENV=~/.env.prod
./ui_prod.sh
```

