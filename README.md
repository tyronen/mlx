# mlx

Portfolio of projects written during the MLX Institute course

## Week 0 - CNN over MNist data

Week 0's project (written before the course started) was to use CNN to do single-digit recognition on the MNIST data set.

You can train the model with:

```
pip install -r training/requirements.txt -c training/constraints.txt
python -m training.train_mnist_cnn --model_path data/mnist_cnn.pth
```

Deploy to serverless as follows (from laptop):

```
docker build --platform linux/amd64 -t $DOCKER_USERNAME/inference:v$VERSION -f inference/Dockerfile .
docker push $DOCKER_USERNAME/inference:v$VERSION
```

Create a serverless endpoint using `docker.io/$DOCKER_USERNAME/inference`