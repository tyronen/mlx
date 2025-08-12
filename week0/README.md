# PyTorch Digit Recogniser

Built per the instructions at https://programme.mlx.institute/interview/project?hash=iBFvgvTsYLGG

## Access the running instance

Currently live at http://pytorch.tyronenicholas.com. If DNS has not yet propagated, try http://18.175.38.13. Deployed
to AWS Lightsail.

## Running locally

You should already have Docker and Python3 installed.

Run the following once:

```
pip install -r requirements.txt
pip install pytorch
```

To train the model, run `python train_model.py`. To run the webserver, run `docker-compose up`. You can press Ctrl-C to
stop the webserver.

## Build and deploy instructions

Any hosting service accepting Docker will do. You need at least 2GB RAM.

### Setup

1. Install Docker Engine on your server. Docker should be runnable by someone other than root. Follow the instructions
   on the Docker website.
2. Your server must have a static IP address.
3. Create an entry in your `~/.ssh/config` file, or use `ssh-agent` to alias a name to your server that incorporates its
   username and any private key needed to log into it.
4. `docker context create server-name --docker "ssh://server-name"`

### Deploy

You must have trained the model locally before running these commands.

```
docker context use server-name
docker compose -f docker-compose.yml --env-file .env.prod up --build -d
```