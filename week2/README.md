# MLXTwoTowers
Week 2 coursework - Search and Retrieval


## Tyrone's implemenation

To get a complete server running end to end, you should have a Computa server.

### Setup the environment

```bash
# on your machine
pip install -r requirements.txt
mkdir data
ssh -i <key-file> -p <computa-port> root@<computa-server>
# on Computa server
git clone https://github.com/AdamBeedell/MLXTwoTowers.git
cd MLXTwoTowers
pip install wandb tqdm
mkdir data
```

### Running the core workflow

```bash
# On Computa server
python word2vec.py
# On your machine
scp -C -i <key-file> -P <computa-port> root@216.249.100.66:MLXTwoTowers/data/word2vec_skipgram.pth data/
python preprocess_data.py 
scp -C -i <key-file> -P <computa-port> data/datasets.pt root@216.249.100.66:MLXTwoTowers/data/
# On Computa server
python train_models.py
# On your machine
scp -C -i <key-file> -P <computa-port> root@216.249.100.66:MLXTwoTowers/data/models.pth data/
docker compose up --build
python store_documents.py
```

You can now visit http://localhost:8000/search?query=url+encoded+query. Click pretty-print to see it as nice JSON.
