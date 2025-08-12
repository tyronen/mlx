# recurrent-rebels

MLX Institute, Recurrent Rebels group.
Try me out here: http://91.99.96.27:8501/

## How to run this project

Install requirements: Run `pip install -r requirements.txt`

## Directories

- `app` - inference server
- `big_model` - prediction model code
- `data` holds raw data files, including downloads from SQL
- `embeddings` - embeddings
- `models` - neural network output
- `runs` - output of training runs
- `skipgram_models` - skipgram models
- `title_regression_model` - some work done on determining title embeddings
- `utils` - some utils code

## Downloading data from the HackerNews DB

`download.py` will download whatever you ask it to, using command line flags. Available choices are titles, comments, items (posts only) and users. Data is saved in Pandas parquet format.

## Building user features

`tokenizer.py` will tokenize comments and titles into `hn_corpus.txt`.

`process_items.py` takes the large file downloaded (`items.parquet`) and backfills a lot of user data - at post time! into it and creates a new file, `posts.parquet`.

## Build prediction models for title embeddings

To run the word2vec code, run `python word2vec.py `from the command line. Or use the notebook `word2vec.ipynb`, which wraps it, on Google Colab. This produces embeddings in the filename specified as the argument. Specify `--corpus` for the corpus and `--model` for the output file.

Once the model is built, you can test it with `python tester.py <word> <model-file>`. Or even a bunch of words: `python word2vec_test.py <model-file>`

We can evaluate models side by side with `python word2vec_eval.py model1 model2`


## Training the model

`python big_model/mp_preprocess_data.py` takes the input data (`posts.parquet`) and preprocesses it into large tensor files in both training and test sets.

Then you can run `python big_model/train.py` to actually train the model.

## Inference
Model inference: To predict a score for an HN post, provide a dict with (at least) the following fields:
- `by: str` -- username of the author 
- `title: str` -- title of the post
- `url: str` -- domain associated to the post
- `time: int` -- time in [UNIX TIME](https://en.wikipedia.org/wiki/Unix_time)

Then run ... for inference.

For more information, see https://github.com/HackerNews/API

## FastAPI and Streamlit
You must create an inference cache first: `python big_model/inference_preprocess.py`.

Specify the path to your model in the `MODEL_PATH` environment variable.

Start the app: `uvicorn app.main:app --reload` to run locally on http://127.0.0.1:8000/

To run a prediction on HN post with a given `id`: Call `http://127.0.0.1:8000/predict_hn/{id}`
(e.g. `http://127.0.0.1:8000/predict_hn/130`)

To interact with a graphical interface, start the streamlit app:
`streamlit run streamlit_app.py`
and open it in your browser (this needs to run parallel to the FastAPI). Have fun!

## Data features we built

### ✅ User Features

- Karma  ❌ (leakage)
- Number of submitted posts (total posts)  ✅
- Max upvotes per post  ✅
- Min upvotes per post  ✅
- Mean upvotes per post  ✅
- Posts per year ✅
- Descendants (total number of comments under the post) ✅  
- Deepest sub comment level  ✅
- Number of comments / sub comments ✅ 
- Mean number of comments per post  ✅
- Total number of comments  ✅
- Account age (time they have been a user)  ✅
- Average number of people engaging on their post (same as average comments per post?) ✅

### 🕒 Temporal Features

- Post time of day (hour bucket: morning, afternoon, evening, night) 
- Post weekday/weekend indicator
- Account age at post time (how old the account was when posting) ✅
- Post within first N minutes of daily HN activity cycle
- Days since previous post ✅

## 🔄 User Interaction Diversity

- Number of distinct people commenting on user’s posts ✅
- Number of unique threads user participated in (non-own posts) ✅
- Fraction of posts that got at least 1 upvote (success rate) ✅

###  📝 Post Content Features

- Title length (word count, character count)
- Average word length in title
- Presence of question mark (`?`)
- Presence of exclamation mark (`!`)
- Sentiment score of title
- Embedding of title (TF-IDF, Word2Vec, BERT etc.)
- Title language (non-English titles tend to perform worse): can do this as binary

###  🔗 URL Features

- Domain popularity (e.g., github.com, nytimes.com, etc.)
- External domain reputation (popular domains may have higher base upvotes)
- Is submission linking to user's own site (self-promotion)

###  👥 Community-specific Features

- Submission category/topic (AI, Crypto, Programming, Startups, etc.)
- Average performance of user's previous posts in same category
- Number of reposts of the same URL

###  ⚡ Meta Engagement Features

- Average time to first comment ✅
- Time to 10 comments (velocity indicator) ✅
- Fraction of posts receiving moderator flag ✅
- Upvote velocity (upvotes per minute/hour after posting) ❌ (don't have voting times)
- Time taken to hit score thresholds (10, 50, 100 upvotes) ❌ (don't have voting times)

###  💡 Additional Advanced Features

- Network centrality (user’s interaction graph on HN)
- User influence score (e.g., followers or mentions elsewhere)
