import logging
import pandas as pd
import os
import pickle

CACHE_FILE = "data/inference_cache.pkl"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

DROP_COLS = ["id", "score"]
def save_user_data():
    """
    Return (columns, user_features, global_Tmin, global_Tmax).

    If a serialized cache exists, load from it; otherwise build the cache
    from `data/posts.parquet` and save it for next time.
    """
    # No cache yet → build it
    logging.info("Building inference cache …")
    posts_df = pd.read_parquet("data/posts.parquet")
    global_Tmin = posts_df["time"].min()
    global_Tmax = posts_df["time"].max()

    cols = [col for col in posts_df.columns if col not in DROP_COLS]
    user_features = {
        user: group.iloc[-1].drop(labels=DROP_COLS).to_dict()
        for user, group in posts_df.groupby("by", sort=False)
    }

    cache = dict(
        columns=cols,
        global_Tmin=global_Tmin,
        global_Tmax=global_Tmax,
        user_features=user_features,
    )
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(cache, fh)
    logging.info(f"Saved inference cache to {CACHE_FILE}")

if __name__ == "__main__":
    # Build or refresh the cache on demand
    save_user_data()
    logging.info("Done.")