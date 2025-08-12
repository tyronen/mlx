import pandas as pd
import argparse
import logging
import numpy as np

from collections import defaultdict

SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY

parser = argparse.ArgumentParser()
parser.add_argument("--items", default="data/items.parquet", help="Items file")
parser.add_argument("--posts", default="data/posts.parquet", help="Posts file")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

def mean(x):
    return 0 if len(x) == 0 else sum(x) / len(x)

def percent(num, denom):
    return np.where(denom == 0, 0, 100.0 * num / denom)

def comment_calculations(parent_map, comments, _story_ids_unused=None):
    """
    Compute per‑story maximum depth and arrival times using a single
    Python itertuples loop (faster than using pandas).

    Returns
    -------
    depth_dict : dict[story_id, int]
        Maximum depth reached under each story.
    story_comment_times : dict[story_id, list[int]]
        Arrival timestamps (epoch‑seconds) of every comment under
        the story (unsorted).
    """
    depth_dict = defaultdict(int)
    story_comment_times = defaultdict(list)
    depth_cache = {}

    def _depth(cid):
        if cid in depth_cache:
            return depth_cache[cid]
        pid = parent_map.get(cid)
        if pid is None or pd.isna(pid):
            depth_cache[cid] = 0
            return 0
        d = _depth(pid) + 1
        depth_cache[cid] = d
        return d

    for row in comments.itertuples(index=False):
        d = _depth(row.id)
        root = row.id
        p = parent_map.get(root)
        while p is not None and not pd.isna(p):
            root = p
            p = parent_map.get(root)

        if d > depth_dict[root]:
            depth_dict[root] = d
        story_comment_times[root].append(row.time)

    return depth_dict, story_comment_times

def get_story_author(parent_map, story_author_map, comment_id):
    # Ascend to the root story to see whose post this comment belongs to
    pid = parent_map.get(comment_id)
    while pid is not None and not pd.isna(pid):
        comment_id = pid
        pid = parent_map.get(comment_id)
    return story_author_map.get(comment_id)

# Running min / max / mean helper
def expanding_shifted(stories_df, col, fn):
    return stories_df.groupby("by", sort=False)[col].transform(
        lambda x: getattr(x.expanding(), fn)().shift())


def main(items_file, posts_file):
    logging.info(f"Reading {items_file}")
    items = pd.read_parquet(items_file)
    logging.info(f"Sorting items")
    items = items.sort_values("time").reset_index(drop=True)
    logging.info("Filtering for dead")
    not_dead = items[items["dead"].isnull()]
    logging.info("Building parent lookup map")
    parent_map = not_dead.set_index("id")["parent"].to_dict()
    logging.info("Filtering for stories")
    stories = not_dead[not_dead["type"] == "story"].set_index("id")
    logging.info("Filtering for comments")
    comments = not_dead[not_dead["type"] == "comment"].copy()
    logging.info("Calculating comment length")
    comments["text_length"] = comments["text"].fillna("").str.len()
    logging.info("Grouping by parent")
    groups = comments.groupby("parent")
    logging.info("Calculating comment counts")
    comment_counts = groups.size().to_dict()
    logging.info("Calculating distinct commenters")
    distinct_commenter_counts = (
        comments.dropna(subset=["by"])
                .groupby("parent", sort=False)["by"]
                .nunique()
                .astype(int)
                .to_dict()
    )
    logging.info("Building story → time map")
    story_time_map = stories["time"].to_dict()

    logging.info("Calculating depth and comment arrival times per story")
    depth_dict, story_comment_times = comment_calculations(parent_map, comments)

    logging.info("Computing first/10th comment delays")
    first_comment_delta = {}
    tenth_comment_delta = {}
    for sid, times in story_comment_times.items():
        times.sort()
        t_story = story_time_map.get(sid)
        if t_story is None:
            continue  # skip if story timestamp missing
        if len(times) >= 1:
            delay = (times[0] - t_story).total_seconds()
            first_comment_delta[sid] = max(delay, 0.0)
        if len(times) >= 10:
            delay10 = (times[9] - t_story).total_seconds()
            tenth_comment_delta[sid] = max(delay10, 0.0)

    # Compute all per‑story features without
    # any Python‑level per‑item loop.
    logging.info("Preparing story feature frame")
    stories_df = (
        stories.reset_index()               # bring id back as a column
               .sort_values("time")         # chronological for merge_asof later
               .copy()
    )
    story_ids = stories_df["id"]

    # 2. Cumulative / prior‑post user statistics ---------------------------
    # ------------------------------------------------------------------
    # Dead‑post stats must consider *all* stories (dead + live).
    # Build a small frame with cumulative dead counts per author, then
    # merge the relevant columns back onto stories_df (which only
    # contains live stories).
    # ------------------------------------------------------------------
    logging.info("Computing dead‑post history from all stories")
    stories_all = (
        items[items["type"] == "story"]
        .sort_values("time")
        .copy()
    )
    if stories_all["time"].dtype != "datetime64[ns]":
        stories_all["time"] = pd.to_datetime(stories_all["time"], unit="s")

    stories_all["is_dead"] = stories_all["dead"].notna().astype(int)
    all_stories_by_author = stories_all.groupby("by", sort=False)
    stories_all["num_posts_all"] = all_stories_by_author.cumcount()
    stories_all["cum_dead_posts"] = all_stories_by_author["is_dead"].cumsum().shift().fillna(0)

    dead_cols = stories_all.loc[stories_all["dead"].isnull(), ["id", "num_posts_all", "cum_dead_posts"]]
    stories_df = stories_df.merge(dead_cols, on="id", how="left")

    logging.info("Per‑story fields")
    stories_df["num_comments"]      = story_ids.map(comment_counts).fillna(0).astype(int)
    stories_df["post_commenters"]   = story_ids.map(distinct_commenter_counts).fillna(0).astype(int)
    stories_df["depth"]             = story_ids.map(depth_dict).fillna(0).astype(int)
    stories_df["first_comment_delay"]  = story_ids.map(first_comment_delta).fillna(-1)
    stories_df["tenth_comment_delay"]  = story_ids.map(tenth_comment_delta).fillna(-1)
    stories_df["score_above_1"]        = (stories_df["score"] > 1).astype(int)

    logging.info("Elapsed time stats")
    if stories_df["time"].dtype != "datetime64[ns]":
        stories_df["time"] = pd.to_datetime(stories_df["time"], unit="s")

    # 2. Cumulative / prior‑post user statistics ---------------------------
    logging.info("Computing cumulative user statistics")
    stories_by_author = stories_df.groupby("by", sort=False)

    logging.info("Cumulative post counts")
    stories_df["num_posts"] = stories_by_author.cumcount()

    logging.info("First & previous post times")
    stories_df["first_post_time"]   = stories_by_author["time"].transform("first")
    stories_df["prev_post_time"]    = stories_by_author["time"].shift()

    secs_since_first = (stories_df["time"] - stories_df["first_post_time"]).dt.total_seconds()
    secs_since_prev  = (stories_df["time"] - stories_df["prev_post_time"]).dt.total_seconds()
    stories_df["days_since_first_post"] = secs_since_first / SECONDS_PER_DAY
    stories_df["days_since_last_post"]  = secs_since_prev  / SECONDS_PER_DAY
    # For a user's very first story, both deltas are undefined; use -1 sentinel
    first_post_mask = stories_df["num_posts"] == 0
    stories_df.loc[first_post_mask, ["days_since_first_post", "days_since_last_post"]] = -1
    stories_df["elapsed_years"]         = secs_since_first / SECONDS_PER_YEAR + 1
    stories_df["posts_per_year"]        = stories_df["num_posts"] / stories_df["elapsed_years"]

    logging.info("Cumulative counts for dead posts & scores>1")
    stories_df["cum_scores_gt1"] = (
            stories_by_author["score_above_1"].cumsum() - stories_df["score_above_1"]
    )
    stories_df["percent_scores_above_1"] = percent(stories_df["cum_scores_gt1"], stories_df["num_posts"])

    # Ensure numerator ≤ denominator (can mismatch if merge mis‑aligns)
    stories_df["cum_dead_posts"] = np.minimum(stories_df["cum_dead_posts"], stories_df["num_posts_all"])

    stories_df["percent_posts_dead"] = percent(
        stories_df["cum_dead_posts"], stories_df["num_posts_all"]
    )

    logging.info("Computing expanding mins / maxes / means")
    for col in ["depth", "descendants", "num_comments", "post_commenters",
                "first_comment_delay", "tenth_comment_delay", "score"]:
        stories_df[f"user_{col}_min"]  = expanding_shifted(stories_df, col, "min")
        stories_df[f"user_{col}_max"]  = expanding_shifted(stories_df, col, "max")
        stories_df[f"user_{col}_mean"] = expanding_shifted(stories_df, col, "mean")

    # On a user's very first post every user_* aggregate should be 0
    user_cols = [c for c in stories_df.columns if c.startswith("user_")]
    stories_df[user_cols] = stories_df[user_cols].fillna(0)

    # 3. Final selection and write‑out -------------------------------------
    output_cols = [
        "id", "by", "time", "title", "url", "score",
        "num_posts",
        "percent_posts_dead", "percent_scores_above_1",
        "posts_per_year", "days_since_first_post", "days_since_last_post",
        # user running aggregates (min/max/mean for each metric)
    ] + [c for c in stories_df.columns if c.startswith("user_")]
    logging.info(f"Output columns: {output_cols}")

    post_df = stories_df[output_cols].copy()

    logging.info(f"Writing {posts_file}")
    post_df.to_parquet(posts_file, index=False)
    logging.info(f"Wrote {post_df.shape[0]} posts")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.items, args.posts)