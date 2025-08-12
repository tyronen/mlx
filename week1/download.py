import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
import argparse
from multiprocessing import Pool

# Run pip install sqlalchemy psycopg2-binary before running this

db_url = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
titles_file = "data/titles.parquet"
comments_file = "data/comments.parquet"
items_file = "data/items.parquet"
users_file = "data/users.parquet"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


def download_query(engine, query, outfile):
    total_rows = 0
    logging.info(f"Downloading to {outfile}...")
    chunks = pd.read_sql_query(query, engine, chunksize=1_000_000)
    writer = None
    for chunk in chunks:
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(outfile, table.schema)
        writer.write_table(table)
        total_rows += len(chunk)
        logging.info(f"⏳ Downloaded {total_rows} rows")
    if writer:
        writer.close()
    logging.info(f"✅ Saved {outfile} ({total_rows} rows)")



def download_items_range(start_id, step, outfile_base, engine_url):
    end_id = start_id + step
    chunk_file = outfile_base.replace(".parquet", f"_{start_id}_{end_id}.parquet")
    query = f'''
        SELECT *
        FROM hacker_news.items
        WHERE id >= {start_id} AND id < {end_id}
    '''
    engine = create_engine(engine_url)
    download_query(engine, query, chunk_file)
    return chunk_file

def download_items_in_chunks(engine, outfile, step=1_000_000, id_start=0, id_end=40_000_000):
    ranges = [(start_id, step, outfile, db_url) for start_id in range(id_start, id_end, step)]
    with Pool(processes=4) as pool:
        output_files = pool.starmap(download_items_range, ranges)
    tables = [pq.read_table(f) for f in output_files]
    pq.write_table(pa.concat_tables(tables), outfile)


def download_hn_data(args):
    """Download HN data from PostgreSQL"""

    # Create database connection
    logging.info("Connecting to database...")
    engine = create_engine(db_url)

    # Query 1: High-scoring titles
    titles_query = """
       SELECT title
       FROM hacker_news.items
       WHERE title IS NOT NULL
         AND dead IS NULL
         AND score > 2"""

    # Query 2: Long comments with children
    comments_query = """
         SELECT text
         FROM hacker_news.items
         WHERE LENGTH(text) > 2048
           AND CARDINALITY(KIDS) > 4
           AND dead IS NULL"""

    # Query 4: Users
    users_query = """
        SELECT
          COALESCE(u.id, i."by") AS user_id,
          MIN(u.created) AS user_created,
          MIN(i.time) AS first_post_time
        FROM
          hacker_news.users u
        FULL OUTER JOIN
          hacker_news.items i
          ON u.id = i."by"
        GROUP BY
          COALESCE(u.id, i."by")
    """

    if args.comments:
        download_query(engine, comments_query, comments_file)
    if args.titles:
        download_query(engine, titles_query, titles_file)
    if args.users:
        download_query(engine, users_query, users_file)
    if args.items:
        download_items_in_chunks(engine, items_file)
    logging.info(f"✅ All requested data saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", action="store_true", help="Download titles data")
    parser.add_argument("--comments", action="store_true", help="Download comments data")
    parser.add_argument("--items", action="store_true", help="Download items data")
    parser.add_argument("--users", action="store_true", help="Download users data")
    args = parser.parse_args()
    download_hn_data(args)
