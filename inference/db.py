import logging
import os
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_DATABASE_URL = os.environ.get("DATABASE_URL")

# ------- Engine / Connection helpers (non-Streamlit path) -------
_ENGINE: Engine | None = None


def _get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(_DATABASE_URL, pool_pre_ping=True)
    return _ENGINE


@contextmanager
def _session():
    """Simple context manager yielding a connection w/ transaction."""
    engine = _get_engine()
    with engine.begin() as conn:
        yield conn


# ------- Public API -------
def setup_database():
    """
    Create the predictions table if it doesn't exist.
    """
    with _session() as conn:
        conn.execute(
            text(
                """CREATE TABLE IF NOT EXISTS predictions (
                   id SERIAL PRIMARY KEY,
                   timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                   prediction SMALLINT,
                   confidence REAL,
                   true_label SMALLINT
               )"""
            )
        )
    logging.info("Database setup complete!")


def log_prediction(prediction: int, confidence: float, true_label: int):
    """
    Insert one row. Safe to call from RunPod Serverless handler.
    """
    with _session() as conn:
        conn.execute(
            text(
                """INSERT INTO predictions (prediction, confidence, true_label)
                    VALUES (:pred, :conf, :label)"""
            ),
            {"pred": prediction, "conf": confidence, "label": true_label},
        )


def get_all_predictions(tz: str = "Europe/London"):
    """
    Return a pandas DataFrame of recent predictions (UI convenience).
    """

    sql = """SELECT timestamp AT TIME ZONE 'UTC' AS timestamp_utc,
                    prediction, confidence, true_label
             FROM predictions
             ORDER BY timestamp DESC"""

    df = pd.read_sql(sql, _get_engine())

    if not df.empty:
        # Convert to desired timezone and prettify columns
        df["timestamp"] = (
            pd.to_datetime(df["timestamp_utc"], utc=True)
            .dt.tz_convert(tz)
            .dt.tz_localize(None)  # Streamlit tables look cleaner w/o tz suffix
        )
        df = df.drop(columns=["timestamp_utc"])
        df = df.rename(
            columns={
                "timestamp": "Timestamp",
                "prediction": "Prediction",
                "confidence": "Confidence",
                "true_label": "True Label",
            }
        )
    return df
