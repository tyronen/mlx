import os

import streamlit as st
from sqlalchemy import text


@st.cache_resource
def get_connection():
    return st.connection(
        "postgres",
        type="sql",
        url=f"postgresql://{os.environ.get('POSTGRES_USER', 'postgres')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'postgres')}@"
        f"{os.environ.get('DB_HOST', 'postgres')}/"
        f"{os.environ.get('POSTGRES_DB', 'mnist')}",
    )


def setup_database():
    conn = get_connection()
    with conn.session as session:
        session.execute(
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
        session.commit()
    print("Database setup complete!")


def log_prediction(prediction, confidence, true_label):
    conn = get_connection()
    with conn.session as session:
        session.execute(
            text(
                """INSERT INTO predictions (prediction, confidence, true_label) 
                   VALUES (:pred, :conf, :label)"""
            ),
            {"pred": prediction, "conf": confidence, "label": true_label},
        )
        session.commit()


@st.cache_data
def get_all_predictions():
    conn = get_connection()
    df = conn.query(
        """SELECT timestamp, prediction, confidence, true_label
           FROM predictions ORDER BY timestamp DESC""",
        ttl="2s",
    )
    if not df.empty:
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/London")
        df.columns = [col.replace("_", " ").title() for col in df.columns]
    return df
