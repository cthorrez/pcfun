import requests
import polars as pl
import numpy as np

DATA_URL = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240629_public.json"

def download_data(url=DATA_URL):
    file_name = url.split("/")[-1] 
    response = requests.get(url)
    with open(f"data/{file_name}", "wb") as out_file:
        out_file.write(response.content)


def process_data(file_name=DATA_URL.split("/")[-1]):
    df = pl.read_json(f"data/{file_name}")
    df = df.filter(
       (pl.col("anony") == True)
       & (pl.col("winner") != 'tie')
    ).sort("tstamp")
    df = df.with_columns(
        pl.when(pl.col("winner") == "model_a").then(pl.col("model_a")).otherwise(pl.col("model_b")).alias("model_a"),
        pl.when(pl.col("winner") == "model_a").then(pl.col("model_b")).otherwise(pl.col("model_a")).alias("model_b"),
    )

    models = pl.DataFrame({"model" : pl.concat([df["model_a"], df["model_b"]]).unique().sort()}).with_row_index()
    print(f"Num models: {len(models)}")

    df = df.join(models.rename({"model": "model_a", "index": "index_a"}), on="model_a")
    df = df.join(models.rename({"model": "model_b", "index": "index_b"}), on="model_b")

    matchups = df.select("index_a", "index_b").to_numpy().astype(np.int32)
    np.savez("data/matchups.npz", matchups=matchups)


if __name__ == '__main__':
    # download_data()
    process_data()
    
