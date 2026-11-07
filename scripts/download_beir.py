"""Download a BEIR dataset into data/<name>/.

    python scripts/download_beir.py scifact

Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip
(The zip contains a top-level <name>/ folder.)

TODO: download, extract, and skip the download if the data is already there.
"""
from pathlib import Path
import requests
import sys

def download_beir(name:str) -> None:
    """Download a BEIR dataset into data/<name>/."""
    try:
        files=["corpus.jsonl", "queries.jsonl", "qrels/test.tsv"]
        data_path=Path(__file__).resolve().parent.parent/"data"
        for rel_path in files:
            link=f"https://huggingface.co/datasets/mteb/{name}/resolve/main/{rel_path}"
            res=requests.get(link,stream=True,timeout=10)
            res.raise_for_status()
            file_path=data_path / name / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path,mode="wb") as file:
                 for content in res.iter_content(chunk_size=8192):
                     file.write(content)
        print(f"Downloaded {name} dataset to {data_path}.")
    except Exception as e:
        print(f"Error downloading {name} dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    name=sys.argv[1] if len(sys.argv) > 1 else "scifact"
    if Path(f"data/{name}").exists():
        print(f"Dataset {name} already exists in data/{name}/.")
        sys.exit(0)
    download_beir(name)