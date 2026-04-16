"""
Download the Parkinson's Disease dataset and save it as data/parkinsons.csv.

Strategy
--------
1. Try the canonical UCI ML Repository URL first.
2. If UCI is unavailable (server error, timeout), fall back to OpenML.
   The OpenML copy is recoded to match the UCI schema exactly.

Dataset : Parkinson's Disease Data Set
Source  : UCI Machine Learning Repository (primary)
          OpenML dataset #1488 (fallback)
"""
import io
import os
import ssl
import urllib.request


# URLs and column metadata
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
)

OPENML_META_URL  = "https://api.openml.org/api/v1/json/data/1488"
OPENML_DATA_URL  = "https://api.openml.org/data/v1/download/1592280"

DATA_PATH = "data/parkinsons.csv"

# Proper UCI column names for the 22 acoustic features
UCI_FEATURE_COLS = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]


# Helpers
def _ssl_context():
    """Return an SSL context; tolerate environments without CA bundles."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _fetch(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; academic-download/1.0)"},
    )
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=timeout) as r:
        return r.read()


# Downloads
def _try_uci() -> str:
    """Return raw CSV text from UCI, or raise an exception."""
    content = _fetch(UCI_URL).decode("utf-8")
    # Quick sanity check: UCI file has a header row starting with 'name'
    if not content.strip().startswith("name"):
        raise ValueError("Unexpected content from UCI URL")
    return content


def _try_openml() -> str:
    """Download the ARFF from OpenML and convert to UCI-compatible CSV."""
    import csv

    raw = _fetch(OPENML_DATA_URL).decode("utf-8")

    # Parse the @data section of the ARFF file
    data_lines = []
    in_data = False
    for line in raw.splitlines():
        if line.strip().lower() == "@data":
            in_data = True
            continue
        if in_data and line.strip():
            data_lines.append(line.strip())

    # Build a CSV string matching the UCI format
    # UCI: name, 22 features, status  (status: 0=Healthy, 1=Parkinson's)
    # OpenML: 22 features, Class       (Class: 1=Healthy, 2=Parkinson's)
    header = ["name"] + UCI_FEATURE_COLS + ["status"]
    rows = [header]
    for i, line in enumerate(data_lines):
        values = next(csv.reader([line]))
        features = values[:22]
        openml_class = values[22].strip()
        status = "1" if openml_class == "2" else "0"
        rows.append([f"subject_{i + 1:03d}"] + features + [status])

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()



# Public entry point
def download_data(filepath: str = DATA_PATH) -> None:
    """
    Fetch the Parkinson's dataset from UCI (primary) or OpenML (fallback).
    Saves to *filepath* as a standard CSV.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        print(f"Data already present at {filepath}. Skipping download.")
        return

    try:
        print("Attempting download from UCI ML Repository …")
        csv_text = _try_uci()
        with open(filepath, "w", newline="") as f:
            f.write(csv_text)
        print(f"Downloaded from UCI → {filepath}")
        return
    except Exception as exc:
        print(f"UCI unavailable ({exc}). Falling back to OpenML …")

    try:
        csv_text = _try_openml()
        with open(filepath, "w", newline="") as f:
            f.write(csv_text)
        print(f"Downloaded from OpenML → {filepath}")
    except Exception as exc:
        raise RuntimeError(
            f"Both UCI and OpenML downloads failed.\n"
            f"Please download parkinsons.data from:\n"
            f"  {UCI_URL}\n"
            f"and save it as {filepath}."
        ) from exc


if __name__ == "__main__":
    download_data()
