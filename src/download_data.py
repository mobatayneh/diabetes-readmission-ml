"""Download the original UCI archive without Kaggle credentials.

Usage: python -m src.download_data
The raw patient records are ignored by Git. Existing files are not overwritten.
"""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from src.paths import RAW_DATA_DIR

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
)
DATASET_DOI = "https://doi.org/10.24432/C5230J"


def extract_dataset(archive: bytes, destination: Path) -> dict:
    """Extract only the two expected CSVs; do not trust archive member paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    with ZipFile(io.BytesIO(archive)) as zf:
        for filename in ("diabetic_data.csv", "IDS_mapping.csv"):
            matches = [i for i in zf.infolist() if Path(i.filename).name == filename]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {filename} in the UCI archive")
            data = zf.read(matches[0])
            output = destination / filename
            if output.exists() and output.read_bytes() != data:
                raise FileExistsError(f"Refusing to overwrite a different existing file: {output}")
            if not output.exists():
                output.write_bytes(data)
            records[filename] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args(argv)
    with urlopen(DATASET_URL, timeout=60) as response:
        archive = response.read()
    records = extract_dataset(archive, args.destination)
    manifest = {
        "source": DATASET_URL,
        "doi": DATASET_DOI,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "files": records,
    }
    (args.destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Dataset ready in {args.destination}")
    for name, info in records.items():
        print(f"  {name}: {info['bytes']:,} bytes, SHA-256 {info['sha256']}")


if __name__ == "__main__":
    main()
