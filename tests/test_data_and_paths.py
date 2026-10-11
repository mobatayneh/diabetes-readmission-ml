import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from src import data_preprocessing
from src.download_data import extract_dataset
from src.paths import PROJECT_ROOT, PROCESSED_DATA_DIR, RAW_DATA_DIR


class DataTests(unittest.TestCase):
    def test_preprocessing_writes_to_canonical_directory(self):
        self.assertEqual(data_preprocessing.PROCESSED_DATA_DIR, PROCESSED_DATA_DIR)
        self.assertEqual(PROCESSED_DATA_DIR, PROJECT_ROOT / "data/diabetes130/processed")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(data_preprocessing, "PROCESSED_DATA_DIR", Path(directory)):
                data_preprocessing.save_processed_data(pd.DataFrame({"readmitted_binary": [0, 1]}), "test.csv")
            self.assertEqual(pd.read_csv(Path(directory) / "test.csv").readmitted_binary.tolist(), [0, 1])

    def test_mapping_file_contains_three_tables(self):
        mappings = data_preprocessing.read_multiple_tables_csv(RAW_DATA_DIR / "IDS_mapping.csv")
        self.assertEqual(len(mappings), 3)
        for frame, name in zip(mappings, ["admission_type_id", "discharge_disposition_id", "admission_source_id"]):
            self.assertIn(name, frame)
            self.assertIn("description", frame)

    def test_downloader_selects_expected_members_and_preserves_existing_data(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("dataset_diabetes/diabetic_data.csv", "encounter_id,readmitted\n1,NO\n")
            archive.writestr("dataset_diabetes/IDS_mapping.csv", "id,description\n1,Example\n")
            archive.writestr("../../unexpected.txt", "not extracted")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "raw"
            records = extract_dataset(buffer.getvalue(), destination)
            self.assertEqual(set(records), {"diabetic_data.csv", "IDS_mapping.csv"})
            self.assertEqual(set(p.name for p in destination.iterdir()), set(records))
            self.assertFalse((Path(directory) / "unexpected.txt").exists())
            (destination / "diabetic_data.csv").write_text("existing local data")
            with self.assertRaises(FileExistsError):
                extract_dataset(buffer.getvalue(), destination)
            self.assertEqual((destination / "diabetic_data.csv").read_text(), "existing local data")


if __name__ == "__main__":
    unittest.main()
