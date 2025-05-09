from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    img_source_URL: str
    label_source_URL: str
    local_img_data_file: Path
    local_label_data_file: Path
    unzip_dir: Path
