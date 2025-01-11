import os.path as path
from multiprocessing import cpu_count
from datasets import Dataset, DatasetDict, load_dataset
from typing import Literal, Optional
from pydantic import BaseModel, model_validator
from pydantic_core import from_json

type DataSourceType = Literal["hf_hub", "hf_disk", "parquet", "csv"]

class DataSourceConfig(BaseModel):
    source_type: DataSourceType
    source_path: Optional[str] = ""
    source_files: Optional[list[str]] = []
    description: Optional[str] = ""

    @model_validator(mode="after")
    def has_path_or_files(self):
        if self.source_path == "" and len(self.source_files) == 0:
            raise ValueError("Must have either source_path or source_files.")
        return self

class DataSource:
    def __init__(self, json_path: str):
        self.name, extension = path.splitext(path.basename(json_path))
        with open(json_path, "rb") as f:
            self.config = DataSourceConfig.model_validate_json(f.read().decode("utf-8"))
        self.source_path = self.config.source_path

        if self.config.source_type != "hf_hub":
            if self.source_path.startswith("./") | self.source_path.startswith("../"):
                self.source_path = path.join(path.dirname(json_path), self.source_path)
            if not path.isdir(self.source_path):
                raise FileNotFoundError(f"\"{self.source_path}\" is not a valid directory.")
        
        # Load the actual dataset
        num_proc = min(cpu_count(), 8)
        match self.config.source_type:
            case "hf_hub":
                self.dataset = load_dataset(self.source_path, num_proc=num_proc)
            case "hf_disk":
                self.dataset = Dataset.load_from_disk(self.source_path)
            case "parquet":
                if len(self.config.source_files) > 0:
                    self.dataset = load_dataset("parquet", 
                                            name=self.name,
                                            data_files=self.config.source_files, 
                                            num_proc=num_proc
                                            )
                else:
                    self.dataset = load_dataset("parquet", 
                                            name=self.name,
                                            data_dir=self.source_path, 
                                            num_proc=num_proc
                                            )
            case "csv":
                if len(self.config.source_files) > 0:
                    raise NotImplementedError()
                self.dataset = load_dataset("csv",
                                            name=self.name,
                                            data_dir=self.source_path,
                                            num_proc=num_proc)