import os.path as path
from multiprocessing import cpu_count
from datasets import Dataset, DatasetDict, load_dataset
from typing import Literal, Optional
from pydantic import BaseModel
from pydantic_core import from_json

type DataRecipeSourceType = Literal["source", "recipe"]

class DataRecipeOperationConfig(BaseModel):
    name: str
    args: Optional[dict] = { }

class DataRecipeSourceConfig(BaseModel):
    type: Optional[DataRecipeSourceType] = "source"
    probability: Optional[float] = 1.0
    operations: Optional[list[DataRecipeOperationConfig]] = [ ]

class DataRecipeConfig(BaseModel):
    sources: dict[str, DataRecipeSourceConfig] | list[str]
    final_operations: Optional[list[DataRecipeOperationConfig]] = [ ]
    test_split_ratio: float = 0.0

class DataRecipe:
    def __init__(self, json_path: str):
        self.name, extension = path.splitext(path.basename(json_path))
        self.sources: dict[str, DataRecipeSourceConfig] = { }
        self.built_dataset: Dataset | DatasetDict = None
        self.built_directory = ""
        self.built = False
        self.building = False
        self.modified = False
        self.references: list[str] = [ ]
        self.referenced_by: list[str] = [ ]

        with open(json_path, "rb") as f:
            self.config = DataRecipeConfig.model_validate_json(f.read().decode("utf-8"))
        self.modification_time = path.getmtime(json_path)

        if isinstance(self.config.sources, dict):
            for source_name, source_config in self.config.sources.items():
                self.sources[source_name] = source_config
                if source_config.type == "recipe":
                    self.references.append(source_name)
        else:
            for source_name in self.config.sources:
                self.sources[source_name] = DataRecipeSourceConfig()
    