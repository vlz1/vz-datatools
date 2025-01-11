from datasets import Dataset, DatasetDict
from typing import Type
from common import *

class Operation:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, dataset: Dataset | DatasetDict, **kwargs) -> Dataset | DatasetDict:
        raise NotImplementedError

    def trace(self, object):
        log_trace(f"{self.__class__.__name__}: {object}")

    @staticmethod
    def create(operation_name: str) -> "Operation":
        op_class = Operation.registered_operations.get(operation_name)
        if op_class is None:
            raise ValueError(f"Invalid operation '{operation_name}'")
        return op_class(operation_name)

    @classmethod
    def register(cls, name):
        Operation.registered_operations[name] = cls
        log_trace(f"Registered operation '{name}'")

    registered_operations: dict[str, Type["Operation"]] = { }

class RemapOperation(Operation):
    """Rename or remove columns from the data source.

    Keyword arguments:
    columns -- Dictionary mapping source column names to destination column names.
               Columns mapped to a blank ("") destination name are removed from the dataset.
    remove_others -- If true, all columns that aren't explicitly mapped will be removed. (Default: false)
    """

    def __call__(self, dataset: Dataset | DatasetDict, **kwargs) -> Dataset | DatasetDict:
        mappings: dict[str, str] = kwargs.get("columns", { })
        mappings = {k: v for k, v in mappings.items() if v != ""}
        remove_others: bool = kwargs.get("remove_others", False)
        
        columns_before = dataset.column_names
        removed = list(filter(lambda x: RemapOperation.filter_removals(x, mappings, remove_others), dataset.column_names))
        if len(removed) > 0:
            dataset = dataset.remove_columns(removed)
        
        dataset = dataset.rename_columns(mappings)
        self.trace(f"{columns_before} => {dataset.column_names}")
        return dataset

    @staticmethod
    def filter_removals(src_column: str, column_mappings: dict[str, str], remove_others: bool):
        dst = column_mappings.get(src_column)
        if dst is None and remove_others:
            return True
        elif dst != "":
            return False
        return True

RemapOperation.register("remap")