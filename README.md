# vz-datatools

TODO: Documentation

## Sources

To import something and include it in a dataset, you have to write a source: a short JSON file that describes where to read/download the data from. 

Internally, each source is loaded by the HuggingFace Datasets library and any existing splits are automatically merged. Train/test splits are only present during output, and can be configured by a [recipe](#recipes).

### Source Schema

- source_type (string, required):
Indicates the type of data source. Possible values are:
    - "hf_hub"  - Uses source_path as a HF hub location (ex: stanfordnlp/imdb).
    - "hf_disk" - Loads a dataset from source_path that was saved via Dataset.save_to_disk().
    - "parquet" - Loads parquet files from source_path or source_files.
    - "csv"     - Loads CSV files from source_path or source_files.

- source_path (string, ignored if source_files is provided):
Specifies the directory containing the data source. Defaults to an empty string if not provided.

- source_files (array of strings, ignored if source_path is provided):
A list of file names associated with the data source. Defaults to an empty list if not provided.

- description (string, optional):
A brief description of the data source. Defaults to an empty string if not provided.

### Examples

sources/imdb.json
```json
{
    "source_type": "hf_hub",
    "source_path": "stanfordnlp/imdb"
}
```

sources/directory_examples.json
```json
{
    "source_type": "parquet",
    "source_path": "./example_dataset"
}
```

sources/scattered_examples.json
```json
{
    "source_type": "parquet",
    "source_files": [
        "./example_dataset0/data.parquet",
        "./example_dataset1/data.parquet"
    ]
}
```

## Recipes

Recipes are used to process sources and mix them together.

### Recipe Schema

- sources (array of strings or map of strings to recipe sources, required):
List or map of recipe sources to include in the recipe.

- test_split_ratio (float, optional):
Ratio describing the size of the test split. If 0 or not provided, no test split is created.

- final_operations (array of operations, optional):
Sequence of operations to perform before saving the dataset.

### Recipe Source Schema

### Examples

recipes/examples.json
```json
{
    "sources": [
        "directory_examples",
        "scattered_examples"
    ]
}
```

recipes/examples_tokenized.json
```json
{
    "sources": {
        "examples": {
            "type": "recipe"
        }
    },
    "final_operations": [
        {
            "name": "tokenize_text",
            "args": {
                "model": "FacebookAI/roberta-base",
                "text_column": "text",
            }
        }
    ],
    "test_split_ratio": 0.15
}
```

## Operations

