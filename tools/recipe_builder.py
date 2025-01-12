from dataclasses import dataclass
from datasets import Dataset, DatasetDict, interleave_datasets, concatenate_datasets
from data_source import *
from data_recipe import *
from operation import *
from time import time
from colorama import Fore
import numpy as np
import math

class RecipeBuilder:
    def __init__(self, sources_directory: str, recipes_directory: str, output_directory: str):
        self.sources_directory = sources_directory
        self.recipes_directory = recipes_directory
        self.output_directory = output_directory
        self.loaded_recipes = { }
        self.loaded_sources = { }

    def load_dependencies(self, recipe: DataRecipe) -> bool:
        """Recurse through a recipe's dependencies to figure out
        what was modified and needs to be rebuilt.

        Returns false if nothing needs to be done.
        """
        # TODO: Detect circular dependencies
        rebuild = False
        for referenced_name in recipe.references:
            referenced = self.get_recipe(referenced_name)
            if self.load_dependencies(referenced):
                rebuild = True
            
        try:
            modification_time = 0.0
            if recipe.config.test_split_ratio > 0:
                modification_time = path.getmtime(path.join(recipe.built_directory, "dataset_dict.json"))
            else:
                modification_time = path.getmtime(path.join(recipe.built_directory, "dataset_info.json"))
            
            if recipe.modification_time > modification_time:
                rebuild = True
        except FileNotFoundError:
            rebuild = True
        
        if not rebuild:
            log_ok(f"{recipe.built_directory} is up to date.")
            recipe.built = True
            if recipe.config.test_split_ratio > 0:
                recipe.built_dataset = DatasetDict.load_from_disk(recipe.built_directory)
            else:
                recipe.built_dataset = Dataset.load_from_disk(recipe.built_directory)
        recipe.modified = rebuild
        return rebuild

    def build(self, recipe_name: str) -> DataRecipe:
        recipe = self.get_recipe(recipe_name)
        if recipe.built:
            return recipe

        if not self.load_dependencies(recipe):
            return recipe

        recipe.building = True
        start_time = time()
        log_info(f"Building recipe '{recipe_name}'...")

        sources: list[str] = [ ]
        source_datasets: list[Dataset] = [ ]
        source_probabilities: list[float] = [ ]
        total_rows = 0.0
        use_probabilities = False
        for source_name, source_config in recipe.sources.items():
            if source_config.probability != 1: 
                use_probabilities = True
            # Resolve referenced source/recipe into a dataset
            dataset: Dataset
            if source_config.type == "source":
                source = self.get_source(source_name)
                dataset = source.dataset
                if isinstance(dataset, DatasetDict):
                    dataset = concatenate_datasets(dataset.values())
            elif source_config.type == "recipe":
                if source_name == recipe_name:
                    raise RecursionError(f"Recipe '{recipe_name}' tried to use itself as a source")
                dependency = self.build(source_name)
                dataset = dependency.built_dataset
                if isinstance(dataset, DatasetDict):
                    dataset = concatenate_datasets(dataset.values())
            dataset = self.apply_operations(dataset, source_config.operations)
            sources.append(source_name)
            source_datasets.append(dataset)
            source_probabilities.append(source_config.probability)
            total_rows += len(dataset)
        
        # Interleave all the sources based on their probabilities
        seed = 42
        if use_probabilities:
            adjusted_probabilities = np.array(source_probabilities) / np.array(source_probabilities).sum()
            recipe.built_dataset = interleave_datasets(source_datasets, probabilities=adjusted_probabilities, seed=seed)

            log_info(f"Approximate distribution of {recipe.name} after interleaving:")
            interleaved_examples = len(recipe.built_dataset)
            for i, source_name in enumerate(sources):
                approx_examples = math.floor(adjusted_probabilities[i] * interleaved_examples)
                approx_percent = (approx_examples / interleaved_examples) * 100
                log_info(f"\t{source_name:<40} {approx_examples} examples ({approx_percent:.2f}%)")
        else:
            recipe.built_dataset = interleave_datasets(source_datasets, seed=seed)

        # Run final operations
        recipe.built_dataset = self.apply_operations(recipe.built_dataset, recipe.config.final_operations)

        # Save it
        if recipe.config.test_split_ratio > 0:
            splits = recipe.built_dataset.train_test_split(test_size=recipe.config.test_split_ratio)
            splits.save_to_disk(recipe.built_directory)
        else:
            recipe.built_dataset.save_to_disk(recipe.built_directory)
        recipe.built = True
        recipe.building = False
        log_ok(f"Built '{recipe_name}' in {time() - start_time} seconds ({len(recipe.built_dataset)} rows)")
        return recipe

    def apply_operations(self, dataset: Dataset | DatasetDict, operations: list[Operation]) -> Dataset | DatasetDict:
        for op_config in operations:
            op = Operation.create(op_config.name)
            dataset = op(dataset, **op_config.args)
        return dataset
    
    def get_source(self, name: str) -> DataSource:
        source = RecipeBuilder.source_cache.get(name)
        if source is not None: return source
        RecipeBuilder.source_cache[name] = source = DataSource(path.join(self.sources_directory, name + ".json"))
        return source

    def get_recipe(self, name: str) -> DataRecipe:
        recipe = RecipeBuilder.recipe_cache.get(name)
        if recipe is not None: return recipe
        RecipeBuilder.recipe_cache[name] = recipe = DataRecipe(path.join(self.recipes_directory, name + ".json"))
        recipe.built_directory = path.join(self.output_directory, recipe.name)
        return recipe
    


    recipe_cache: dict[str, DataRecipe] = { }
    source_cache: dict[str, DataSource] = { }