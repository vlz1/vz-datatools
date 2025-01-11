import sys
import os
import os.path as path
from argparse import ArgumentParser
from pathlib import Path

main_directory = path.dirname(path.realpath(__file__))
sys.path.append(path.join(main_directory, "tools"))
sources_directory = path.join(main_directory, "sources")
recipes_directory = path.join(main_directory, "recipes")
output_directory = path.join(main_directory, "built")

if not path.isdir(sources_directory):
    os.mkdir(sources_directory)
if not path.isdir(recipes_directory):
    os.mkdir(recipes_directory)
if not path.isdir(output_directory):
    os.mkdir(output_directory)

usage = "usage: vz-datatools [-h] {build,list-recipes,list-operations}"
if len(sys.argv) < 2:
    print(usage)
    exit()
elif sys.argv[1] == "-h" or sys.argv[1] == "--help":
    print(usage)
    exit()
action = sys.argv[1]

def list_recipes():
    print("Available recipes:")
    paths = Path(recipes_directory).glob("*.json")
    for recipe_json in paths:
        if not recipe_json.is_file():
            continue
        print(f"\t{recipe_json.name[:-5]}")

match action:
    case "build":
        build_parser = ArgumentParser(
            prog="vz-datatools build",
            description="Simple framework for mixing together HF datasets"
        )
        build_parser.add_argument("recipe", help="Data recipe to build.")
        args = build_parser.parse_args(sys.argv[2:])
        from tools import *
        builder = RecipeBuilder(sources_directory, recipes_directory, output_directory)
        try:
            builder.build(args.recipe)
        except FileNotFoundError:
            log_failed(f"Invalid recipe '{args.recipe}'")
            list_recipes()
        exit()

    case "list-recipes":
        list_recipes()
        exit()

    case "list-operations":
        from tools import *
        print("Available operations:")
        for name in Operation.registered_operations.keys():
            print(f"\t{name}")
        exit()



