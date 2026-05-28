"""Rich utils to print config tree."""

from collections.abc import Sequence
from pathlib import Path

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        print_order: Determines in what order config components are printed. Default is ``("data", "model",
        "callbacks", "logger", "trainer", "paths", "extras")``.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
        save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue: list[str] = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field_key in cfg:
        field_str = str(field_key)
        if field_str not in queue:
            queue.append(field_str)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        branch_content: str
        if isinstance(config_group, DictConfig):
            yaml_output = OmegaConf.to_yaml(config_group, resolve=resolve)
            branch_content = str(yaml_output) if yaml_output is not None else ""
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    console = Console(width=1000)
    console.print(tree)

    # save config tree to file
    if save_to_file:
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Use a wider console for file output to avoid path truncation
        with open(output_dir / "config_tree.log", "w") as file:
            console = Console(file=file, width=1000)
            console.print(tree)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    Args:
        cfg: A DictConfig composed by Hydra.
        save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:  # type: ignore
            raise ValueError("Specify tags before launching a multirun!")  # noqa

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags_input: str = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags: list[str] = [t.strip() for t in tags_input.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "tags.log", "w") as file:
            rich.print(cfg.tags, file=file)
