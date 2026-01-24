#! /usr/bin/env python3
#
# CLI utilities for standalone main() functions
#
# Jan-2026, Pat Welch, pat@mousebrains.com

"""CLI utilities for reducing boilerplate in standalone module entry points."""

from __future__ import annotations

from typing import Callable


def create_standalone_main(
    add_arguments_func: Callable,
    run_func: Callable,
    description: str,
    epilog: str | None = None,
) -> Callable[[], None]:
    """
    Factory for creating standalone main() functions.

    Creates a main() function with standardized argument parsing, logging setup,
    and execution flow. Reduces boilerplate in modules that provide both
    subcommand and standalone CLI interfaces.

    Args:
        add_arguments_func: Function that adds module-specific arguments to parser.
                           Signature: (parser) -> None
        run_func: Function to execute with parsed arguments.
                 Signature: (args) -> None
        description: Description for ArgumentParser
        epilog: Optional epilog text for ArgumentParser

    Returns:
        A main() function suitable for use as module entry point.

    Example:
        >>> # In a module (e.g., deramp.py):
        >>> from wamos_tpw.cli_utils import create_standalone_main
        >>> main = create_standalone_main(
        ...     _add_arguments,
        ...     run,
        ...     "Apply deramp correction to radar data"
        ... )
        >>> if __name__ == "__main__":
        ...     main()
    """

    def main() -> None:
        from argparse import ArgumentParser

        from wamos_tpw.logging_config import add_logging_arguments, setup_logging

        parser = ArgumentParser(description=description, epilog=epilog)
        add_logging_arguments(parser)
        add_arguments_func(parser)
        args = parser.parse_args()
        setup_logging(args)
        run_func(args)

    return main
