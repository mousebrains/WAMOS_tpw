#!/usr/bin/env python3
"""
WAMOS radar data processing CLI.

Master command that provides access to all WAMOS processing tools via subcommands.
"""

import argparse
import sys


def main() -> None:
    """Main entry point for the wamos CLI."""
    parser = argparse.ArgumentParser(
        prog='wamos',
        description='WAMOS marine radar data processing tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wamos combine "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR --plot
  wamos process "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR --plot
  wamos view "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR --plot-intensity
  wamos list "2022-04-05 14:00" "2022-04-05 15:00" /path/to/POLAR
  wamos parse /path/to/file.pol --show-header
"""
    )

    subparsers = parser.add_subparsers(
        dest='command',
        title='commands',
        description='Available processing commands',
        metavar='COMMAND'
    )

    # Register subcommands from each module
    from wamos_tpw import combine, processed, files, bearing
    from wamos_tpw import filenames, polarfile, timestamp, config
    from wamos_tpw import deramp, destreak

    combine.add_subparser(subparsers)      # wamos combine
    processed.add_subparser(subparsers)    # wamos process
    files.add_subparser(subparsers)        # wamos view
    bearing.add_subparser(subparsers)      # wamos bearing
    filenames.add_subparser(subparsers)    # wamos list
    polarfile.add_subparser(subparsers)    # wamos parse
    timestamp.add_subparser(subparsers)    # wamos timestamp
    config.add_subparser(subparsers)       # wamos config
    deramp.add_subparser(subparsers)       # wamos deramp
    destreak.add_subparser(subparsers)     # wamos destreak

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Call the command's function
    args.func(args)


if __name__ == "__main__":
    main()
