"""
Main command-line interface for FiberWatch.

This module provides the main CLI entry point with subcommands
for different FiberWatch operations.
"""

import argparse
import sys
from pathlib import Path

from fiberwatch.config import get_default_config, load_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FiberWatch - OTDR Event Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s analyze data/test.txt --output results/
    %(prog)s visualize data/test.txt --baseline data/baseline.txt
    %(prog)s web
    %(prog)s --version
            """,
    )

    parser.add_argument("--version", action="version", version="FiberWatch 0.1.0")

    parser.add_argument("--config", type=Path, help="Path to configuration file")

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze OTDR data file",
        description="Run OTDR event detection on a data file",
    )
    _add_analyze_args(analyze_parser)

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Visualize OTDR data with analysis",
        description="Create visualization plots for OTDR analysis",
    )
    _add_visualize_args(visualize_parser)

    # Web command
    web_parser = subparsers.add_parser(
        "web",
        help="Launch web interface",
        description="Start the Streamlit web interface",
    )
    _add_web_args(web_parser)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load configuration
    if args.config:
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config = get_default_config()

    if args.debug:
        config.debug = True

    # Execute command
    try:
        if args.command == "analyze":
            _run_analyze(args, config)
        elif args.command == "visualize":
            _run_visualize(args, config)
        elif args.command == "web":
            _run_web(args, config)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if config.debug:
            raise
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _add_analyze_args(parser):
    """Add arguments for analyze command."""
    parser.add_argument("--input_file", type=Path, help="Input OTDR data file")

    parser.add_argument("--baseline", type=Path, help="Baseline reference file")

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="output",
        help="Output directory (default: output)",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=20.0,
        help="Fiber length in km (default: 20.0)",
    )

    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format for results (default: csv)",
    )


def _add_visualize_args(parser):
    """Add arguments for visualize command."""
    parser.add_argument("--input_file", type=Path, help="Input OTDR data file")

    parser.add_argument("--baseline", type=Path, help="Baseline reference file")

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="output",
        help="Output directory (default: output)",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=20.0,
        help="Fiber length in km (default: 20.0)",
    )

    parser.add_argument(
        "--no-save", action="store_true", help="Don't save plots to files",
    )


def _add_web_args(parser):
    """Add arguments for web command."""
    parser.add_argument(
        "--host", default="localhost", help="Host to bind to (default: localhost)"
    )

    parser.add_argument(
        "--port", type=int, default=8501, help="Port to bind to (default: 8501)"
    )


def _run_analyze(args, config):
    """Run analyze command."""
    # Import here to avoid circular imports
    from .analyze import run_analysis

    run_analysis(
        input_file=args.input_file,
        baseline_file=args.baseline,
        output_dir=args.output,
        distance_km=args.distance,
        output_format=args.format,
        config=config,
    )


def _run_visualize(args, config):
    """Run visualize command."""
    # Import here to avoid circular imports
    from .visualize import run_visualization

    run_visualization(
        input_file=args.input_file,
        baseline_file=args.baseline,
        output_dir=args.output,
        distance_km=args.distance,
        save_plots=not args.no_save,
        config=config,
    )


def _run_web(args, config):
    """Run web command."""
    import subprocess
    import sys
    from pathlib import Path

    # Get path to the web app module
    web_app_path = Path(__file__).parent.parent / "web" / "app.py"

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(web_app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]

    print(f"Starting web interface at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    # Run streamlit
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
