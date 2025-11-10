#!/usr/bin/env python3
"""A command line tool for extracting text and images from PDF and
output it to plain text, html, xml or tags.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from pdf2zh_next.config import ConfigManager
from pdf2zh_next.high_level import do_translate_file_async

__version__ = "2.6.4"

logger = logging.getLogger(__name__)


def find_all_files_in_directory(directory_path):
    """
    Recursively search all PDF files in the given directory and return their paths as a list.

    :param directory_path: str, the path to the directory to search
    :return: list of PDF file paths
    """
    directory_path = Path(directory_path)
    # Check if the provided path is a directory
    if not directory_path.is_dir():
        raise ValueError(f"The provided path '{directory_path}' is not a directory.")

    file_paths = []

    # Walk through the directory recursively
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Check if the file is a PDF
            if file.lower().endswith(".pdf"):
                # Append the full file path to the list
                file_paths.append(Path(root) / file)

    return file_paths


async def main() -> int:
    """Main async entry for running translation tasks.

    Assumes logging has already been configured by the caller.
    """

    settings = ConfigManager().initialize_config()

    # If config requests debug, elevate root logger
    if settings.basic.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # disable noisy third-party logs
    for name in ("httpx", "openai", "httpcore", "http11"):
        logging.getLogger(name).setLevel("CRITICAL")
        logging.getLogger(name).propagate = False

    for v in logging.Logger.manager.loggerDict.values():
        if not isinstance(v, logging.Logger):
            continue
        if (
            v.name.startswith("pdfminer")
            or v.name.startswith("peewee")
            or v.name.startswith("httpx")
            or "http11" in v.name
            or "openai" in v.name
            or "pdfminer" in v.name
        ):
            v.disabled = True
            v.propagate = False

    logger.debug(f"settings: {settings}")

    if settings.basic.version:
        print(f"pdf2zh-next version: {__version__}")
        return 0

    if settings.basic.gui:
        from pdf2zh_next.gui import setup_gui

        setup_gui(
            auth_file=settings.gui_settings.auth_file,
            welcome_page=settings.gui_settings.welcome_page,
            server_port=settings.gui_settings.server_port,
        )
        return 0

    # If config indicates HTTP API, caller should start server instead
    if settings.basic.http_api:
        logger.error("Configuration requests HTTP API mode; caller should start server")
        return -1

    assert len(settings.basic.input_files) >= 1, "At least one input file is required"
    await do_translate_file_async(settings, ignore_error=True)
    return 0


def configure_logging(debug: bool = False) -> None:
    """Configure root logging using rich if available.

    This should be called once at startup before other modules log.
    Logs are written to both console and a rotating file.
    """
    from logging.handlers import RotatingFileHandler
    
    # Create logs directory if it doesn't exist
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Setup file handler with rotation (max 10MB, keep 5 backup files)
    log_file = logs_dir / "pdf2zh_next.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    handlers: list[logging.Handler] = [file_handler]

    try:
        from rich.logging import RichHandler

        rich_handler = RichHandler(
            show_time=True,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=debug,
        )
        handlers.append(rich_handler)
    except Exception:
        # Fallback to standard console handler if rich is not available
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(console_handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Logging to file: {log_file}")
    
    # Configure uvicorn loggers to use the same handlers
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = logging.getLogger().handlers
        uvicorn_logger.propagate = False


def run():
    """Synchronous entry point. Parses args, configures logging and dispatches.

    Behavior:
    - If `--http-api` is passed, start the server directly.
    - Otherwise run the async `main()` flow which loads full config.
    """

    parser = argparse.ArgumentParser(prog="pdf2zh-next")
    parser.add_argument("--http-api", action="store_true", help="Start HTTP API server")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging early
    configure_logging(debug=args.debug)

    # If CLI flag requests http_api, start server directly without loading full config
    if args.http_api:
        from pdf2zh_next.http_api import run_server

        logger.info("Starting HTTP API server...")
        run_server(host="0.0.0.0", port=11008, reload=args.debug)
        return

    # Otherwise run the async main flow (which loads and validates config)
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    run()
