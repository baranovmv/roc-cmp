"""Command line interface."""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config, ConfigError, load_config
from .relay import run


async def _serve(config: Config) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await run(config, stop)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="udp-proxy", description="UDP proxy with loss/delay/jitter simulation per stream."
    )
    p.add_argument("config", type=Path, help="YAML file with the list of streams")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = load_config(args.config)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_serve(config))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except ConfigError as e:
        print("udp-proxy: {}".format(e), file=sys.stderr)
        return 2
    except OSError as e:
        print("udp-proxy: {}".format(e), file=sys.stderr)
        return 1
    return 0
