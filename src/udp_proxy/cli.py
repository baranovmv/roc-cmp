"""Command line interface."""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Callable, List, Optional

from .config import Config, ConfigError, config_from_streams, load_config
from .relay import run


async def _serve(config: Config, on_ready: Callable[[], None]) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await run(config, stop, on_ready)


def _write_ready_file(path: Path) -> None:
    # written atomically so that a watcher never sees a half-written file
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("{}\n".format(os.getpid()))
    os.replace(str(tmp), str(path))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="udp-proxy", description="UDP proxy with loss/delay/jitter simulation per stream."
    )
    p.add_argument("config", type=Path, nargs="?", help="YAML file with the list of streams")
    p.add_argument(
        "--stream",
        action="append",
        default=[],
        metavar="SPEC",
        help="stream instead of a config file, repeatable: "
        "local_port:remote_ip:remote_port[,loss=PERCENT][,delay_ms=MS][,jitter_ms=MS]",
    )
    p.add_argument("--bind", metavar="IP", help="address to listen on (default: 127.0.0.1)")
    p.add_argument("--seed", type=int, help="seed of the loss/jitter generator, makes runs reproducible")
    p.add_argument(
        "--ready-file",
        type=Path,
        metavar="PATH",
        help="create this file (containing the pid) once all ports are bound",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = p.parse_args(argv)
    if bool(args.config) == bool(args.stream):
        p.error("give either a config file or at least one --stream")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    def on_ready() -> None:
        logging.getLogger("udp-proxy").info("ready")
        if args.ready_file:
            _write_ready_file(args.ready_file)

    try:
        if args.config:
            config = load_config(args.config)
            if args.bind is not None:
                config = config._replace(bind=args.bind)
            if args.seed is not None:
                config = config._replace(seed=args.seed)
        else:
            config = config_from_streams(args.stream, args.bind, args.seed)
        if args.ready_file and args.ready_file.exists():
            args.ready_file.unlink()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_serve(config, on_ready))
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
