"""YAML configuration: streams, their ports and network simulator parameters."""

import ipaddress
from collections import namedtuple
from typing import Any, Optional, Tuple

import yaml


class ConfigError(ValueError):
    pass


# loss is in percent
Stream = namedtuple("Stream", "local_port remote_host remote_port loss delay_ms jitter_ms")
Stream.__new__.__defaults__ = (0.0, 0.0, 0.0)

# shared_source_port: send all streams to their remotes from one UDP port. roc-recv assigns
# repair (FEC) packets to a session by the sender's source address, so it needs this.
Config = namedtuple("Config", "streams bind seed shared_source_port")
Config.__new__.__defaults__ = ("127.0.0.1", None, True)


def addr_key(addr: tuple) -> Tuple[str, int]:
    """Normalised (ip, port) of a socket address, usable as a dict key."""
    return str(ipaddress.ip_address(addr[0].partition("%")[0])), addr[1]


def _port(text: str, what: str) -> int:
    if not text.isdecimal() or not 1 <= int(text) <= 65535:
        raise ConfigError("invalid {} {!r}".format(what, text))
    return int(text)


def parse_ports(spec: Any) -> Tuple[int, str, int]:
    """Parse ssh-like `local_port:remote_ip:remote_port` (IPv6 remote as `[::1]`)."""
    if not isinstance(spec, str):
        raise ConfigError("ports must be a string, got {!r}".format(spec))
    head, sep, rest = spec.strip().partition(":")
    host, sep2, remote_port = rest.rpartition(":")
    if not sep or not sep2 or not host:
        raise ConfigError("ports {!r} must look like local_port:remote_ip:remote_port".format(spec))
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        ipaddress.ip_address(host)
    except ValueError:
        raise ConfigError("invalid remote ip {!r} in {!r}".format(host, spec)) from None
    return _port(head, "local port"), host, _port(remote_port, "remote port")


def _number(value: Any, name: str, lo: float, hi: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{} must be a number, got {!r}".format(name, value))
    if value < lo or (hi is not None and value > hi):
        raise ConfigError(
            "{} must be in [{}, {}], got {}".format(name, lo, hi if hi is not None else "inf", value)
        )
    return float(value)


def _check_keys(obj: Any, allowed: set, where: str) -> dict:
    if not isinstance(obj, dict):
        raise ConfigError("{} must be a mapping".format(where))
    unknown = set(obj) - allowed
    if unknown:
        raise ConfigError(
            "unknown key(s) in {}: {}".format(where, ", ".join(sorted(map(str, unknown))))
        )
    return obj


def parse_config(data: Any) -> Config:
    data = _check_keys(data, {"bind", "seed", "shared_source_port", "streams"}, "config")
    streams_raw = data.get("streams")
    if not isinstance(streams_raw, list) or not streams_raw:
        raise ConfigError("streams must be a non-empty list")

    bind = str(data.get("bind", "127.0.0.1"))
    try:
        ipaddress.ip_address(bind)
    except ValueError:
        raise ConfigError("invalid bind address {!r}".format(bind)) from None
    seed = data.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigError("seed must be an integer, got {!r}".format(seed))
    shared = data.get("shared_source_port", True)
    if not isinstance(shared, bool):
        raise ConfigError("shared_source_port must be true or false, got {!r}".format(shared))

    streams = []
    seen = set()
    remotes = set()
    for i, raw in enumerate(streams_raw):
        where = "streams[{}]".format(i)
        raw = _check_keys(raw, {"ports", "loss", "delay_ms", "jitter_ms"}, where)
        if "ports" not in raw:
            raise ConfigError("{}: ports is required".format(where))
        try:
            local, host, remote = parse_ports(raw["ports"])
            stream = Stream(
                local,
                host,
                remote,
                _number(raw.get("loss", 0), "loss", 0, 100),
                _number(raw.get("delay_ms", 0), "delay_ms", 0),
                _number(raw.get("jitter_ms", 0), "jitter_ms", 0),
            )
        except ConfigError as e:
            raise ConfigError("{}: {}".format(where, e)) from None
        if local in seen:
            raise ConfigError("{}: local port {} used by more than one stream".format(where, local))
        seen.add(local)
        remote_key = addr_key((stream.remote_host, stream.remote_port))
        if shared and remote_key in remotes:
            raise ConfigError(
                "{}: remote {}:{} used by more than one stream, replies could not be told apart "
                "(set shared_source_port: false to allow it)".format(where, host, remote)
            )
        remotes.add(remote_key)
        streams.append(stream)
    return Config(streams, bind, seed, shared)


def load_config(path) -> Config:
    try:
        with open(str(path)) as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError("cannot read {}: {}".format(path, e)) from None
    return parse_config(data)
