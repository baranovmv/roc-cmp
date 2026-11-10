"""asyncio sockets: relays forwarding each stream through the network simulator."""

import asyncio
import logging
import random
import socket
from typing import Callable, Optional, Tuple

from .config import Config, Stream, addr_key
from .netsim import NetSim

log = logging.getLogger("udp-proxy")


class _Endpoint(asyncio.DatagramProtocol):
    def __init__(self, on_packet: Callable[[bytes, tuple], None]):
        self._on_packet = on_packet
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self._on_packet(data, addr)

    def error_received(self, exc):
        # e.g. ICMP port unreachable when nothing listens on the other side yet
        log.debug("socket error: %s", exc)


class Upstream:
    """Unconnected socket towards the remotes. Replies are told apart by their source address."""

    def __init__(self, ipv6: bool):
        self._ipv6 = ipv6
        self._handlers = {}
        self._endpoint = None

    async def start(self) -> None:
        loop = asyncio.get_event_loop()
        family, wildcard = (socket.AF_INET6, "::") if self._ipv6 else (socket.AF_INET, "0.0.0.0")
        _, self._endpoint = await loop.create_datagram_endpoint(
            lambda: _Endpoint(self._on_packet), local_addr=(wildcard, 0), family=family
        )

    def register(self, remote: Tuple[str, int], handler: Callable[[bytes], None]) -> None:
        self._handlers[addr_key(remote)] = handler

    def send(self, data: bytes, remote: Tuple[str, int]) -> None:
        if self._endpoint and self._endpoint.transport and not self._endpoint.transport.is_closing():
            self._endpoint.transport.sendto(data, remote)

    def close(self) -> None:
        if self._endpoint and self._endpoint.transport:
            self._endpoint.transport.close()

    def _on_packet(self, data: bytes, addr: tuple) -> None:
        handler = self._handlers.get(addr_key(addr))
        if handler:
            handler(data)
        else:
            log.debug("dropping packet from unknown remote %s:%s", addr[0], addr[1])


class Relay:
    """Forwards one stream: clients <-> local socket <-> proxy <-> upstream socket <-> remote."""

    def __init__(self, stream: Stream, bind: str, rng: random.Random, upstream: Upstream):
        self.stream = stream
        self.bind = bind
        self.upstream = upstream
        self._remote = (stream.remote_host, stream.remote_port)
        self._up = NetSim(stream, rng)
        self._down = NetSim(stream, rng)
        self._client = None
        self._local = None
        self._loop = None

    async def start(self) -> None:
        self._loop = asyncio.get_event_loop()
        self.upstream.register(self._remote, self._from_remote)
        _, self._local = await self._loop.create_datagram_endpoint(
            lambda: _Endpoint(self._from_client), local_addr=(self.bind, self.stream.local_port)
        )

    def close(self) -> None:
        if self._local and self._local.transport:
            self._local.transport.close()

    def _from_client(self, data: bytes, addr: tuple) -> None:
        self._client = addr
        self._forward(self._up, lambda: self.upstream.send(data, self._remote))

    def _from_remote(self, data: bytes) -> None:
        client = self._client
        if client is not None:
            self._forward(self._down, lambda: self._send_local(data, client))

    def _send_local(self, data: bytes, client: tuple) -> None:
        if self._local and self._local.transport and not self._local.transport.is_closing():
            self._local.transport.sendto(data, client)

    def _forward(self, sim: NetSim, send: Callable[[], None]) -> None:
        when = sim.schedule(self._loop.time())
        if when is None:
            return
        if when <= self._loop.time():
            send()
        else:
            self._loop.call_at(when, send)


async def run(config: Config, stop: Optional[asyncio.Event] = None) -> None:
    """Serve until `stop` is set (or forever)."""
    rng = random.Random(config.seed)
    shared = {}
    upstreams = []
    relays = []
    for s in config.streams:
        ipv6 = ":" in s.remote_host
        upstream = shared.get(ipv6) if config.shared_source_port else None
        if upstream is None:
            upstream = Upstream(ipv6)
            upstreams.append(upstream)
            shared[ipv6] = upstream
        relays.append(Relay(s, config.bind, rng, upstream))
    stop = stop or asyncio.Event()
    try:
        for u in upstreams:
            await u.start()
        for r in relays:
            await r.start()
            s = r.stream
            log.info(
                "%s:%d -> %s:%d loss=%g%% delay=%gms jitter=%gms",
                config.bind, s.local_port, s.remote_host, s.remote_port,
                s.loss, s.delay_ms, s.jitter_ms,
            )
        await stop.wait()
    finally:
        for r in relays:
            r.close()
        for u in upstreams:
            u.close()
        await asyncio.sleep(0)  # let the transports finish closing
