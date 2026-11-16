import asyncio
import os
import random
import socket
import struct
import subprocess
import sys
import time
from typing import Optional

import pytest

from udp_proxy import main as udp_proxy_main
from udp_proxy.config import Config, ConfigError, Stream, config_from_streams, parse_config, parse_ports
from udp_proxy.netsim import NetSim
from udp_proxy.relay import run


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_parse_ports():
    assert parse_ports("5004:127.0.0.1:6004") == (5004, "127.0.0.1", 6004)
    assert parse_ports("5004:[::1]:6004") == (5004, "::1", 6004)


@pytest.mark.parametrize("spec", ["5004", "5004:6004", "x:127.0.0.1:1", "1:example.com:2", "1:127.0.0.1:70000", "0:127.0.0.1:1", 5, ""])
def test_parse_ports_invalid(spec):
    with pytest.raises(ConfigError):
        parse_ports(spec)


def test_parse_config():
    c = parse_config(
        {
            "seed": 3,
            "streams": [
                {"ports": "1000:127.0.0.1:2000", "loss": 5, "delay_ms": 50, "jitter_ms": 10},
                {"ports": "1001:127.0.0.1:2001"},
            ],
        }
    )
    assert c.bind == "127.0.0.1" and c.seed == 3
    assert c.streams[0] == Stream(1000, "127.0.0.1", 2000, 5, 50, 10)
    assert c.streams[1] == Stream(1001, "127.0.0.1", 2001, 0, 0, 0)


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"streams": []},
        {"streams": [{}]},
        {"streams": [{"ports": "1:127.0.0.1:2", "loss": 101}]},
        {"streams": [{"ports": "1:127.0.0.1:2", "delay_ms": -1}]},
        {"streams": [{"ports": "1:127.0.0.1:2", "jitter_ms": "x"}]},
        {"streams": [{"ports": "1:127.0.0.1:2", "typo": 1}]},
        {"streams": [{"ports": "1:127.0.0.1:2"}, {"ports": "1:127.0.0.1:3"}]},
        {"bind": "nope", "streams": [{"ports": "1:127.0.0.1:2"}]},
        {"extra": 1, "streams": [{"ports": "1:127.0.0.1:2"}]},
        {"shared_source_port": "yes", "streams": [{"ports": "1:127.0.0.1:2"}]},
        {"streams": [{"ports": "1:127.0.0.1:2"}, {"ports": "3:127.0.0.1:2"}]},
    ],
)
def test_parse_config_invalid(data):
    with pytest.raises(ConfigError):
        parse_config(data)


def test_shared_remote_allowed_with_separate_source_ports():
    c = parse_config(
        {
            "shared_source_port": False,
            "streams": [{"ports": "1:127.0.0.1:2"}, {"ports": "3:127.0.0.1:2"}],
        }
    )
    assert not c.shared_source_port and len(c.streams) == 2


def test_netsim_loss_rate():
    sim = NetSim(Stream(1, "127.0.0.1", 2, loss=20), random.Random(1))
    n = 20000
    dropped = sum(sim.schedule(0.0) is None for _ in range(n))
    assert abs(dropped / n - 0.2) < 0.01


def test_netsim_jitter_bounds_and_order():
    sim = NetSim(Stream(1, "127.0.0.1", 2, delay_ms=50, jitter_ms=40), random.Random(1))
    now, last, delays = 0.0, float("-inf"), []
    for _ in range(5000):
        now += 0.001
        t = sim.schedule(now)
        assert t > last
        last = t
        delays.append(t - now)
    assert min(delays) >= 0.0
    assert len({round(d, 3) for d in delays}) > 20  # jitter is really applied


async def _exchange(streams, shared_source_port=True):
    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    task = loop.create_task(run(Config(streams, "127.0.0.1", 42, shared_source_port), stop))
    await asyncio.sleep(0.1)
    return stop, task


async def _recv_all(sock: socket.socket, duration: float):
    loop = asyncio.get_event_loop()
    out = []
    end = loop.time() + duration
    while end - loop.time() > 0:
        try:
            data = await asyncio.wait_for(loop.sock_recv(sock, 2048), end - loop.time())
        except asyncio.TimeoutError:
            break
        out.append((data, time.monotonic()))
    return out


async def _recvfrom(sock: socket.socket, timeout: float):
    # loop.sock_recvfrom() exists only since Python 3.11
    loop = asyncio.get_event_loop()
    fut = asyncio.Future(loop=loop)

    def readable() -> None:
        loop.remove_reader(sock)
        if not fut.done():
            fut.set_result(sock.recvfrom(2048))

    loop.add_reader(sock, readable)
    try:
        return await asyncio.wait_for(fut, timeout)
    finally:
        loop.remove_reader(sock)


def _sock(port: Optional[int] = None) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port or 0))
    s.setblocking(False)
    return s


def run_async(main):
    """asyncio.run() replacement: it appeared only in Python 3.7."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(main())
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_passthrough_and_reverse_path():
    async def main():
        lp, rp = free_port(), free_port()
        remote, client = _sock(rp), _sock()
        stop, task = await _exchange([Stream(lp, "127.0.0.1", rp)])
        loop = asyncio.get_event_loop()
        client.sendto(b"hello", ("127.0.0.1", lp))
        data, proxy_addr = await _recvfrom(remote, 1)
        assert data == b"hello"
        remote.sendto(b"reply", proxy_addr)
        assert await asyncio.wait_for(loop.sock_recv(client, 2048), 1) == b"reply"
        stop.set()
        await task

    run_async(main)


def test_loss_delay_jitter_on_wire_and_streams_independent():
    async def main():
        lp1, rp1, lp2, rp2 = (free_port() for _ in range(4))
        r1, r2, client = _sock(rp1), _sock(rp2), _sock()
        stop, task = await _exchange(
            [
                Stream(lp1, "127.0.0.1", rp1, loss=30, delay_ms=60, jitter_ms=40),
                Stream(lp2, "127.0.0.1", rp2),
            ]
        )
        n = 200
        sent = {}
        for i in range(n):
            sent[i] = time.monotonic()
            pkt = struct.pack("!I", i)
            client.sendto(pkt, ("127.0.0.1", lp1))
            client.sendto(pkt, ("127.0.0.1", lp2))
            await asyncio.sleep(0.002)

        got1, got2 = await asyncio.gather(_recv_all(r1, 0.4), _recv_all(r2, 0.4))
        stop.set()
        await task

        assert len(got2) == n  # clean stream untouched
        assert 0.6 * n < len(got1) < 0.8 * n
        seq = [struct.unpack("!I", d)[0] for d, _ in got1]
        assert seq == sorted(seq)  # no reordering
        lat = [t - sent[i] for i, (_, t) in zip(seq, got1)]
        assert min(lat) > 0.015  # delay - jitter, with scheduling slack
        assert max(lat) - min(lat) > 0.03  # jitter is visible

    run_async(main)


def _two_streams_one_client(shared: bool):
    async def main():
        lp1, rp1, lp2, rp2 = (free_port() for _ in range(4))
        r1, r2, c1, c2 = _sock(rp1), _sock(rp2), _sock(), _sock()
        stop, task = await _exchange(
            [Stream(lp1, "127.0.0.1", rp1), Stream(lp2, "127.0.0.1", rp2)], shared
        )
        loop = asyncio.get_event_loop()
        c1.sendto(b"one", ("127.0.0.1", lp1))
        c2.sendto(b"two", ("127.0.0.1", lp2))
        d1, a1 = await _recvfrom(r1, 1)
        d2, a2 = await _recvfrom(r2, 1)
        assert (d1, d2) == (b"one", b"two")

        # each reply must return to the client of its own stream
        r2.sendto(b"from2", a2)
        r1.sendto(b"from1", a1)
        assert await asyncio.wait_for(loop.sock_recv(c1, 2048), 1) == b"from1"
        assert await asyncio.wait_for(loop.sock_recv(c2, 2048), 1) == b"from2"
        stop.set()
        await task
        return a1, a2

    return run_async(main)


def test_shared_source_port_for_all_streams():
    a1, a2 = _two_streams_one_client(shared=True)
    assert a1 == a2


def test_separate_source_port_per_stream():
    a1, a2 = _two_streams_one_client(shared=False)
    assert a1 != a2


def test_stream_specs():
    c = config_from_streams(
        ["1000:127.0.0.1:2000,loss=5", "1001:127.0.0.1:2001,delay_ms=50,jitter_ms=10"], seed=3
    )
    assert c.seed == 3 and c.bind == "127.0.0.1"
    assert c.streams[0] == Stream(1000, "127.0.0.1", 2000, 5, 0, 0)
    assert c.streams[1] == Stream(1001, "127.0.0.1", 2001, 0, 50, 10)
    assert config_from_streams(["1:127.0.0.1:2"], bind="127.0.0.2").bind == "127.0.0.2"


@pytest.mark.parametrize(
    "spec",
    ["1:127.0.0.1:2,", "1:127.0.0.1:2,loss", "1:127.0.0.1:2,loss=x", "1:127.0.0.1:2,speed=1",
     "1:127.0.0.1:2,loss=1,loss=2", "1:127.0.0.1:2,loss=101", "1:127.0.0.1"],
)
def test_stream_specs_invalid(spec):
    with pytest.raises(ConfigError):
        config_from_streams([spec])


def test_cli_ready_file_and_forwarding(tmp_path):
    lp, rp = free_port(), free_port()
    remote, client = _sock(rp), _sock()
    ready = tmp_path / "ready"
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    proc = subprocess.Popen(
        [sys.executable, "-m", "udp_proxy", "--ready-file", str(ready),
         "--stream", "{}:127.0.0.1:{},loss=0".format(lp, rp)],
        env=env,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists() and ready.read_text().strip() == str(proc.pid)

        async def main():
            client.sendto(b"ping", ("127.0.0.1", lp))
            return await _recvfrom(remote, 2)

        assert run_async(main)[0] == b"ping"
    finally:
        proc.terminate()
        assert proc.wait(10) == 0
        remote.close()
        client.close()


def test_cli_requires_config_or_stream():
    with pytest.raises(SystemExit) as e:
        udp_proxy_main([])
    assert e.value.code == 2
