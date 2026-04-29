# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import atexit
import ctypes
import os
import random
import shutil
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

import ray

# RayActor replaced with ray.remote
# env var passthrough removed (torchspec-specific)
import logging
logger = logging.getLogger(__name__)


def _wait_for_tcp_ready(
    host: str,
    port: int,
    *,
    total_timeout_s: float,
    per_attempt_timeout_s: float = 0.5,
    sleep_s: float = 0.1,
) -> None:
    """Wait until a TCP connect to host:port succeeds.

    This is used to avoid flaky startup races where a process is alive but
    hasn't bound its ports yet.
    """
    deadline = time.monotonic() + max(total_timeout_s, 0.0)
    last_exc: Exception | None = None

    # Use appropriate socket family for IPv6
    sock_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    while time.monotonic() <= deadline:
        try:
            sock = socket.socket(sock_family, socket.SOCK_STREAM)
            sock.settimeout(max(0.01, per_attempt_timeout_s))
            try:
                sock.connect((host, port))
                return
            finally:
                sock.close()
        except OSError as exc:
            last_exc = exc
            time.sleep(sleep_s)

    if last_exc is None:
        raise TimeoutError(f"Timed out waiting for TCP {host}:{port} to become ready")
    raise TimeoutError(
        f"Timed out waiting for TCP {host}:{port} to become ready (last error: {last_exc})"
    ) from last_exc


def _is_ipv6_only_environment() -> bool:
    """Detect if the environment is IPv6-only (no IPv4 addresses available).

    IPv6-only environments require special handling for Mooncake because:
    1. Mooncake's coro_http library has a DNS resolution bug - it cannot resolve
       "0.0.0.0" or "127.0.0.1" when no IPv4 stack is available.
    2. The HTTP metadata server and metrics server both use coro_http and are
       hardcoded to bind to "0.0.0.0", causing them to fail in IPv6-only mode.

    Solution: Use P2PHANDSHAKE mode which:
    - Disables the HTTP metadata server (avoids coro_http bug)
    - Disables the metrics server (avoids coro_http bug)
    - Binds RPC server to "::" (IPv6 any address)
    - Uses peer-to-peer handshaking instead of HTTP metadata server

    Returns:
        True if no IPv4 addresses are found, False otherwise.
    """
    try:
        # Check if we have any IPv4 addresses
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        for addr in result.stdout.strip().split():
            if "." in addr and ":" not in addr:  # IPv4 address
                return False
        # If we get here, no IPv4 addresses found
        return True
    except Exception:
        return False


def resolve_mooncake_master_bin() -> str:
    """Resolve the path to the mooncake_master binary."""
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")

    which_result = shutil.which("mooncake_master")
    if which_result:
        return which_result

    home = os.path.expanduser("~")
    return os.path.join(home, "build/mooncake-store/src/mooncake_master")


def _subprocess_preexec():
    """Pre-exec setup for the mooncake master subprocess.

    - os.setpgrp(): Create a new process group so that os.killpg() can kill
      the wrapper script AND the real binary it spawns (grandchild).
    - PR_SET_PDEATHSIG: Kernel sends SIGTERM when the parent (Ray worker) dies,
      preventing orphans on crashes.
    """
    os.setpgrp()
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


@ray.remote(num_cpus=0)
class MooncakeMaster:
    """Ray actor that wraps the mooncake master subprocess.

    Provides automatic lifecycle management — when the actor is killed or garbage
    collected, the subprocess is terminated. Logs are streamed through Ray's
    logging pipeline instead of written to files.
    """

    def __init__(self):
        self._process = None
        self._info = {}

    def start(
        self,
        port: int,
        http_port: int,
        http_host: str = "0.0.0.0",
        kv_lease_ttl_s: float = 5.0,
    ) -> dict:
        """Launch the mooncake master subprocess.

        Args:
            port: gRPC port for mooncake master.
            http_port: HTTP metadata server port (ignored in IPv6-only mode).
            http_host: HTTP metadata server host (ignored in IPv6-only mode).
            kv_lease_ttl_s: Default KV object lease TTL in seconds.

        Returns:
            Dict with "master_addr", "metadata_port", and "metadata_server".

        Raises:
            FileNotFoundError: If binary is not found.
            RuntimeError: If process fails to start.
        """
        mooncake_bin = resolve_mooncake_master_bin()
        if not os.path.exists(mooncake_bin):
            raise FileNotFoundError(f"mooncake_master binary not found at {mooncake_bin}")

        # Detect IPv6-only environment and use P2PHANDSHAKE mode
        #
        # Background: Mooncake uses two C++ libraries:
        # - coro_rpc: for the RPC server (master gRPC)
        # - coro_http: for HTTP metadata server and metrics server
        #
        # Problem: In IPv6-only environments, coro_http has a DNS resolution bug:
        # - It tries to resolve "0.0.0.0" to bind the server
        # - getaddrinfo("0.0.0.0") fails with "Host not found" when no IPv4 stack
        # - This causes both metadata server (port 8090) and metrics server (port 9003) to fail
        #
        # Solution: P2PHANDSHAKE mode
        # - Disable HTTP metadata server (--enable_http_metadata_server=false)
        # - Disable metrics server (--enable_metric_reporting=false)
        # - Bind RPC to IPv6 any address (--rpc_address=::)
        # - Client uses metadata_server="P2PHANDSHAKE" for peer-to-peer handshaking
        #
        # This avoids coro_http entirely, using only coro_rpc which respects --rpc_address.
        ipv6_only = _is_ipv6_only_environment()

        if ipv6_only:
            # P2PHANDSHAKE mode: bypass coro_http bugs in IPv6-only environments
            # - Disable HTTP metadata server (coro_http has IPv6 DNS resolution bug)
            # - Bind RPC to :: (IPv6 any address)
            # - Client will use metadata_server="P2PHANDSHAKE"
            cmd = [
                mooncake_bin,
                f"--rpc_port={port}",
                "--rpc_address=::",
                "--enable_http_metadata_server=false",
                "--enable_metric_reporting=false",
                f"--default_kv_lease_ttl={int(kv_lease_ttl_s * 1000)}",
            ]
            env = os.environ.copy()
            env["MC_USE_IPV6"] = "1"
            logger.info("Starting mooncake master in IPv6-only mode (P2PHANDSHAKE, port=%s)", port)
        else:
            # The mooncake_master binary always tries to bind ``--metrics_port``
            # (default 9003), even with ``--enable_metric_reporting=false``;
            # if that port is held by a stale master, the new process exits 1.
            # Bind ``metrics_port`` to ``rpc_port + 1`` so the address is
            # always co-derived with the user-chosen rpc port and never
            # collides with another verl run.
            metrics_port = int(port) + 1
            cmd = [
                mooncake_bin,
                f"--rpc_port={port}",
                f"--http_metadata_server_port={http_port}",
                f"--http_metadata_server_host={http_host}",
                "--enable_http_metadata_server=true",
                "--enable_metric_reporting=false",
                f"--metrics_port={metrics_port}",
                f"--default_kv_lease_ttl={int(kv_lease_ttl_s * 1000)}",
            ]
            env = os.environ.copy()
            logger.info(
                "Starting mooncake master (grpc_port=%s, http_port=%s, metrics_port=%s)",
                port, http_port, metrics_port,
            )

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_subprocess_preexec,
            env=env,
        )

        # Stream stdout/stderr through logger in background daemon threads
        self._start_log_thread(self._process.stdout, "stdout")
        self._start_log_thread(self._process.stderr, "stderr")

        # Give the subprocess a moment to initialize.
        time.sleep(0.5)

        if self._process.poll() is not None:
            raise RuntimeError(
                f"mooncake master failed to start (exit code: {self._process.returncode})"
            )

        host = ray.util.get_node_ip_address()

        # Avoid flaky races: process is running but ports are not bound yet.
        # Try both the node IP and loopback because the binary may bind to one
        # but not the other depending on its defaults.
        # In IPv6-only environments, also try ::1 (IPv6 loopback).
        start_timeout_s = float(os.getenv("MOONCAKE_STARTUP_TIMEOUT_S", "30"))
        candidate_hosts = [host, "127.0.0.1", "localhost"]
        if ipv6_only:
            # In IPv6-only mode, RPC binds to :: (reachable via ::1)
            candidate_hosts.insert(0, "::1")
        for candidate_host in candidate_hosts:
            try:
                _wait_for_tcp_ready(
                    candidate_host,
                    port,
                    total_timeout_s=start_timeout_s,
                )
                host = candidate_host  # Use the reachable host
                break
            except TimeoutError:
                continue
        else:
            raise RuntimeError(
                f"mooncake master process started but gRPC port {port} did not become reachable within {start_timeout_s}s"
            )

        # Metadata server readiness is best-effort; it can lag behind gRPC.
        # Skip in IPv6-only mode since we use P2PHANDSHAKE (no HTTP server).
        if not ipv6_only:
            for candidate_host in candidate_hosts:
                try:
                    _wait_for_tcp_ready(
                        candidate_host,
                        http_port,
                        total_timeout_s=start_timeout_s,
                    )
                    break
                except TimeoutError:
                    continue

        # Format address with brackets for IPv6
        if ":" in host and not host.startswith("["):
            master_addr_formatted = f"[{host}]:{port}"
        else:
            master_addr_formatted = f"{host}:{port}"

        # In IPv6-only mode, use P2PHANDSHAKE instead of HTTP metadata server
        if ipv6_only:
            metadata_server = "P2PHANDSHAKE"
        else:
            if ":" in host:
                metadata_server = f"http://[{host}]:{http_port}/metadata"
            else:
                metadata_server = f"http://{host}:{http_port}/metadata"

        self._info = {
            "master_addr": master_addr_formatted,
            "metadata_port": http_port,
            "metadata_server": metadata_server,
            "ipv6_only": ipv6_only,
        }

        logger.info(f"mooncake master started (PID: {self._process.pid}, metadata={metadata_server})")
        return self._info

    def health_check(self) -> bool:
        """Check if the subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_info(self) -> dict:
        """Return the master address and metadata port."""
        return self._info

    def _start_log_thread(self, stream, name: str) -> None:
        """Start a daemon thread that reads lines from stream and logs them."""

        def _reader():
            for line in stream:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.rstrip("\n")
                if line:
                    logger.debug(f"[mooncake_master {name}] {line}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def shutdown(self):
        """Gracefully terminate the subprocess and its entire process group."""
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except Exception:
                    pass
            self._process = None

    def __del__(self):
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass


def check_mooncake_master_available(
    master_server_address: str,
    metadata_server: str,
    timeout: float = 5.0,
) -> None:
    """Verify mooncake master services are reachable.

    Probes the gRPC endpoint via TCP connect and the HTTP metadata endpoint
    so actors fail fast with a clear error before expensive model loading.

    Args:
        master_server_address: gRPC address, e.g. "10.1.2.3:50051" or "[::1]:50051".
        metadata_server: HTTP metadata URL, e.g. "http://10.1.2.3:8090/metadata",
                        or "P2PHANDSHAKE" for IPv6-only mode.
        timeout: Total time budget for each probe in seconds.

    Raises:
        RuntimeError: If either service is unreachable.
    """
    # gRPC port check (TCP connect)
    # Handle IPv6 addresses: [::1]:port or ::1:port
    grpc_host: str
    grpc_port: int

    if master_server_address.startswith("["):
        # IPv6 with brackets: [::1]:50051
        bracket_end = master_server_address.find("]")
        if bracket_end == -1:
            raise RuntimeError(f"Invalid IPv6 address format: {master_server_address!r}")
        grpc_host = master_server_address[1:bracket_end]
        if bracket_end + 1 < len(master_server_address) and master_server_address[bracket_end + 1] == ":":
            grpc_port = int(master_server_address[bracket_end + 2:])
        else:
            grpc_port = 50051  # default port
    elif ":" in master_server_address and master_server_address.count(":") >= 2:
        # IPv6 without brackets: ::1:50051 - last segment after final : is port
        # But IPv6 addresses have colons, so we need to be careful
        # If it starts with ::, it's IPv6
        if master_server_address.startswith("::") or master_server_address.count(":") >= 3:
            # Assume last :port is the port specification
            # e.g., ::1:50051 -> host=::1, port=50051
            last_colon = master_server_address.rfind(":")
            potential_port = master_server_address[last_colon + 1:]
            if potential_port.isdigit():
                grpc_host = master_server_address[:last_colon]
                grpc_port = int(potential_port)
            else:
                grpc_host = master_server_address
                grpc_port = 50051
        else:
            # IPv4:port
            grpc_host, grpc_port_str = master_server_address.rsplit(":", 1)
            grpc_port = int(grpc_port_str)
    else:
        # IPv4:port or just port
        try:
            grpc_host, grpc_port_str = master_server_address.rsplit(":", 1)
            grpc_port = int(grpc_port_str)
        except ValueError as exc:
            raise RuntimeError(f"Invalid master_server_address {master_server_address!r}") from exc

    grpc_deadline = time.monotonic() + max(float(timeout), 0.0)
    grpc_last_exc: OSError | None = None
    while time.monotonic() <= grpc_deadline:
        try:
            # Use appropriate socket family for IPv6
            sock_family = socket.AF_INET6 if ":" in grpc_host else socket.AF_INET
            sock = socket.socket(sock_family, socket.SOCK_STREAM)
            sock.settimeout(min(0.5, float(timeout)))
            try:
                sock.connect((grpc_host, grpc_port))
                grpc_last_exc = None
                break
            finally:
                sock.close()
        except OSError as exc:
            grpc_last_exc = exc
            time.sleep(0.1)
    if grpc_last_exc is not None:
        raise RuntimeError(
            f"Mooncake master gRPC unreachable at {master_server_address}: {grpc_last_exc}"
        ) from grpc_last_exc

    # Skip HTTP check for P2PHANDSHAKE mode (IPv6-only)
    # In P2PHANDSHAKE mode, the HTTP metadata server is disabled and clients
    # use peer-to-peer handshaking instead. Only the gRPC port needs to be checked.
    if metadata_server == "P2PHANDSHAKE":
        logger.info(
            "Mooncake master reachable (P2PHANDSHAKE mode): %s",
            master_server_address,
        )
        return

    # HTTP metadata server check (TCP connect to parsed host:port)
    parsed = urlparse(metadata_server)
    http_host = parsed.hostname
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if http_host is None:
        raise RuntimeError(f"Cannot parse host from metadata_server URL: {metadata_server!r}")

    http_deadline = time.monotonic() + max(float(timeout), 0.0)
    http_last_exc: OSError | None = None
    while time.monotonic() <= http_deadline:
        try:
            sock_family = socket.AF_INET6 if (http_host and ":" in http_host) else socket.AF_INET
            sock = socket.socket(sock_family, socket.SOCK_STREAM)
            sock.settimeout(min(0.5, float(timeout)))
            try:
                sock.connect((http_host, http_port))
                http_last_exc = None
                break
            finally:
                sock.close()
        except OSError as exc:
            http_last_exc = exc
            time.sleep(0.1)
    if http_last_exc is not None:
        raise RuntimeError(
            f"Mooncake metadata server unreachable at {metadata_server}: {http_last_exc}"
        ) from http_last_exc

    logger.info(
        "Mooncake services reachable: master=%s, metadata=%s",
        master_server_address,
        metadata_server,
    )


def launch_mooncake_master(args):
    """Launch the mooncake master as a Ray actor.

    Auto-resolves master_server_address and metadata_port if not configured.
    When master_server_address specifies a host IP, pins the actor to that node so the
    mooncake master process starts on the intended machine.
    Writes resolved values back to args for downstream code.

    Args:
        args: Arguments namespace with mooncake_master_server_address, mooncake_metadata_port, etc.

    Returns:
        The MooncakeMasterActor handle, or None if binary not found.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    master_addr = getattr(args, "mooncake_master_server_address", None)
    scheduling_strategy = None

    if master_addr is None:
        host = ray.util.get_node_ip_address()
        port = random.randint(51000, 52000)
        master_addr = f"{host}:{port}"
        args.mooncake_master_server_address = master_addr
        logger.info(f"Auto-resolved mooncake master_server_address: {master_addr}")
    else:
        try:
            host, port_str = master_addr.rsplit(":", 1)
            port = int(port_str)
        except Exception:
            host = master_addr
            port = getattr(args, "mooncake_master_port", 50051)

        # If the user config uses localhost, pin to the driver's node IP so Ray
        # doesn't treat it as an arbitrary hostname.
        if host in ("localhost", "127.0.0.1", "::1"):
            host = ray.util.get_node_ip_address()

    # Pin actor to the user-specified node so the master starts on the right machine.
    # Pin actor to the specified host node
    nodes = ray.nodes()
    node_id = None
    for node in nodes:
        if node.get("NodeManagerAddress") == host and node["Alive"]:
            node_id = node["NodeID"]
            break
    scheduling_strategy = NodeAffinitySchedulingStrategy(
        node_id=node_id or "", soft=True
    ) if node_id else None

    http_port = getattr(args, "mooncake_metadata_port", None) or getattr(
        args, "mooncake_http_port", None
    )
    if http_port is None:
        http_port = random.randint(8100, 9100)
        args.mooncake_metadata_port = http_port
        logger.info(f"Auto-resolved mooncake metadata_port: {http_port}")
    http_host = getattr(args, "mooncake_http_host", "0.0.0.0")

    # Check binary existence before creating the actor
    mooncake_bin = resolve_mooncake_master_bin()
    if not os.path.exists(mooncake_bin):
        logger.warning(f"Binary not found at {mooncake_bin}, skipping launch")
        return None

    actor_options = {"name": "mooncake_master"}
    if scheduling_strategy is not None:
        actor_options["scheduling_strategy"] = scheduling_strategy
    actor = MooncakeMaster.options(**actor_options).remote()

    kv_lease_ttl_s = getattr(args, "mooncake_kv_lease_ttl_s", 5.0)

    try:
        info = ray.get(actor.start.remote(port, http_port, http_host, kv_lease_ttl_s))
        # Write back resolved values (actor may have updated host from node IP)
        args.mooncake_master_server_address = info["master_addr"]
        args.mooncake_metadata_port = info["metadata_port"]
        # Use metadata_server from info (P2PHANDSHAKE for IPv6-only, or HTTP URL)
        args.mooncake_metadata_server = info["metadata_server"]
        logger.info(f"mooncake master actor started: {info}")
    except Exception as e:
        logger.error(f"Failed to launch mooncake master actor: {e}")
        return None

    args._mooncake_master_actor = actor

    def _cleanup():
        try:
            ray.get(actor.shutdown.remote(), timeout=10)
        except Exception:
            pass

    atexit.register(_cleanup)

    return actor
