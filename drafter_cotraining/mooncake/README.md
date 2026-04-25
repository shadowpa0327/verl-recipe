# Mooncake IPv6-only Environment Support

This document describes the special handling required for Mooncake in IPv6-only environments.

## Background

Mooncake uses two C++ libraries from the coro_rpc project:
- **coro_rpc**: Used for the RPC server (master gRPC on port 50051)
- **coro_http**: Used for HTTP metadata server (port 8090) and metrics server (port 9003)

### The Problem

In IPv6-only environments (no IPv4 addresses available), `coro_http` has a DNS resolution bug:

```
ERROR [coro_http_server.hpp:725] bad address: 0.0.0.0 error: Host not found (authoritative)
```

The issue occurs because:
1. `coro_http` is hardcoded to bind to `0.0.0.0`
2. When binding, it calls `getaddrinfo("0.0.0.0", ...)`
3. In IPv6-only environments, there's no IPv4 stack, so DNS resolution fails
4. This causes both the metadata server and metrics server to fail

### Why coro_rpc Works

The RPC server uses `coro_rpc`, which respects the `--rpc_address` flag. When we set `--rpc_address=::`, it binds to the IPv6 any address and works correctly.

## Solution: P2PHANDSHAKE Mode

We use P2PHANDSHAKE mode to bypass the `coro_http` bug entirely:

| Component | Normal Mode | P2PHANDSHAKE Mode |
|-----------|-------------|-------------------|
| RPC Server | `--rpc_port=50051` | `--rpc_port=50051 --rpc_address=::` |
| Metadata Server | `--enable_http_metadata_server=true` | `--enable_http_metadata_server=false` |
| Metrics Server | Enabled by default | `--enable_metric_reporting=false` |
| Client metadata_server | `http://host:8090/metadata` | `P2PHANDSHAKE` |

### How It Works

In P2PHANDSHAKE mode:
1. The HTTP metadata server is disabled (avoiding `coro_http`)
2. The metrics server is disabled (avoiding `coro_http`)
3. The RPC server binds to `::` (IPv6 any address)
4. Clients use peer-to-peer handshaking instead of HTTP metadata lookup

## Usage

### Automatic Detection

The system automatically detects IPv6-only environments and enables P2PHANDSHAKE mode. No manual configuration is needed.

Detection logic (`_is_ipv6_only_environment()`):
```python
# Check if any IPv4 addresses exist
result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
for addr in result.stdout.strip().split():
    if "." in addr and ":" not in addr:  # IPv4 address
        return False
return True  # No IPv4 found
```

### Manual Configuration

If you need to manually configure P2PHANDSHAKE mode, set in your config:

```yaml
mooncake:
  master_server_address: "[::1]:50051"  # IPv6 with brackets
  metadata_server: "P2PHANDSHAKE"        # Use P2P handshaking
  local_hostname: "::1"
  protocol: tcp
```

### Master Startup

For IPv6-only environments, start the master with:

```bash
MC_USE_IPV6=1 mooncake_master \
  --rpc_port=50051 \
  --rpc_address=:: \
  --enable_http_metadata_server=false \
  --enable_metric_reporting=false
```

### Client Configuration

Clients need to set `MC_USE_IPV6=1` and use `metadata_server="P2PHANDSHAKE"`:

```python
import os
os.environ["MC_USE_IPV6"] = "1"

from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
store.setup(
    local_hostname="::1",
    metadata_server="P2PHANDSHAKE",
    master_server_addr="[::1]:50051",
    protocol="tcp",
    # ... other params
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MC_USE_IPV6=1` | Enable IPv6 mode in Mooncake's transfer engine |

## Testing

Run the test script to verify IPv6 support:

```bash
python recipe/drafter_cotraining/scripts/test_mooncake_store.py
```

Expected output in IPv6-only environment:
```
[IPv6] MC_USE_IPV6=1 enabled for client (P2PHANDSHAKE mode)
Launching mooncake_master (IPv6-only/P2PHANDSHAKE mode)
mooncake_master ready on ::1:50051 (P2PHANDSHAKE mode)
```

## Troubleshooting

### Error: "bad address: 0.0.0.0 error: Host not found"

This indicates `coro_http` is still trying to bind to IPv4. Ensure:
1. P2PHANDSHAKE mode is enabled (`metadata_server="P2PHANDSHAKE"`)
2. HTTP metadata server is disabled (`--enable_http_metadata_server=false`)
3. Metrics reporting is disabled (`--enable_metric_reporting=false`)

### Error: "not valid LAN address found"

Set `MC_USE_IPV6=1` before importing Mooncake:
```python
import os
os.environ["MC_USE_IPV6"] = "1"
# Now import mooncake
```

### Connection refused on localhost

In IPv6-only mode, use `::1` instead of `localhost` or `127.0.0.1`:
- Master: `--rpc_address=::`
- Client: `master_server_addr="[::1]:50051"`

## Files Modified

| File | Changes |
|------|---------|
| `mooncake/master.py` | IPv6 detection, P2PHANDSHAKE mode for master startup |
| `mooncake/store.py` | IPv6 environment variable setup for client |
| `mooncake/config.py` | IPv6 bracket notation handling |
| `draft_model_pretrain_trainer.py` | P2PHANDSHAKE port parsing |
| `scripts/test_mooncake_store.py` | Test script with P2PHANDSHAKE support |

## Upstream Issues

This workaround is needed due to upstream bugs in Mooncake's dependencies:
- `coro_http` DNS resolution fails in IPv6-only environments
- HTTP servers are hardcoded to bind to `0.0.0.0`

Once these are fixed in Mooncake or coro_rpc upstream, P2PHANDSHAKE mode may no longer be necessary for IPv6-only environments.

## References

- Mooncake: https://github.com/kvcache-ai/Mooncake
- coro_rpc: https://github.com/JoybeanAI/coro_rpc
