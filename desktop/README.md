# BondBreak Desktop App

This Electron wrapper launches the local web UI inside a desktop window and starts the Python backend automatically.

## Requirements
- Node.js 18+
- Python 3.9+

## Run locally

```bash
npm install
npm run start
```

If your Python executable is not on PATH, set `BOND_BREAK_PYTHON` to the full path:

```bash
BOND_BREAK_PYTHON=/usr/bin/python3 npm run start
```

Optional overrides:
- `BOND_BREAK_HOST` (default: 127.0.0.1)
- `BOND_BREAK_PORT` (default: 8000)

## Build installers

```bash
npm run dist
```

Note: Packaging still expects Python to be available on the target machine. For a fully standalone bundle, we can ship a Python runtime or generate a backend binary and point `BOND_BREAK_PYTHON` to it.
