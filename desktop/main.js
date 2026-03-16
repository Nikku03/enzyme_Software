const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const path = require("path");

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8000;

let serverProcess = null;
let mainWindow = null;
let activePort = null;

function getProjectRoot() {
  if (app.isPackaged) {
    return process.resourcesPath;
  }
  return path.join(__dirname, "..");
}

function buildPythonEnv(rootDir) {
  const pythonPath = path.join(rootDir, "src");
  const existing = process.env.PYTHONPATH;
  const combined = existing ? `${pythonPath}${path.delimiter}${existing}` : pythonPath;
  return {
    ...process.env,
    PYTHONPATH: combined,
  };
}

function commandCandidates() {
  const candidates = [];
  if (process.env.BOND_BREAK_PYTHON) {
    candidates.push(process.env.BOND_BREAK_PYTHON);
  }
  if (process.env.PYTHON) {
    candidates.push(process.env.PYTHON);
  }
  candidates.push("python");
  candidates.push("python3");
  return [...new Set(candidates.filter(Boolean))];
}

function waitForServer(url, timeoutMs, proc) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    let settled = false;

    const cleanup = () => {
      if (proc) {
        proc.removeListener("exit", onExit);
      }
    };

    const onExit = () => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(new Error("Python server exited before becoming ready."));
    };

    const attempt = () => {
      if (settled) {
        return;
      }
      const req = http.get(url, (res) => {
        res.resume();
        settled = true;
        cleanup();
        resolve();
      });
      req.setTimeout(1000, () => {
        req.destroy();
      });
      req.on("error", () => {
        if (Date.now() - start > timeoutMs) {
          settled = true;
          cleanup();
          reject(new Error("Timed out waiting for Python server."));
          return;
        }
        setTimeout(attempt, 300);
      });
    };

    if (proc) {
      proc.once("exit", onExit);
    }
    attempt();
  });
}

async function startServer(rootDir, host, port) {
  const args = ["-m", "enzyme_software.web_app", "--host", host, "--port", String(port)];
  const env = buildPythonEnv(rootDir);
  const candidates = commandCandidates();
  let lastError = null;

  for (const command of candidates) {
    const proc = spawn(command, args, {
      cwd: rootDir,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stderr = "";
    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    proc.on("error", (err) => {
      lastError = err;
    });

    try {
      await waitForServer(`http://${host}:${port}`, 8000, proc);
      serverProcess = proc;
      return;
    } catch (err) {
      proc.kill();
      lastError = new Error(stderr.trim() || err.message);
    }
  }

  throw lastError || new Error("Unable to start Python server.");
}

function isAddressInUse(err) {
  if (!err) {
    return false;
  }
  const message = err.message || String(err);
  return message.includes("Address already in use") || message.includes("EADDRINUSE");
}

function checkPortAvailable(host, port) {
  return new Promise((resolve) => {
    const tester = net
      .createServer()
      .once("error", () => resolve(false))
      .once("listening", () => {
        tester.close(() => resolve(true));
      });
    tester.listen(port, host);
  });
}

function findEphemeralPort(host) {
  return new Promise((resolve, reject) => {
    const tester = net.createServer();
    tester.once("error", reject);
    tester.listen(0, host, () => {
      const address = tester.address();
      const port = typeof address === "object" && address ? address.port : DEFAULT_PORT;
      tester.close(() => resolve(port));
    });
  });
}

async function selectPort(host, preferredPort) {
  if (Number.isFinite(preferredPort) && preferredPort > 0 && preferredPort < 65536) {
    const available = await checkPortAvailable(host, preferredPort);
    if (available) {
      return preferredPort;
    }
  }
  return findEphemeralPort(host);
}

async function startServerWithFallback(rootDir, host, preferredPort) {
  let port = await selectPort(host, preferredPort);
  try {
    await startServer(rootDir, host, port);
    return port;
  } catch (err) {
    if (isAddressInUse(err)) {
      port = await findEphemeralPort(host);
      await startServer(rootDir, host, port);
      return port;
    }
    throw err;
  }
}

function createWindow(host, port) {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 820,
    minWidth: 960,
    minHeight: 700,
    backgroundColor: "#f3efe6",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadURL(`http://${host}:${port}`);

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function stopServer() {
  if (serverProcess) {
    serverProcess.kill();
    serverProcess = null;
  }
}

app.whenReady().then(async () => {
  const host = process.env.BOND_BREAK_HOST || DEFAULT_HOST;
  const preferredPort = Number(process.env.BOND_BREAK_PORT || DEFAULT_PORT);
  const rootDir = getProjectRoot();

  try {
    activePort = await startServerWithFallback(rootDir, host, preferredPort);
    createWindow(host, activePort);
  } catch (err) {
    dialog.showErrorBox(
      "BondBreak startup failed",
      `${err.message}\n\nEnsure Python 3.9+ is installed and accessible. You can set BOND_BREAK_PYTHON to the Python executable path.`
    );
    app.quit();
  }
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0 && mainWindow === null) {
    const host = process.env.BOND_BREAK_HOST || DEFAULT_HOST;
    const port = activePort || Number(process.env.BOND_BREAK_PORT || DEFAULT_PORT);
    createWindow(host, port);
  }
});

app.on("before-quit", () => {
  stopServer();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
