#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");
const path = require("node:path");

const rootDir = path.resolve(__dirname, "..");
const srcDir = path.join(rootDir, "src");
const args = process.argv.slice(2);

function pythonCandidates() {
  const candidates = [];
  if (process.env.PYTHON && process.env.PYTHON.trim()) {
    candidates.push(process.env.PYTHON.trim());
  }
  candidates.push("python3", "python");
  if (process.platform === "win32") {
    candidates.push("py");
  }
  return [...new Set(candidates)];
}

function printCaptured(result) {
  if (result && result.stdout) {
    process.stdout.write(String(result.stdout));
  }
  if (result && result.stderr) {
    process.stderr.write(String(result.stderr));
  }
}

function runCli(pythonExec) {
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH ? `${srcDir}:${env.PYTHONPATH}` : srcDir;
  return spawnSync(
    pythonExec,
    ["-m", "verbx.cli", ...args],
    {
      encoding: "utf8",
      stdio: "pipe",
      env,
    },
  );
}

function bootstrapPythonDeps(pythonExec) {
  return spawnSync(
    pythonExec,
    ["-m", "pip", "install", "--user", rootDir],
    { encoding: "utf8", stdio: "pipe" },
  );
}

let result = null;
let selectedPython = null;
for (const candidate of pythonCandidates()) {
  const attempt = runCli(candidate);
  if (attempt.error && attempt.error.code === "ENOENT") {
    continue;
  }
  result = attempt;
  selectedPython = candidate;
  break;
}
if (!result || !selectedPython) {
  console.error(
    "[verbx npm launcher] Python not found. Install Python 3.11+ and retry, or set $PYTHON.",
  );
  process.exit(1);
}

const stderrText = String(result.stderr || "");
const missingModule = stderrText.includes("ModuleNotFoundError");
if (!missingModule) {
  printCaptured(result);
  process.exit(typeof result.status === "number" ? result.status : 1);
}

printCaptured(result);
console.error(
  `[verbx npm launcher] Missing Python dependencies; bootstrapping with ${selectedPython} -m pip install --user ...`,
);
const bootstrap = bootstrapPythonDeps(selectedPython);
printCaptured(bootstrap);
if (bootstrap.status !== 0) {
  console.error(
    "[verbx npm launcher] Bootstrap failed. Ensure pip is available and that your user site bin path is on PATH.",
  );
  process.exit(bootstrap.status || 1);
}

result = runCli(selectedPython);
printCaptured(result);
process.exit(result.status || 0);
