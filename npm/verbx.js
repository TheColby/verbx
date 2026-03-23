#!/usr/bin/env node
"use strict";

const { spawnSync } = require("node:child_process");
const path = require("node:path");

const rootDir = path.resolve(__dirname, "..");
const srcDir = path.join(rootDir, "src");
const args = process.argv.slice(2);

function pythonCommand() {
  return process.env.PYTHON || "python3";
}

function runCli() {
  const env = { ...process.env };
  env.PYTHONPATH = env.PYTHONPATH ? `${srcDir}:${env.PYTHONPATH}` : srcDir;
  return spawnSync(
    pythonCommand(),
    ["-m", "verbx.cli", ...args],
    {
      stdio: "inherit",
      env,
    },
  );
}

function bootstrapPythonDeps() {
  return spawnSync(
    pythonCommand(),
    ["-m", "pip", "install", "--user", rootDir],
    { stdio: "inherit" },
  );
}

let result = runCli();
if (result.status === 0) {
  process.exit(0);
}

const stderrText = String(result.stderr || "");
if (!stderrText.includes("ModuleNotFoundError")) {
  process.exit(result.status || 1);
}

console.error(
  "[verbx npm launcher] Missing Python dependencies detected; installing with pip --user...",
);
const bootstrap = bootstrapPythonDeps();
if (bootstrap.status !== 0) {
  process.exit(bootstrap.status || 1);
}

result = runCli();
process.exit(result.status || 0);
