import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    png: {
      type: "string",
      short: "p",
    },
    members: {
      type: "string",
      short: "m",
      default: "members.json",
    },
    style: {
      type: "string",
      short: "s",
      default: "small",
    },
    python: {
      type: "string",
      default: "python",
    },
  },
  strict: true,
});

if (!values.png) {
  console.error("Error: --png <path> is required");
  process.exit(1);
}

if (!values.members) {
  console.error("Error: --members <path> is required");
  process.exit(1);
}

// Resolve python interpreter: explicit flag > venv > system python
const cwd = process.cwd();
const venvWin = join(cwd, "venv", "Scripts", "python.exe");
const venvUnix = join(cwd, "venv", "bin", "python");
const pythonBin =
  values.python ||
  (existsSync(venvWin) ? venvWin : existsSync(venvUnix) ? venvUnix : "python");

const pngData = readFileSync(values.png);
const base64image = pngData.toString("base64");

const membersRaw = readFileSync(values.members, "utf8");
const members = JSON.parse(membersRaw);

const payload = JSON.stringify({ members, base64image });

const gpq = spawn(pythonBin, ["gpq.py", "--subprocess", "1", "--style", values.style], {
  cwd: process.cwd(),
});

let stdout = "";
let stderr = "";

gpq.stdout.on("data", (chunk) => {
  stdout += chunk.toString();
});

gpq.stderr.on("data", (chunk) => {
  stderr += chunk.toString();
});

gpq.on("close", (code) => {
  if (stderr) {
    console.error("--- stderr ---");
    console.error(stderr.trim());
  }
  if (code !== 0) {
    console.error(`gpq.py exited with code ${code}`);
    process.exit(code ?? 1);
  }
  try {
    const result = JSON.parse(stdout.trim());
    console.log(JSON.stringify(result, null, 2));
  } catch {
    console.log(stdout.trim());
  }
});

gpq.stdin.on("error", (err) => {
  if (err.code !== "EPIPE" && err.code !== "EOF") {
    console.error("stdin error:", err.message);
  }
});

gpq.stdin.write(payload);
gpq.stdin.end();
