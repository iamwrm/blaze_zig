import fs from 'fs';
import { execSync } from 'child_process';

const LOG_PREFIX = '[pixi-setup]';
const LOCAL_BIN = `${process.env.HOME}/.local/bin`;
const PIXI_PATH = `${LOCAL_BIN}/pixi`;

function log(msg) {
  console.error(`${LOG_PREFIX} ${msg}`);
}

function updatePath() {
  const envFile = process.env.CLAUDE_ENV_FILE;
  if (envFile) {
    fs.appendFileSync(envFile, `export PATH="${LOCAL_BIN}:$PATH"\n`);
    log('PATH persisted to CLAUDE_ENV_FILE');
  }
}

function run() {
  // Only run in remote Claude Code environment
  if (process.env.CLAUDE_CODE_REMOTE !== 'true') {
    log('Not a remote session, skipping');
    process.exit(0);
  }

  log('Remote session detected, checking pixi...');

  // Check if pixi is already available
  try {
    const version = execSync('pixi --version', { encoding: 'utf8' }).trim();
    log(`pixi already available: ${version}`);
    process.exit(0);
  } catch (e) {
    // pixi not found, continue with installation
  }

  // Check if pixi exists in local bin
  if (fs.existsSync(PIXI_PATH)) {
    log(`pixi found in ${LOCAL_BIN}`);
    updatePath();
    process.exit(0);
  }

  log('Installing pixi...');

  // Create local bin directory
  fs.mkdirSync(LOCAL_BIN, { recursive: true });

  try {
    // Install pixi using the official install script
    // PIXI_HOME controls where pixi is installed
    execSync(
      `curl -fsSL --proto '=https' --tlsv1.2 --connect-timeout 5 --max-time 120 https://pixi.sh/install.sh | PIXI_HOME="${process.env.HOME}/.local" sh`,
      { stdio: 'inherit', shell: '/bin/bash' }
    );

    updatePath();

    const version = execSync(`${PIXI_PATH} --version`, { encoding: 'utf8' }).trim();
    log(`pixi installed successfully: ${version}`);
  } catch (e) {
    log(`Failed to install pixi: ${e.message}`);
  }

  process.exit(0);
}

run();
