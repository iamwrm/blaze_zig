import fs from 'fs';
import { execSync } from 'child_process';

const log = (msg) => console.error(`[pixi-setup] ${msg}`);
const { HOME, CLAUDE_CODE_REMOTE, CLAUDE_ENV_FILE } = process.env;

if (CLAUDE_CODE_REMOTE !== 'true') process.exit(0);

const pixiBin = `${HOME}/.local/bin/pixi`;

// Check if pixi is already installed (either in PATH or at expected location)
try {
  log(execSync('pixi --version', { encoding: 'utf8' }).trim());
  process.exit(0);
} catch {}

// Check if pixi exists at the expected location but not in PATH
try {
  const version = execSync(`${pixiBin} --version`, { encoding: 'utf8' }).trim();
  log(`Found pixi at ${pixiBin}: ${version}`);
  if (CLAUDE_ENV_FILE) fs.appendFileSync(CLAUDE_ENV_FILE, `export PATH="${HOME}/.local/bin:$PATH"\n`);
  process.exit(0);
} catch {}

log('Installing pixi...');
try {
  execSync(`curl -fsSL https://pixi.sh/install.sh | PIXI_HOME="${HOME}/.local" sh`, {
    stdio: 'inherit',
    shell: '/bin/bash'
  });
  if (CLAUDE_ENV_FILE) fs.appendFileSync(CLAUDE_ENV_FILE, `export PATH="${HOME}/.local/bin:$PATH"\n`);
  log(execSync(`${pixiBin} --version`, { encoding: 'utf8' }).trim());
} catch (e) {
  log(`Failed: ${e.message}`);
}
