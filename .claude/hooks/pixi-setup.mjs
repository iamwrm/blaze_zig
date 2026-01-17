import fs from 'fs';
import { execSync } from 'child_process';

const log = (msg) => console.error(`[pixi-setup] ${msg}`);
const { HOME, CLAUDE_CODE_REMOTE, CLAUDE_ENV_FILE } = process.env;

if (CLAUDE_CODE_REMOTE !== 'true') process.exit(0);

try {
  log(execSync('pixi --version', { encoding: 'utf8' }).trim());
  process.exit(0);
} catch {}

log('Installing pixi...');
try {
  execSync(`curl -fsSL https://pixi.sh/install.sh | PIXI_HOME="${HOME}/.local" sh`, {
    stdio: 'inherit',
    shell: '/bin/bash'
  });
  if (CLAUDE_ENV_FILE) fs.appendFileSync(CLAUDE_ENV_FILE, `export PATH="${HOME}/.local/bin:$PATH"\n`);
  log(execSync(`${HOME}/.local/bin/pixi --version`, { encoding: 'utf8' }).trim());
} catch (e) {
  log(`Failed: ${e.message}`);
}
