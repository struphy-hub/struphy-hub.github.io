const fs = require('node:fs/promises');
const { exampleCache } = require('./example-cache.cjs');

const selectionPath = '.github/pitagora-examples.txt';

async function selectedPitagoraExamples() {
  const text = await fs.readFile(selectionPath, 'utf8');
  const examples = text
    .split('\n')
    .map(line => line.split('#', 1)[0].trim())
    .filter(Boolean);
  if (!examples.length) throw new Error(`${selectionPath} contains no examples`);
  if (new Set(examples).size !== examples.length) {
    throw new Error(`${selectionPath} contains duplicate examples`);
  }
  for (const example of examples) {
    await fs.access(`docs/src/examples/${example}.py`);
  }
  return examples;
}

async function listPitagoraExamples({ cache, glob, core }) {
  const examples = await selectedPitagoraExamples();
  const misses = [];
  const cachedPaths = [];
  for (const example of examples) {
    const { key, paths } = await exampleCache(example, 'pitagora-example-v1', '', glob);
    const restored = await cache.restoreCache(paths, key);
    if (restored === key) {
      await fs.access(`docs/public/examples/${example}.plotly.json`);
      await fs.access(`docs/public/examples/${example}.html`);
      cachedPaths.push(...paths);
      core.info(`${example}: Pitagora cache hit`);
    } else {
      misses.push(example);
      core.info(`${example}: Pitagora cache miss`);
    }
  }
  return { examples, misses, cachedPaths };
}

module.exports = { listPitagoraExamples, selectedPitagoraExamples };
