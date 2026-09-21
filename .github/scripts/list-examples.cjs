const fs = require('node:fs/promises');
const { exampleCache } = require('./example-cache.cjs');

module.exports = async function listExamples({ cache, glob, core }) {
  const examples = (await fs.readdir('docs/src/examples'))
    .filter(name => name.endsWith('.py') && !name.startsWith('_') && name !== 'poisson-source.py')
    .map(name => name.slice(0, -3)).sort();
  const misses = [];
  const cachedPaths = [];
  for (const example of examples) {
    const { key, paths } = await exampleCache(example, 'struphy-example-serial-v1', '', glob);
    const restored = await cache.restoreCache(paths, key);
    if (restored === key) {
      // Fail early on incomplete cached output, just as the simulation job does.
      await fs.access(`docs/public/examples/${example}.plotly.json`);
      await fs.access(`docs/public/examples/${example}.html`);
      cachedPaths.push(...paths);
      core.info(`${example}: cache hit`);
    } else {
      misses.push(example);
      core.info(`${example}: cache miss`);
    }
  }
  core.setOutput('examples', JSON.stringify(misses));
  core.setOutput('cached-paths', cachedPaths.join('\n'));
  core.info(`${examples.length - misses.length} cached; ${misses.length} examples to run`);
};
