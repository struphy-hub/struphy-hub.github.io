const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { test } = require('node:test');
const listExamples = require('./list-examples.cjs');
const { exampleCache } = require('./example-cache.cjs');

async function plan(t, hits, { partial = false, incomplete = false, failure = false } = {}) {
  const cwd = process.cwd();
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'example-cache-test-'));
  t.after(async () => { process.chdir(cwd); await fs.rm(root, { recursive: true, force: true }); });
  process.chdir(root);
  await fs.mkdir('docs/src/examples', { recursive: true });
  await fs.mkdir('docs/public/examples', { recursive: true });
  for (const name of ['alpha.py', 'beta.py', '_gallery.py', 'poisson-source.py', 'README.md']) {
    await fs.writeFile(`docs/src/examples/${name}`, '');
  }
  const outputs = {};
  const restored = [];
  await listExamples({
    glob: { hashFiles: async () => 'content-hash' },
    core: { info() {}, setOutput: (key, value) => { outputs[key] = value; } },
    cache: {
      async restoreCache(paths, key) {
        if (failure) throw new Error('cache service unavailable');
        const example = paths[0].match(/\/([^/]+)\*$/)[1];
        restored.push(example);
        assert.equal(key, `struphy-example-serial-v1-${example}-content-hash`);
        if (!hits.includes(example)) return undefined;
        await fs.writeFile(`docs/public/examples/${example}.plotly.json`, '{}');
        if (!incomplete) await fs.writeFile(`docs/public/examples/${example}.html`, 'figure');
        return partial ? `${key}-other` : key;
      },
    },
  });
  assert.deepEqual(restored, ['alpha', 'beta']);
  return outputs;
}

test('all hits skip the simulation matrix and publish cached outputs', async t => {
  const output = await plan(t, ['alpha', 'beta']);
  assert.deepEqual(JSON.parse(output.examples), []);
  assert.equal(output['cached-paths'].split('\n').length, 6);
});

test('all misses schedule every hosted example without a cached artifact', async t => {
  const output = await plan(t, []);
  assert.deepEqual(JSON.parse(output.examples), ['alpha', 'beta']);
  assert.equal(output['cached-paths'], '');
});

test('mixed results publish only hits and schedule only misses', async t => {
  const output = await plan(t, ['alpha']);
  assert.deepEqual(JSON.parse(output.examples), ['beta']);
  assert.equal(output['cached-paths'], [
    'docs/public/examples/alpha*', 'docs/public/images/examples/alpha*',
    'docs/public/example-domains/alpha*',
  ].join('\n'));
});

test('a partial key match is not treated as an exact cache hit', async t => {
  const output = await plan(t, ['alpha'], { partial: true });
  assert.deepEqual(JSON.parse(output.examples), ['alpha', 'beta']);
  assert.equal(output['cached-paths'], '');
});

test('incomplete cached output fails rather than silently omitting an example', async t => {
  await assert.rejects(plan(t, ['alpha'], { incomplete: true }), /ENOENT/);
});

test('cache client failures do not silently omit examples', async t => {
  await assert.rejects(plan(t, [], { failure: true }), /cache service unavailable/);
});

test('shared keys retain prefix, suffix, example and source hash', async () => {
  let patterns;
  const result = await exampleCache('alpha', 'pitagora-example-v1', '-custom', {
    hashFiles: async value => { patterns = value.split('\n'); return 'hash'; },
  });
  assert.equal(result.key, 'pitagora-example-v1-custom-alpha-hash');
  assert.equal(patterns[0], 'docs/src/examples/alpha.py');
  assert.ok(patterns.includes('submodules/struphy/src/struphy/**/*.py'));
  assert.ok(patterns.includes('docs/src/examples/_gallery.py'));
});
