import { readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// Mirrors generate-catalogue-index.mjs: examples live in docs/public/examples/
// (a Python-generated, non-module-graph directory), so this plain Node script
// -- not an Astro page -- reads it and writes a real src/data/ file that
// pages can statically import. The example's page directory (docs/src/pages/
// examples/<slug>/) is expected to match its metadata filename's slug, e.g.
// `maxwell-wave.metadata.json` <-> `/examples/maxwell-wave/`.

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const examplesDir = join(root, 'public', 'examples');
const outFile = join(root, 'src', 'data', 'examples-index.json');

function toSlug(value) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-zA-Z0-9-]/g, '')
    .replace(/-+/g, '-')
    .toLowerCase();
}

const files = (await readdir(examplesDir)).filter((name) => name.endsWith('.metadata.json')).sort();
const examples = [];
for (const filename of files) {
  const slug = filename.replace(/\.metadata\.json$/, '');
  const data = JSON.parse(await readFile(join(examplesDir, filename), 'utf8'));
  examples.push({
    slug,
    href: `/examples/${slug}/`,
    name: data.name,
    description: data.description,
    model: data.model,
    modelSlug: data.model ? toSlug(data.model) : null,
    thumbnail: data.thumbnail ?? null,
  });
}

await writeFile(outFile, `${JSON.stringify(examples, null, 2)}\n`);
console.log(`Wrote ${examples.length} example(s) to ${outFile}`);
