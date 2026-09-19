import { access, readdir, readFile, writeFile } from 'node:fs/promises';
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
const imagesDir = join(root, 'public', 'images', 'examples');
const scriptsDir = join(root, 'src', 'examples');
const outFile = join(root, 'src', 'data', 'examples-index.json');

function toSlug(value) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-zA-Z0-9-]/g, '')
    .replace(/-+/g, '-')
    .toLowerCase();
}

const exists = (path) => access(path).then(() => true, () => false);

// Everything in public/examples and public/images/examples is generated (and gitignored), so a fresh
// clone has none of it. Each example page imports its metadata JSON, so stop here with a clear
// instruction rather than letting the build fail on a missing import.
const scripts = (await readdir(scriptsDir))
  .filter((name) => name.endsWith('.py') && !name.startsWith('_'))
  .map((name) => name.replace(/\.py$/, ''));
const missing = [];
for (const slug of scripts) {
  if (!(await exists(join(examplesDir, `${slug}.metadata.json`)))) missing.push(slug);
}
if (missing.length > 0) {
  console.error(
    `\nMissing example metadata for: ${missing.join(', ')}\n\n` +
      'These files are generated, not committed. From the repository root run\n' +
      '  python cli.py metadata --all      # metadata only; needs Struphy, not its compiled kernels\n' +
      '  python cli.py run <example>       # metadata, figures and thumbnails of one example\n' +
      'or, for the pages to show their figures, python cli.py run --all (slow).\n',
  );
  process.exit(1);
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
    // Keep the complete record here as well: prerendered detail pages consume
    // this generated index, which avoids resolving generated public files from
    // Astro's emitted server modules.
    data,
    // Named after the script, like every generated file; absent until the example has been run.
    thumbnail: (await exists(join(imagesDir, `${slug}.png`))) ? `/images/examples/${slug}.png` : null,
  });
}

await writeFile(outFile, `${JSON.stringify(examples, null, 2)}\n`);
console.log(`Wrote ${examples.length} example(s) to ${outFile}`);
