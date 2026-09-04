import { readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dataRoot = join(root, 'src', 'data');

function toSlug(value) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-zA-Z0-9-]/g, '')
    .replace(/-+/g, '-')
    .toLowerCase();
}

function excerpt(value = '', maxLength = 170) {
  const text = value.replace(/<[^>]+>/g, ' ').replace(/@@MATH\d+@@/g, 'mathematical formulation').replace(/\s+/g, ' ').trim();
  return text.length > maxLength ? `${text.slice(0, maxLength).trimEnd()}…` : text;
}

function sampleLine(values = [], size = 120) {
  if (values.length <= size) return values;
  return Array.from({ length: size }, (_, index) => values[Math.round(index * (values.length - 1) / (size - 1))]);
}

function sampleMatrix(values = [], size = 28) {
  const rows = values.length;
  const columns = rows ? values[0].length : 0;
  if (!rows || !columns) return [];
  return Array.from({ length: size }, (_, y) => {
    const row = values[Math.round(y * (rows - 1) / (size - 1))];
    return Array.from({ length: size }, (_, x) => row[Math.round(x * (columns - 1) / (size - 1))]);
  });
}

function compactSlice(slice, fields) {
  if (!slice) return null;
  return {
    axes: slice.axes,
    value: slice.value,
    fields: Object.fromEntries(fields.filter((field) => slice.fields?.[field]?.values).map((field) => [field, { values: sampleMatrix(slice.fields[field].values) }])),
  };
}

function compactRecord(data, category, slug) {
  const planes = ['xy', 'xz', 'yz'];
  const lineAxes = category === 'equilibria' ? ['x', 'y', 'z'] : ['eta1', 'eta2', 'eta3'];
  const fields = category === 'equilibria' ? ['p', 'n'] : ['u'];
  const slices = Object.fromEntries(planes.map((plane) => [plane, compactSlice(data.slices?.[plane] ?? data.slice, fields)]));
  const centerlines = data.centerlines ? Object.fromEntries(lineAxes.map((axis) => {
    const line = data.centerlines[axis];
    return [axis, line ? { fields: Object.fromEntries(fields.filter((field) => line.fields?.[field]).map((field) => [field, sampleLine(line.fields[field])])) } : null];
  })) : null;
  return {
    slug,
    type: data.type,
    description_html: data.description_html,
    given_in_basis: data.given_in_basis,
    parameters: data.parameters,
    parameter_descriptions: data.parameter_descriptions,
    slices,
    centerlines,
  };
}

async function readCategory(category) {
  const directory = join(dataRoot, category);
  const files = (await readdir(directory)).filter((name) => name.endsWith('.json')).sort();
  const records = [];
  const details = [];
  const seen = new Set();
  for (const filename of files) {
    const data = JSON.parse(await readFile(join(directory, filename), 'utf8'));
    if (seen.has(data.type)) continue;
    seen.add(data.type);
    const slug = toSlug(data.type);
    records.push({
      file: relative(dataRoot, join(directory, filename)),
      type: data.type,
      slug,
      parameterCount: Object.keys(data.parameters ?? {}).filter((key) => key !== 'self').length,
      ...(category === 'equilibria' ? { description: excerpt(data.description_html) } : {}),
      ...(category === 'perturbations' && data.given_in_basis ? { basis: data.given_in_basis } : {}),
    });
    details.push(compactRecord(data, category, slug));
  }
  return {
    records: records.sort((a, b) => a.type.localeCompare(b.type)),
    details: details.sort((a, b) => a.type.localeCompare(b.type)),
  };
}

const equilibria = await readCategory('equilibria');
const perturbations = await readCategory('perturbations');
const index = {
  equilibria: equilibria.records,
  perturbations: perturbations.records,
};
const details = { equilibria: equilibria.details, perturbations: perturbations.details };
await writeFile(join(dataRoot, 'catalogue-index.json'), `${JSON.stringify(index, null, 2)}\n`);
await writeFile(join(dataRoot, 'catalogue-details.json'), `${JSON.stringify(details)}\n`);
