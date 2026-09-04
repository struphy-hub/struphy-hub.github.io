export function toSlug(value: string) {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
    .replace(/[_\s]+/g, '-')
    .replace(/[^a-zA-Z0-9-]/g, '')
    .replace(/-+/g, '-')
    .toLowerCase();
}

export function excerpt(value = '', maxLength = 170) {
  const text = value
    .replace(/<[^>]+>/g, ' ')
    .replace(/@@MATH\d+@@/g, 'mathematical formulation')
    .replace(/\s+/g, ' ')
    .trim();

  return text.length > maxLength ? `${text.slice(0, maxLength).trimEnd()}…` : text;
}

export function sampleGrid(values: number[][], size = 28) {
  const rows = values.length;
  const cols = rows ? values[0].length : 0;
  const sampled = Array.from({ length: size * size }, (_, index) => {
    const x = index % size;
    const y = Math.floor(index / size);
    const row = values[Math.min(rows - 1, Math.floor(y * rows / size))];
    const value = row?.[Math.min(cols - 1, Math.floor(x * cols / size))] ?? 0;
    return Number.isFinite(value) ? value : 0;
  });
  const min = Math.min(...sampled);
  const max = Math.max(...sampled);
  const span = max - min || 1;

  return {
    size,
    min,
    max,
    values: sampled.map((value) => Math.round(((value - min) / span) * 10000) / 10000),
  };
}
