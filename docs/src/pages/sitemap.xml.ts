import rawModels from '../data/models.json';
import catalogueIndex from '../data/catalogue-index.json';
import examples from '../data/examples-index.json';
import { toSlug } from '../lib/catalogue';

export async function GET() {
  const paths = [
    '/', '/examples/', '/models/', '/domains/', '/equilibria/', '/perturbations/', '/search/', '/citation/',
    ...examples.map((example: any) => example.href),
    ...rawModels.map((model: any) => `/models/${toSlug(model.className)}/`),
    ...catalogueIndex.equilibria.map((item) => `/equilibria/${item.slug}/`),
    ...catalogueIndex.perturbations.map((item) => `/perturbations/${item.slug}/`),
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${paths.map((path) => `  <url><loc>https://struphy-hub.github.io${path}</loc></url>`).join('\n')}\n</urlset>\n`;
  return new Response(body, { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
}
