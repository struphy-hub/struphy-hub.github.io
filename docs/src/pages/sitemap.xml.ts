import rawModels from '../data/models.json';
import rawPropagators from '../data/propagators.json';
import rawBackgrounds from '../data/kinetic-backgrounds.json';
import catalogueIndex from '../data/catalogue-index.json';
import examples from '../data/examples-index.json';
import { toSlug } from '../lib/catalogue';

export async function GET() {
  const paths = [
    '/', '/examples/', '/models/', '/propagators/', '/domains/', '/equilibria/', '/perturbations/', '/backgrounds/', '/simulation/', '/numerics/', '/feec/', '/time-integration/', '/search/', '/citation/',
    ...examples.map((example: any) => example.href),
    ...rawModels.map((model: any) => `/models/${toSlug(model.className)}/`),
    ...rawPropagators.map((propagator: any) => `/propagators/${toSlug(propagator.className)}/`),
    ...rawBackgrounds.map((background: any) => `/backgrounds/${toSlug(background.className)}/`),
    ...catalogueIndex.equilibria.map((item) => `/equilibria/${item.slug}/`),
    ...catalogueIndex.perturbations.map((item) => `/perturbations/${item.slug}/`),
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${paths.map((path) => `  <url><loc>https://struphy-hub.github.io${path}</loc></url>`).join('\n')}\n</urlset>\n`;
  return new Response(body, { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
}
