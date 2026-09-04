import rawModels from '../data/models.json';
import { toSlug } from '../lib/catalogue';

export async function GET() {
  const equilibriumModules = import.meta.glob('../data/equilibria/*.json', { eager: true, import: 'default' });
  const perturbationModules = import.meta.glob('../data/perturbations/*.json', { eager: true, import: 'default' });
  const equilibria = [...new Set(Object.values(equilibriumModules).map((item: any) => item.type))];
  const perturbations = [...new Set(Object.values(perturbationModules).map((item: any) => item.type))];
  const paths = [
    '/', '/models/', '/domains/', '/equilibria/', '/perturbations/', '/search/', '/citation/',
    ...rawModels.map((model: any) => `/models/${toSlug(model.className)}/`),
    ...equilibria.map((name) => `/equilibria/${toSlug(name)}/`),
    ...perturbations.map((name) => `/perturbations/${toSlug(name)}/`),
  ];
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${paths.map((path) => `  <url><loc>https://struphy-hub.github.io${path}</loc></url>`).join('\n')}\n</urlset>\n`;
  return new Response(body, { headers: { 'Content-Type': 'application/xml; charset=utf-8' } });
}
