import catalogueIndex from '../../../data/catalogue-index.json';
import catalogueDetails from '../../../data/catalogue-details.json';
import rawBackgrounds from '../../../data/kinetic-backgrounds.json';
import { toSlug } from '../../../lib/catalogue';

export function getStaticPaths() {
  const cataloguePaths = (['equilibria', 'perturbations'] as const).flatMap((category) =>
    catalogueIndex[category].map((record) => ({
      params: { category, slug: record.slug },
      props: { category, slug: record.slug },
    })),
  );
  const backgroundPaths = (rawBackgrounds as any[]).map((background) => ({
    params: { category: 'backgrounds', slug: toSlug(background.className) },
    props: { category: 'backgrounds', slug: toSlug(background.className) },
  }));
  return [...cataloguePaths, ...backgroundPaths];
}

export function GET({ props }: { props: { category: 'equilibria' | 'perturbations' | 'backgrounds'; slug: string } }) {
  if (props.category === 'backgrounds') {
    const item = (rawBackgrounds as any[]).find((entry) => toSlug(entry.className) === props.slug);
    if (!item) return new Response('Not found', { status: 404 });
    return new Response(JSON.stringify(item), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
  }
  const item = catalogueDetails[props.category].find((entry) => entry.slug === props.slug);
  if (!item) return new Response('Not found', { status: 404 });
  return new Response(JSON.stringify(item), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
