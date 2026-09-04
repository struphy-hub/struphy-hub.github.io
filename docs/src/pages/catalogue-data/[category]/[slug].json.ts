import catalogueIndex from '../../../data/catalogue-index.json';
import catalogueDetails from '../../../data/catalogue-details.json';

export function getStaticPaths() {
  return (['equilibria', 'perturbations'] as const).flatMap((category) =>
    catalogueIndex[category].map((record) => ({
      params: { category, slug: record.slug },
      props: { category, slug: record.slug },
    })),
  );
}

export function GET({ props }: { props: { category: 'equilibria' | 'perturbations'; slug: string } }) {
  const item = catalogueDetails[props.category].find((entry) => entry.slug === props.slug);
  if (!item) return new Response('Not found', { status: 404 });
  return new Response(JSON.stringify(item), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
