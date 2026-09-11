import katex from 'katex';
import rawPropagators from '../../../data/propagators.json';
import { toSlug } from '../../../lib/catalogue';

function renderMath(html = '', items: any[] = []) {
  return items.reduce((result, { token, latex, display }) => {
    let rendered;
    try { rendered = katex.renderToString(latex, { displayMode: display, throwOnError: false, strict: false, output: 'html' }); }
    catch { rendered = `<code>${latex}</code>`; }
    return result.split(token).join(display ? `<div class="math-display">${rendered}</div>` : rendered);
  }, html);
}

export function getStaticPaths() {
  return (rawPropagators as any[]).map((propagator) => ({ params: { slug: toSlug(propagator.className) }, props: { propagator } }));
}

export function GET({ props }: { props: { propagator: any } }) {
  const { descriptionMath, options, ...source } = props.propagator;
  const propagator = { ...source, slug: toSlug(source.className) };
  propagator.descriptionHtml = renderMath(source.descriptionHtml, descriptionMath ?? []);
  propagator.options = (options as any[]).map((option) => ({
    ...option,
    descriptionHtml: renderMath(option.descriptionHtml, option.descriptionMath ?? []),
  }));
  return new Response(JSON.stringify(propagator), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
