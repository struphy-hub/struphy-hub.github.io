import katex from 'katex';
import rawModels from '../../../data/models.json';
import { toSlug } from '../../../lib/catalogue';

const fields = ['pde','longDescription','normalization','discretization','scalarQuantities','useCases','examples'];

function renderMath(html = '', items: any[] = []) {
  return items.reduce((result, { token, latex, display }) => {
    let rendered;
    try { rendered = katex.renderToString(latex, { displayMode: display, throwOnError: false, strict: false, output: 'html' }); }
    catch { rendered = `<code>${latex}</code>`; }
    return result.split(token).join(display ? `<div class="math-display">${rendered}</div>` : rendered);
  }, html);
}

export function getStaticPaths() {
  return rawModels.map((model: any) => ({ params: { slug: toSlug(model.className) }, props: { model } }));
}

export function GET({ props }: { props: { model: any } }) {
  const { cannotBeUsedForHtml, cannotBeUsedForMath, ...source } = props.model;
  const model = { ...source, slug: toSlug(source.className) };
  for (const field of fields) model[`${field}Html`] = renderMath(model[`${field}Html`], model[`${field}Math`] ?? []);
  return new Response(JSON.stringify(model), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
