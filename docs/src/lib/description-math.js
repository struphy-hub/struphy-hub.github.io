import katex from 'katex';

const escapeHtml = (text) => text.replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
})[char]);

// Match equations before escaping prose, so LaTeX alignment (&) and comparisons
// (<, >) reach KaTeX unchanged. Ordinary description text never becomes HTML.
export function renderDescription(text = '', { compact = false } = {}) {
  const math = /\$\$([\s\S]*?)\$\$|:math:`([^`]+)`|^[ \t]*\.\. math::[ \t]*\r?\n(?:[ \t]*\r?\n)*((?:[ \t]+\S[^\n]*(?:\r?\n|$))+)/gm;
  let html = '';
  let cursor = 0;
  for (const match of text.matchAll(math)) {
    html += escapeHtml(text.slice(cursor, match.index));
    const latex = (match[1] ?? match[2] ?? match[3]).trim();
    html += katex.renderToString(latex, {
      displayMode: !compact && match[2] === undefined,
      throwOnError: false,
      strict: false,
      trust: false,
    });
    cursor = match.index + match[0].length;
  }
  return html + escapeHtml(text.slice(cursor));
}
