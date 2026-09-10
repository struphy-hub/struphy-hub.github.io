// Thin theme + filter layer around @scope-profiler/plotly's figure builders,
// used to render scope-profiler's plot-data JSON (durations/gantt/flame)
// natively on each example page instead of embedding a separate report.
import {
  buildDurationsFigure,
  buildFlameFigure,
  buildGanttFigure,
  renderFigure as renderScopeFigure,
} from '@scope-profiler/plotly';

const FONT_FAMILY =
  '"Fira Sans", ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif';

function themeColors() {
  const dark =
    typeof document !== 'undefined' && document.documentElement.dataset.theme === 'dark';
  return dark
    ? { text: '#e8e6df', muted: '#9b9689', grid: '#33312b', hoverBg: '#1c1b17' }
    : { text: '#1c1b17', muted: '#6b675c', grid: '#e4e1d6', hoverBg: '#ffffff' };
}

export const plotConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

// What the durations chart starts filtered to: the top-level integration
// loop, the merged setup span, and the propagators -- not every "kernel:"/
// "accum:"/"setup var:" line, which is what makes an unfiltered chart
// hundreds of pixels taller than useful. "^prop:" is anchored so it doesn't
// also pull in "setup prop: X"; clearing the box shows everything.
export const DEFAULT_REGION_FILTER = 'model.integrate, ^prop:, setup: total';

// A region filter is a comma-separated list of terms, each matched as a
// case-insensitive substring of the region name; a leading "^" anchors a
// term to the start of the name. An empty filter matches every region.
export function parseRegionFilter(text) {
  return String(text ?? '')
    .split(',')
    .map((term) => term.trim().toLowerCase())
    .filter(Boolean);
}

export function matchesRegionFilter(region, terms) {
  const name = String(region ?? '').toLowerCase();
  return terms.some((term) =>
    term.startsWith('^') ? name.startsWith(term.slice(1)) : name.includes(term),
  );
}

// Renders one of the three chart kinds this page uses into `container`,
// themed for the current light/dark mode and filtered to `regionFilterText`
// (ignored for flame, which always shows the full call hierarchy).
export async function renderFigure(Plotly, container, kind, payload, { metric, regionFilterText } = {}) {
  if (!container) return;
  const terms = parseRegionFilter(regionFilterText);
  const filterRegion = terms.length ? (region) => matchesRegionFilter(region, terms) : undefined;

  const c = themeColors();
  const layout = {
    font: { family: FONT_FAMILY, color: c.text, size: 12 },
    legend: { font: { color: c.muted, size: 11 } },
    hoverlabel: { bgcolor: c.hoverBg, bordercolor: c.grid, font: { color: c.text } },
  };

  let figure;
  // Gantt already labels each row with its region name, so a color legend
  // is pure redundancy -- turn it off.
  if (kind === 'gantt') figure = buildGanttFigure(payload, { filterRegion, layout: { ...layout, showlegend: false } });
  else if (kind === 'flame') figure = buildFlameFigure(payload, { layout });
  else if (kind === 'durations') figure = buildDurationsFigure(payload, { filterRegion, metric: metric ?? 'total', layout });
  else throw new Error(`Unknown chart kind: ${kind}`);

  await renderScopeFigure(Plotly, container, figure, plotConfig);
}
