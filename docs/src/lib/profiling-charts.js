// Thin theme + filter layer around @scope-profiler/plotly's figure builders,
// used to render scope-profiler's plot-data JSON (durations/gantt) natively on
// each example page instead of embedding a separate report.
import {
  buildDurationsFigure,
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

// Renders one of the two chart kinds this page uses into `container`, themed
// for the current light/dark mode and filtered to `regionFilterText`.
export async function renderFigure(Plotly, container, kind, payload, { metric, regionFilterText } = {}) {
  if (!container) return;
  const terms = parseRegionFilter(regionFilterText);
  const filterRegion = terms.length ? (region) => matchesRegionFilter(region, terms) : undefined;

  const c = themeColors();
  const layout = {
    font: { family: FONT_FAMILY, color: c.text, size: 12 },
    // `layout` is spread over the builder's own layout, so this replaces its
    // legend config wholesale rather than merging into it -- the orientation
    // has to be restated here or a stacked chart's dozen segments pile up in a
    // scrolling column on top of the plot. Anchoring to the container rather
    // than the plot area keeps it clear of the rotated region labels, whose
    // depth depends on how long the region names happen to be.
    legend: { orientation: 'h', yref: 'container', y: 0, yanchor: 'bottom', x: 0, font: { color: c.muted, size: 11 } },
    hoverlabel: { bgcolor: c.hoverBg, bordercolor: c.grid, font: { color: c.text } },
  };

  let figure;
  // Gantt already labels each row with its region name, so a color legend
  // is pure redundancy -- turn it off.
  if (kind === 'gantt') figure = buildGanttFigure(payload, { filterRegion, layout: { ...layout, showlegend: false } });
  // The payload is exported with --stack-children, so each bar carries a
  // `segment` per callee and the builder stacks them; the region order is the
  // exporter's --sort-by total, which the builder preserves.
  else if (kind === 'durations') figure = buildDurationsFigure(payload, { filterRegion, metric: metric ?? 'total', layout });
  else throw new Error(`Unknown chart kind: ${kind}`);

  await renderScopeFigure(Plotly, container, figure, plotConfig);
}
