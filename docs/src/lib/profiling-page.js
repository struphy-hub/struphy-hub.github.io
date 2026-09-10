// Wires up the "Performance" section on an example detail page: fetches
// scope-profiler's plot-data JSON (durations/gantt/flame + region stats) and
// renders it as native Plotly figures, re-rendering on theme change.
import Plotly from 'plotly.js-dist-min';
import { DEFAULT_REGION_FILTER, matchesRegionFilter, parseRegionFilter, renderFigure } from './profiling-charts.js';

const fmt = (value) => {
  if (typeof value !== 'number' || Number.isNaN(value)) return '-';
  return Math.abs(value) >= 1 ? value.toFixed(4) : value.toExponential(3);
};

const esc = (value) =>
  String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return res.json();
}

function applyTableFilter(input, tbody) {
  const terms = parseRegionFilter(input.value);
  let visible = 0;
  for (const row of tbody.querySelectorAll('tr[data-region]')) {
    const match = !terms.length || matchesRegionFilter(row.dataset.region, terms);
    row.hidden = !match;
    if (match) visible += 1;
  }
  let emptyRow = tbody.querySelector('tr.pf-empty-row');
  if (!visible) {
    if (!emptyRow) {
      emptyRow = document.createElement('tr');
      emptyRow.className = 'pf-empty-row';
      emptyRow.innerHTML = '<td colspan="6">No regions match the filter.</td>';
      tbody.appendChild(emptyRow);
    }
  } else if (emptyRow) {
    emptyRow.remove();
  }
}

async function mountTable(url, tbody, filterInput) {
  const data = await fetchJson(url);
  const file = (data.files || [])[0];
  const stats = file?.region_statistics || {};
  const regions = Array.isArray(data.common_regions) && data.common_regions.length
    ? data.common_regions
    : Object.keys(stats);

  tbody.innerHTML = regions
    .map((region) => {
      const s = stats[region];
      if (!s) return '';
      return `
        <tr data-region="${esc(region)}">
          <td>${esc(region)}</td>
          <td>${s.count ?? '-'}</td>
          <td>${fmt(s.average_duration_seconds)}</td>
          <td>${fmt(s.min_duration_seconds)}</td>
          <td>${fmt(s.max_duration_seconds)}</td>
          <td>${fmt(s.total_duration_seconds)}</td>
        </tr>
      `;
    })
    .join('');
  if (!tbody.innerHTML) tbody.innerHTML = '<tr><td colspan="6">No region statistics available.</td></tr>';

  let timer = null;
  filterInput.addEventListener('input', () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => applyTableFilter(filterInput, tbody), 150);
  });
}

// Re-render every mounted chart when the theme toggle flips `data-theme`, so
// their text/legend colors follow along. The site has no custom theme-change
// event, so this watches the attribute directly.
const themedDraws = new Set();
let themeObserverStarted = false;
function ensureThemeObserver() {
  if (themeObserverStarted || typeof document === 'undefined') return;
  themeObserverStarted = true;
  new MutationObserver(() => {
    for (const draw of themedDraws) draw();
  }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
}

async function mountChart(url, container, { kind, metricSelect, filterInput }) {
  const payload = await fetchJson(url);
  const draw = () =>
    renderFigure(Plotly, container, kind, payload, {
      metric: metricSelect?.value,
      regionFilterText: filterInput?.value ?? '',
    });
  await draw();
  metricSelect?.addEventListener('change', draw);
  let timer = null;
  filterInput?.addEventListener('input', () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(draw, 200);
  });
  ensureThemeObserver();
  themedDraws.add(draw);
  return draw;
}

export function mountProfilingSection(root) {
  if (!root) return;
  const { durations, gantt, flame, stats } = root.dataset;

  const durationsContainer = root.querySelector('#pf-plot-durations');
  const durationsMetric = root.querySelector('#pf-metric');
  const durationsFilter = root.querySelector('#pf-filter-durations');
  // Unfiltered, this is one bar per region -- including every fine-grained
  // "kernel:"/"accum:"/"setup var:" line -- which makes for an absurdly tall
  // chart. Start narrowed to the top-level view; clearing the box (or typing
  // a different filter) shows everything.
  if (durationsFilter && !durationsFilter.value) durationsFilter.value = DEFAULT_REGION_FILTER;
  if (durations && durationsContainer) {
    mountChart(durations, durationsContainer, { kind: 'durations', metricSelect: durationsMetric, filterInput: durationsFilter }).catch(
      (error) => {
        durationsContainer.innerHTML = `<p class="pf-error">Could not load durations: ${esc(error)}</p>`;
      },
    );
  }

  const ganttContainer = root.querySelector('#pf-plot-gantt');
  const ganttFilter = root.querySelector('#pf-filter-gantt');
  if (ganttFilter && !ganttFilter.value) ganttFilter.value = DEFAULT_REGION_FILTER;
  if (gantt && ganttContainer) {
    mountChart(gantt, ganttContainer, { kind: 'gantt', filterInput: ganttFilter }).catch((error) => {
      ganttContainer.innerHTML = `<p class="pf-error">Could not load gantt chart: ${esc(error)}</p>`;
    });
  }

  const flameContainer = root.querySelector('#pf-plot-flame');
  if (flame && flameContainer) {
    mountChart(flame, flameContainer, { kind: 'flame' }).catch((error) => {
      flameContainer.innerHTML = `<p class="pf-error">Could not load flame chart: ${esc(error)}</p>`;
    });
  }

  const tableBody = root.querySelector('#pf-region-body');
  const tableFilter = root.querySelector('#pf-filter-table');
  if (stats && tableBody && tableFilter) {
    mountTable(stats, tableBody, tableFilter).catch((error) => {
      tableBody.innerHTML = `<tr><td colspan="6">Could not load region statistics: ${esc(error)}</td></tr>`;
    });
  }
}
