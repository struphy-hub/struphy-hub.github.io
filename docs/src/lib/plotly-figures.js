const CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
};

const mounted = new Set();
let intersectionObserver;
let themeObserver;
let plotlyPromise;
let wideQuery;

function getPlotly() {
  plotlyPromise ??= import('plotly.js-dist-min').then((module) => module.default ?? module);
  return plotlyPromise;
}

function cssColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function axisTheme(axis = {}, colors) {
  return {
    ...axis,
    color: colors.text,
    gridcolor: colors.grid,
    linecolor: colors.grid,
    zerolinecolor: colors.grid,
  };
}

function themedLayout(layout = {}) {
  const colors = {
    text: cssColor('--paper'),
    muted: cssColor('--muted'),
    grid: cssColor('--rule'),
  };
  const next = {
    ...layout,
    autosize: true,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { ...layout.font, color: colors.text },
    legend: {
      ...layout.legend,
      bgcolor: cssColor('--ink'),
      bordercolor: colors.grid,
      font: { ...layout.legend?.font, color: colors.text },
    },
    hoverlabel: {
      ...layout.hoverlabel,
      bgcolor: cssColor('--ink'),
      bordercolor: colors.grid,
      font: { ...layout.hoverlabel?.font, color: colors.text },
    },
  };

  const axisNames = new Set(['xaxis', 'yaxis']);
  for (const name of Object.keys(layout)) {
    if (/^[xyz]axis\d*$/.test(name)) axisNames.add(name);
  }
  for (const name of axisNames) next[name] = axisTheme(layout[name], colors);

  for (const name of Object.keys(layout).filter((key) => /^scene\d*$/.test(key))) {
    next[name] = {
      ...layout[name],
      bgcolor: 'rgba(0,0,0,0)',
      xaxis: axisTheme(layout[name].xaxis, colors),
      yaxis: axisTheme(layout[name].yaxis, colors),
      zaxis: axisTheme(layout[name].zaxis, colors),
    };
  }
  for (const name of Object.keys(layout).filter((key) => /^polar\d*$/.test(key))) {
    next[name] = {
      ...layout[name],
      bgcolor: 'rgba(0,0,0,0)',
      angularaxis: axisTheme(layout[name].angularaxis, colors),
      radialaxis: axisTheme(layout[name].radialaxis, colors),
    };
  }
  return next;
}

function themedRelayout(layout = {}) {
  const colors = {
    text: cssColor('--paper'),
    grid: cssColor('--rule'),
  };
  const update = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    'font.color': colors.text,
    'legend.bgcolor': cssColor('--ink'),
    'legend.bordercolor': colors.grid,
    'legend.font.color': colors.text,
    'hoverlabel.bgcolor': cssColor('--ink'),
    'hoverlabel.bordercolor': colors.grid,
    'hoverlabel.font.color': colors.text,
  };
  const axisNames = new Set(['xaxis', 'yaxis']);
  for (const name of Object.keys(layout)) {
    if (/^[xyz]axis\d*$/.test(name)) axisNames.add(name);
  }
  for (const name of axisNames) {
    update[`${name}.color`] = colors.text;
    update[`${name}.gridcolor`] = colors.grid;
    update[`${name}.linecolor`] = colors.grid;
    update[`${name}.zerolinecolor`] = colors.grid;
  }
  for (const scene of Object.keys(layout).filter((key) => /^scene\d*$/.test(key))) {
    update[`${scene}.bgcolor`] = 'rgba(0,0,0,0)';
    for (const axis of ['xaxis', 'yaxis', 'zaxis']) {
      update[`${scene}.${axis}.color`] = colors.text;
      update[`${scene}.${axis}.gridcolor`] = colors.grid;
      update[`${scene}.${axis}.linecolor`] = colors.grid;
      update[`${scene}.${axis}.zerolinecolor`] = colors.grid;
    }
  }
  for (const polar of Object.keys(layout).filter((key) => /^polar\d*$/.test(key))) {
    update[`${polar}.bgcolor`] = 'rgba(0,0,0,0)';
    for (const axis of ['angularaxis', 'radialaxis']) {
      update[`${polar}.${axis}.color`] = colors.text;
      update[`${polar}.${axis}.gridcolor`] = colors.grid;
      update[`${polar}.${axis}.linecolor`] = colors.grid;
    }
  }
  return update;
}

async function render(root) {
  if (root.dataset.plotMounted === 'true') return;
  root.dataset.plotMounted = 'true';
  try {
    const [Plotly, response] = await Promise.all([getPlotly(), fetch(root.dataset.plotSrc)]);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const figure = await response.json();
    root._plotlySourceLayout = figure.layout ?? {};
    await Plotly.newPlot(root, figure.data ?? [], themedLayout(root._plotlySourceLayout), CONFIG);
    if (figure.frames?.length) await Plotly.addFrames(root, figure.frames);
    root.setAttribute('aria-busy', 'false');
    mounted.add(root);
  } catch (error) {
    console.error(`Unable to load Plotly figure ${root.dataset.plotSrc}:`, error);
    root.textContent = 'Interactive figure unavailable.';
    root.classList.add('plot-error');
    root.setAttribute('aria-busy', 'false');
    root.dataset.plotMounted = 'error';
  }
}

function ensureObservers() {
  if (!intersectionObserver) {
    intersectionObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        intersectionObserver.unobserve(entry.target);
        render(entry.target);
      }
    }, { rootMargin: '500px 0px' });
  }
  if (!themeObserver) {
    themeObserver = new MutationObserver(async () => {
      if (!mounted.size) return;
      const Plotly = await getPlotly();
      for (const root of mounted) {
        if (!document.contains(root)) {
          mounted.delete(root);
          continue;
        }
        Plotly.relayout(root, themedRelayout(root._plotlySourceLayout));
      }
    });
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
}

export function mountPlotlyFigures() {
  if (typeof document === 'undefined') return;
  if (!wideQuery) {
    wideQuery = window.matchMedia('(min-width: 721px)');
    wideQuery.addEventListener('change', (event) => {
      if (event.matches) mountPlotlyFigures();
    });
  }
  if (!wideQuery.matches) return;
  ensureObservers();
  for (const root of document.querySelectorAll('[data-plotly-figure]:not([data-plot-mounted])')) {
    if (root.dataset.eager === 'true') render(root);
    else intersectionObserver.observe(root);
  }
}
