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

function themedUpdatemenus(updatemenus = [], colors) {
  return updatemenus.map((menu) => ({
    ...menu,
    bgcolor: colors.background,
    bordercolor: colors.accent,
    borderwidth: 1,
    font: { ...menu.font, color: colors.text, size: Math.max(menu.font?.size ?? 0, 14) },
    pad: { ...menu.pad, l: Math.max(menu.pad?.l ?? 0, 8), r: Math.max(menu.pad?.r ?? 0, 8), t: Math.max(menu.pad?.t ?? 0, 6), b: Math.max(menu.pad?.b ?? 0, 6) },
    buttons: menu.buttons?.map((button) => {
      if (button.label === 'Play') return { ...button, label: '▶ Play' };
      if (button.label === 'Pause') return { ...button, label: '⏸ Pause' };
      return button;
    }),
  }));
}

function themedLayout(layout = {}) {
  const colors = {
    text: cssColor('--paper'),
    muted: cssColor('--muted'),
    grid: cssColor('--rule'),
    background: cssColor('--ink'),
    accent: cssColor('--cyan'),
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
    updatemenus: themedUpdatemenus(layout.updatemenus, colors),
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

function setupPlayButton(root, Plotly, hasFrames) {
  if (!hasFrames) return;
  let button = root.querySelector('[data-plot-play]');
  if (!button) {
    button = document.createElement('button');
    button.className = 'plot-play';
    button.dataset.plotPlay = '';
    button.type = 'button';
    button.ariaLabel = 'Play animation';
    button.innerHTML = '<span aria-hidden="true">▶</span> <span data-plot-play-label>Play</span>';
    root.append(button);
  }
  const label = button.querySelector('[data-plot-play-label]');
  if (!label) return;
  button.hidden = false;
  button.addEventListener('click', async () => {
    const playing = button.dataset.playing === 'true';
    if (playing) {
      await Plotly.animate(root, [null], { frame: { duration: 0, redraw: false }, mode: 'immediate' });
      button.dataset.playing = 'false';
      button.setAttribute('aria-label', 'Play animation');
      label.textContent = 'Play';
      button.querySelector('span')?.replaceChildren(document.createTextNode('▶'));
      return;
    }
    await Plotly.animate(root, null, { frame: { duration: 45, redraw: true }, transition: { duration: 0 }, fromcurrent: true });
    button.dataset.playing = 'true';
    button.setAttribute('aria-label', 'Pause animation');
    label.textContent = 'Pause';
    button.querySelector('span')?.replaceChildren(document.createTextNode('Ⅱ'));
  });
}

function themedRelayout(layout = {}) {
  const colors = {
    text: cssColor('--paper'),
    grid: cssColor('--rule'),
    background: cssColor('--ink'),
    accent: cssColor('--cyan'),
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
  if (layout.updatemenus?.length) update.updatemenus = themedUpdatemenus(layout.updatemenus, colors);
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
    root._plotlyHasFrames = Boolean(figure.frames?.length);
    root.querySelector('[data-plot-status]')?.remove();
    const layout = themedLayout(root._plotlySourceLayout);
    if (root._plotlyHasFrames) layout.updatemenus = [];
    await Plotly.newPlot(root, figure.data ?? [], layout, CONFIG);
    if (figure.frames?.length) {
      await Plotly.addFrames(root, figure.frames);
      setupPlayButton(root, Plotly, true);
    }
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
        const update = themedRelayout(root._plotlySourceLayout);
        if (root._plotlyHasFrames) update.updatemenus = [];
        Plotly.relayout(root, update);
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
