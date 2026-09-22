import scriptSource from '../../examples/ordinary-mode-dispersion.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="ordinary-mode-dispersion.py"',
    },
  });
}
