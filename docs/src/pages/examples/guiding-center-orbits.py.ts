import gasExpansion from '../../examples/guiding-center-orbits.py?raw';

export function GET() {
  return new Response(gasExpansion, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="guiding-center-orbits.py"',
    },
  });
}
