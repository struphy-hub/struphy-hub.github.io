import poissonSource from '../../examples/poisson-source.py?raw';

export function GET() {
  return new Response(poissonSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="poisson-source.py"',
    },
  });
}
