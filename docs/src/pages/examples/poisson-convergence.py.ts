import scriptSource from '../../examples/poisson-convergence.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="poisson-convergence.py"',
    },
  });
}
