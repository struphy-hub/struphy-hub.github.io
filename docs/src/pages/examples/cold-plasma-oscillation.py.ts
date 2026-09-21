import scriptSource from '../../examples/cold-plasma-oscillation.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="cold-plasma-oscillation.py"',
    },
  });
}
