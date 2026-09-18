import scriptSource from '../../examples/orszag-tang-vortex.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="orszag-tang-vortex.py"',
    },
  });
}
