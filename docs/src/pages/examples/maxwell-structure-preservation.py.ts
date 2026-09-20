import scriptSource from '../../examples/maxwell-structure-preservation.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="maxwell-structure-preservation.py"',
    },
  });
}
