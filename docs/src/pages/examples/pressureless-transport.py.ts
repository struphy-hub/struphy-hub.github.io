import scriptSource from '../../examples/pressureless-transport.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="pressureless-transport.py"',
    },
  });
}
