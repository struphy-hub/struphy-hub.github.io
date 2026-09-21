import scriptSource from '../../examples/cold-plasma-wave-packet.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="cold-plasma-wave-packet.py"',
    },
  });
}
