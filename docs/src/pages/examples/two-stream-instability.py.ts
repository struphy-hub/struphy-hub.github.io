import twoStreamInstability from '../../examples/two-stream-instability.py?raw';

export function GET() {
  return new Response(twoStreamInstability, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="two-stream-instability.py"',
    },
  });
}
