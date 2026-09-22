import scriptSource from '../../examples/grad-b-drift.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="grad-b-drift.py"',
    },
  });
}
