import scriptSource from '../../examples/damped-alfven-wave.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="damped-alfven-wave.py"',
    },
  });
}
