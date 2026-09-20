import scriptSource from '../../examples/alfven-standing-wave.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="alfven-standing-wave.py"',
    },
  });
}
