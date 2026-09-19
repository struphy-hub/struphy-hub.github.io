import scriptSource from '../../examples/hall-mhd-waves.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="hall-mhd-waves.py"',
    },
  });
}
