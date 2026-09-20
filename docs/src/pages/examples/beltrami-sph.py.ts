import scriptSource from '../../examples/beltrami-sph.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="beltrami-sph.py"',
    },
  });
}
