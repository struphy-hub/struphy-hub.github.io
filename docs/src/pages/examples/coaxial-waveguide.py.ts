import gasExpansion from '../../examples/coaxial-waveguide.py?raw';

export function GET() {
  return new Response(gasExpansion, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="coaxial-waveguide.py"',
    },
  });
}
