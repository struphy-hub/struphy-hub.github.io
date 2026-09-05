import weakLandauDamping from '../../examples/weak-landau-damping.py?raw';

export function GET() {
  return new Response(weakLandauDamping, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="weak-landau-damping.py"',
    },
  });
}
