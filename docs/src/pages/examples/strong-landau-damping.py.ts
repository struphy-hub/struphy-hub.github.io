import strongLandauDamping from '../../examples/strong-landau-damping.py?raw';

export function GET() {
  return new Response(strongLandauDamping, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="strong-landau-damping.py"',
    },
  });
}
