import incompressibleShearRelaxation from '../../examples/incompressible-shear-relaxation.py?raw';

export function GET() {
  return new Response(incompressibleShearRelaxation, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="incompressible-shear-relaxation.py"',
    },
  });
}
