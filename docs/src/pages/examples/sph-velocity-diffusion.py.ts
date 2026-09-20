import scriptSource from '../../examples/sph-velocity-diffusion.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="sph-velocity-diffusion.py"',
    },
  });
}
