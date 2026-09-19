import diffusionMethods from '../../examples/diffusion-methods.py?raw';

export function GET() {
  return new Response(diffusionMethods, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="diffusion-methods.py"',
    },
  });
}
