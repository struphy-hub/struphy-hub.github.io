import mhdSlabWaves from '../../examples/mhd-slab-waves.py?raw';

export function GET() {
  return new Response(mhdSlabWaves, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="mhd-slab-waves.py"',
    },
  });
}
