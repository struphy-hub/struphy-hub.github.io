import shearAlfvenWave from '../../examples/shear-alfven-wave.py?raw';

export function GET() {
  return new Response(shearAlfvenWave, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="shear-alfven-wave.py"',
    },
  });
}
