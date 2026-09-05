import itgDriftWave from '../../examples/itg-drift-wave.py?raw';

export function GET() {
  return new Response(itgDriftWave, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="itg-drift-wave.py"',
    },
  });
}
