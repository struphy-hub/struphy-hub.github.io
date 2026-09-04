import maxwellWave from '../../examples/maxwell-wave.py?raw';

export function GET() {
  return new Response(maxwellWave, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="maxwell-wave.py"',
    },
  });
}
