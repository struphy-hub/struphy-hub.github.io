import hasegawaWakatani from '../../examples/hasegawa-wakatani.py?raw';

export function GET() {
  return new Response(hasegawaWakatani, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="hasegawa-wakatani.py"',
    },
  });
}
