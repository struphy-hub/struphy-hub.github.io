import zeldovichCaustic from '../../examples/zeldovich-caustic.py?raw';

export function GET() {
  return new Response(zeldovichCaustic, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="zeldovich-caustic.py"',
    },
  });
}
