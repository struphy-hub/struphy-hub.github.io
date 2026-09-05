import weibelInstability from '../../examples/weibel-instability.py?raw';

export function GET() {
  return new Response(weibelInstability, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="weibel-instability.py"',
    },
  });
}
