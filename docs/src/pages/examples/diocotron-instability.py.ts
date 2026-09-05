import diocotronInstability from '../../examples/diocotron-instability.py?raw';

export function GET() {
  return new Response(diocotronInstability, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="diocotron-instability.py"',
    },
  });
}
