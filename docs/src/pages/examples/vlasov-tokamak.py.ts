import vlasovTokamak from '../../examples/vlasov-tokamak.py?raw';

export function GET() {
  return new Response(vlasovTokamak, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="vlasov-tokamak.py"',
    },
  });
}
