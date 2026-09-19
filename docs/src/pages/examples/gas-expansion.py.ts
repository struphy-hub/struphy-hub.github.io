import gasExpansion from '../../examples/gas-expansion.py?raw';

export function GET() {
  return new Response(gasExpansion, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="gas-expansion.py"',
    },
  });
}
