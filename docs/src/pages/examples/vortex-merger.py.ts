import gasExpansion from '../../examples/vortex-merger.py?raw';

export function GET() {
  return new Response(gasExpansion, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="vortex-merger.py"',
    },
  });
}
