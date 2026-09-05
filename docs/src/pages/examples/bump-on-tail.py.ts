import bumpOnTail from '../../examples/bump-on-tail.py?raw';

export function GET() {
  return new Response(bumpOnTail, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="bump-on-tail.py"',
    },
  });
}
