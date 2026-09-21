import scriptSource from '../../examples/maxwell-cavity-resonances.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="maxwell-cavity-resonances.py"',
    },
  });
}
