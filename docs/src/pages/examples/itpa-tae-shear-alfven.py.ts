import scriptSource from '../../examples/itpa-tae-shear-alfven.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="itpa-tae-shear-alfven.py"',
    },
  });
}
