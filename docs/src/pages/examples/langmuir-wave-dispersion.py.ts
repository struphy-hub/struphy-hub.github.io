import scriptSource from '../../examples/langmuir-wave-dispersion.py?raw';

export function GET() {
  return new Response(scriptSource, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="langmuir-wave-dispersion.py"',
    },
  });
}
