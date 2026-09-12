import hybridAlfvenIonCoupling from '../../examples/hybrid-alfven-ion-coupling.py?raw';

export function GET() {
  return new Response(hybridAlfvenIonCoupling, {
    headers: {
      'Content-Type': 'text/x-python; charset=utf-8',
      'Content-Disposition': 'attachment; filename="hybrid-alfven-ion-coupling.py"',
    },
  });
}
