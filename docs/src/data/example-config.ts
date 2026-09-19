export interface ExamplePageConfig {
  category: string;
  setupTitle: string;
  plotTitle: string;
  plotAlt: string;
}

/** Small, hand-written presentation layer for the generated example results. */
export const exampleConfig: Record<string, ExamplePageConfig> = {
  'bump-on-tail': { category: 'Kinetic electrostatics', setupTitle: 'A minority beam driving Langmuir waves', plotTitle: 'Electric field energy growth in the bump-on-tail instability', plotAlt: 'Electric field energy growing unsteadily over time as the bump-on-tail instability develops' },
  'coaxial-waveguide': { category: 'Electromagnetism', setupTitle: 'A rotating mode between conducting cylinders', plotTitle: 'Axial magnetic field and error against the exact coaxial waveguide mode', plotAlt: 'Axial magnetic field and error against the exact coaxial waveguide mode' },
  'dam-break': { category: 'Free-surface flow', setupTitle: 'A fluid column released in a closed box', plotTitle: 'Animated dam break simulated with SPH', plotAlt: 'SPH markers and their density estimate during the collapse of a fluid column' },
  'diocotron-instability': { category: 'Kinetic drift dynamics', setupTitle: 'Shear instability in a rotating charged ring', plotTitle: "Animated diocotron instability ring density", plotAlt: 'Density of a charged ring developing a rippled pattern as the diocotron instability grows' },
  'gas-expansion': { category: 'Gas dynamics', setupTitle: 'A gas released into vacuum', plotTitle: 'Animated isothermal gas expansion into vacuum', plotAlt: 'SPH density estimate and marker velocities of an expanding gas' },
  'guiding-center-orbits': { category: 'Particle orbits', setupTitle: 'Passing and trapped guiding centers', plotTitle: 'Animated guiding-center trajectories in a circular tokamak', plotAlt: 'Guiding-center trajectories in the poloidal plane of a circular tokamak' },
  'hybrid-alfven-ion-coupling': { category: 'Hybrid kinetic-MHD', setupTitle: 'Two-way kinetic-MHD coupling', plotTitle: 'Energy exchange between energetic ions and the Alfvén wave', plotAlt: 'Field and fluid energy oscillating as they exchange energy with the energetic ion population' },
  'itg-drift-wave': { category: 'Drift-kinetic turbulence', setupTitle: 'Drift waves from a temperature gradient', plotTitle: 'ITG density perturbation energy growth', plotAlt: 'Density perturbation energy growing exponentially as the ITG drift wave develops' },
  'maxwell-wave': { category: 'Electromagnetic waves', setupTitle: 'A broadband vacuum light wave', plotTitle: 'Power spectrum of the Struphy Maxwell simulation', plotAlt: 'Power spectrum of a simulated Struphy Maxwell simulation' },
  'orszag-tang-vortex': { category: 'Nonlinear MHD', setupTitle: 'Nonlinear evolution of crossed vortices', plotTitle: 'Density and magnetic field lines in the Orszag–Tang vortex', plotAlt: 'Density and magnetic field lines in the Orszag–Tang vortex' },
  'poisson-source': { category: 'Electrostatics', setupTitle: 'A driven electrostatic potential', plotTitle: 'Poisson potential compared with its exact solution', plotAlt: "Struphy's FEEC Poisson potential closely tracking the exact cosine-mode solution" },
  'shear-alfven-wave': { category: 'MHD waves', setupTitle: 'A broadband shear-Alfvén wave', plotTitle: 'Power spectrum of the Struphy shear-Alfvén simulation', plotAlt: 'Power spectrum of a simulated Struphy shear-Alfvén wave' },
  'strong-landau-damping': { category: 'Kinetic electrostatics', setupTitle: 'Particle trapping in a large-amplitude wave', plotTitle: 'Electric field energy in strong Landau damping', plotAlt: 'Electric field energy damping and bouncing as trapped particles oscillate' },
  'two-stream-instability': { category: 'Kinetic electrostatics', setupTitle: 'Free energy in counter-streaming beams', plotTitle: 'Electric field energy growth in the two-stream instability', plotAlt: 'Electric field energy growing exponentially before saturating' },
  'vlasov-tokamak': { category: 'Kinetic particle orbits', setupTitle: 'Test particles in a fixed tokamak field', plotTitle: 'Full-orbit particle trajectories in a tokamak', plotAlt: "Full-orbit particle trajectories gyrating around a tokamak's torus" },
  'vortex-merger': { category: 'Electrostatic drift', setupTitle: 'Two charge blobs in an annulus', plotTitle: 'Animated charge density showing two blobs winding into one core', plotAlt: 'Animated charge density showing two blobs winding into one core' },
  'weak-landau-damping': { category: 'Kinetic electrostatics', setupTitle: 'Phase mixing in a uniform plasma', plotTitle: 'Electric field energy decay in weak Landau damping', plotAlt: 'Electric field energy decaying exponentially over time' },
  'weibel-instability': { category: 'Kinetic electromagnetics', setupTitle: 'Spontaneous field generation from temperature anisotropy', plotTitle: 'Magnetic field energy growth in the Weibel instability', plotAlt: 'Magnetic field energy growing exponentially as the Weibel instability develops' },
  'mhd-slab-waves': { category: 'MHD waves', setupTitle: 'Three waves from one noise spectrum', plotTitle: 'Power spectra of velocity and pressure MHD waves in a magnetized slab', plotAlt: 'Power spectra of velocity and pressure MHD waves in a magnetized slab' },
  'incompressible-shear-relaxation': { category: 'Incompressible flow', setupTitle: 'Removing compression, keeping the shear', plotTitle: 'Shear flow and compressive wave between no-slip walls', plotAlt: 'Shear flow and compressive wave of an incompressible fluid between no-slip walls' },
  'zeldovich-caustic': { category: 'Pressureless flow', setupTitle: 'A gas that falls onto itself', plotTitle: "Density and phase space of a pressureless Zel'dovich collapse", plotAlt: "Density and phase space of a pressureless Zel'dovich collapse" },
  'diffusion-methods': { category: 'Diffusion', setupTitle: 'Diffusion with random and deterministic particles', plotTitle: 'Density mode diffusion by random-walk and deterministic particles', plotAlt: 'Density mode diffusing by random-walk and deterministic particle methods' },
};
