/** Page texts of an additional figure, which the example's `pproc` returns under this key. */
export interface ExampleFigureText {
  alt: string;
  caption: string;
  /** The figure's width/height, e.g. '1500/560', if it is not the default 1100/650. */
  aspect?: string;
}

export interface ExamplePageConfig {
  category: string;
  setupTitle: string;
  plotTitle: string;
  plotAlt: string;
  /** The additional figures, `<slug>-<key>.*`, in page order: their alt texts and captions, by key. */
  figures?: Record<string, ExampleFigureText>;
  /** The Struphy tutorial the example is adapted from. */
  tutorial?: string;
}

/** Small, hand-written presentation layer for the generated example results. */
export const exampleConfig: Record<string, ExamplePageConfig> = {
  'toroidal-shear-alfven': {
    category: 'MHD waves', setupTitle: 'Coupled poloidal modes in a circular tokamak', plotTitle: 'Physical velocity components on a poloidal slice', plotAlt: 'Animated minor-radial, poloidal and toroidal velocity in a hollow tokamak cross-section',
    figures: {
      'velocity-history': {
        alt: 'Time histories of three physical velocity components around a poloidal ring',
        caption: 'Physical velocity at r=0.550, φ=0, over the complete run. The angular structure starts with the m=10,11 perturbations. Each panel uses the same component color range as the slice animation; interpolated display pixels do not add simulation resolution.',
      },
      'radial-profiles-theta-0': {
        alt: 'Radial profiles of three physical velocity components at theta=0 degrees on the phi zero plane',
        caption: 'Signed physical velocity along a radial ray at θ=0°, φ=0, from the inner to the outer boundary. Colors identify the saved times; the dotted line marks the initial Gaussian center r=0.550. Markers are spline evaluation points, not additional simulation cells. Velocities use normalized units.',
      },
      'radial-profiles-theta-45': {
        alt: 'Radial profiles of three physical velocity components at theta=45 degrees on the phi zero plane',
        caption: 'Signed physical velocity along a radial ray at θ=45°, φ=0, from the inner to the outer boundary. Colors identify the saved times; the dotted line marks the initial Gaussian center r=0.550. Markers are spline evaluation points, not additional simulation cells. Velocities use normalized units.',
      },
      'radial-history': {
        alt: 'Radius–time maps of the poloidal RMS of each physical velocity component',
        caption: 'The root-mean-square velocity over poloidal angle at each radius and time: sqrt(〈u²〉θ), evaluated on the φ=0 slice. This angular average shows radial localization without cancellation between positive and negative wave lobes; it is not a volume or flux-surface average. Each component has its own fixed color range in normalized velocity units. The white dotted line marks the initial center r=0.550; the underlying run still uses only 8 radial elements.',
      },
      'poloidal-fft': {
        alt: 'Initial logical radial velocity Fourier amplitudes with the seeded m=10 and m=11 modes',
        caption: 'Poloidal FFT of the initial logical H(div) radial velocity at φ=0. The duplicate periodic endpoint is excluded. Positive-mode amplitudes are 2|FFT|/N, followed by RMS over sampled radii, with no volume weighting. Dotted lines identify the seeded m=10,11 modes.',
      },
      'radial-fft-velocity': {
        alt: 'Normalized radial Fourier amplitudes of u_r for m=9,10,11,12 at fixed toroidal mode',
        caption: 'Physical u_r at the final saved time. A two-dimensional spatial FFT in poloidal and toroidal angle selects sector mode −1 (full-torus |n|=6) and m=9,10,11,12. Both duplicate periodic endpoints are excluded. All four amplitude curves share one normalization: the largest amplitude over these harmonics and all sampled radii, separately for velocity and magnetic perturbation. This preserves relative harmonic strengths; the magnetic field excludes the equilibrium. Markers are spline evaluation points. Inspired by Figure 5 of arXiv:2510.04385 (https://arxiv.org/abs/2510.04385); this is the present LinearMHD run, with no comparison between filtered and unfiltered kinetic simulations and no temporal FFT selection.',
      },
      'radial-fft-magnetic': {
        alt: 'Normalized radial Fourier amplitudes of δB_r for m=9,10,11,12 at fixed toroidal mode',
        caption: 'Physical δB_r at the final saved time. A two-dimensional spatial FFT in poloidal and toroidal angle selects sector mode −1 (full-torus |n|=6) and m=9,10,11,12. Both duplicate periodic endpoints are excluded. All four amplitude curves share one normalization: the largest amplitude over these harmonics and all sampled radii, separately for velocity and magnetic perturbation. This preserves relative harmonic strengths; the magnetic field excludes the equilibrium. Markers are spline evaluation points. Inspired by Figure 5 of arXiv:2510.04385 (https://arxiv.org/abs/2510.04385); this is the present LinearMHD run, with no comparison between filtered and unfiltered kinetic simulations and no temporal FFT selection.',
      },
      'fixed-theta-fft-velocity': {
        alt: 'Normalized toroidal FFT amplitude of u_r versus radius at theta zero and 45 degrees',
        caption: 'Physical u_r at θ=0° and θ=45°, at the final saved time. The FFT is taken only in toroidal angle, selecting sector mode −1 (full-torus |n|=6). All poloidal harmonics contribute coherently at each angle; there is no poloidal or time FFT. The duplicate toroidal endpoint is excluded. Both curves share the same maximum-amplitude normalization over the two angles and all radii, separately for each field. The magnetic field is the perturbation. The requested angles lie on the default evaluation grid; other grids use linear interpolation of the physical radial field where needed.',
      },
      'fixed-theta-fft-magnetic': {
        alt: 'Normalized toroidal FFT amplitude of δB_r versus radius at theta zero and 45 degrees',
        caption: 'Physical δB_r at θ=0° and θ=45°, at the final saved time. The FFT is taken only in toroidal angle, selecting sector mode −1 (full-torus |n|=6). All poloidal harmonics contribute coherently at each angle; there is no poloidal or time FFT. The duplicate toroidal endpoint is excluded. Both curves share the same maximum-amplitude normalization over the two angles and all radii, separately for each field. The magnetic field is the perturbation. The requested angles lie on the default evaluation grid; other grids use linear interpolation of the physical radial field where needed.',
      },
      'time-fft': {
        alt: 'Temporal spectra of three physical velocity components with selected dominant frequency bands',
        caption: 'One-sided power per frequency bin from the signed velocity, averaged over sampled points in the φ=0 plane. DC is omitted and each panel is normalized to its own largest nonzero-frequency bin. Orange shading shows the contiguous half-power band used for reconstruction (no padding). For N saved samples at spacing Δt, the frequency resolution is Δω=2π/(NΔt) and the Nyquist frequency is π/Δt. This short, untapered record has coarse frequency resolution and spectral leakage; its largest bin is not a converged TAE frequency.',
      },
      'radial-time-fft': {
        alt: 'Temporal velocity power as a function of minor radius and angular frequency',
        caption: 'Time FFT at each spatial point, followed by poloidal averaging of the power. Transforming the signed components before squaring avoids the frequency doubling of quadratic diagnostics. Colors show log₁₀(P/Pmax) separately for each component, clipped at −6; DC is omitted. The angular average is not weighted by physical volume. Only the actual frequency bins are displayed.',
      },
      'filtered-velocity': {
        alt: 'Original and dominant-band-filtered velocity traces at a probe on the poloidal slice',
        caption: 'Original and reconstructed physical velocity at r=0.550, θ=45°, φ=0. Each component\'s band is chosen from power summed over the whole sampled poloidal plane, then applied at every point before the inverse time FFT. DC and other bins are removed. This is a finite-record band-pass diagnostic, not an exact eigenmode; leakage and endpoint ringing remain possible.',
      },
      'energy': {
        alt: 'Kinetic, magnetic and compressional perturbation energies over time',
        caption: 'Volume-integrated quadratic perturbation energies from LinearMHD. These exclude the equilibrium magnetic and thermal energies. They show the response of both the shear-Alfvén and magnetosonic propagators in the nonuniform toroidal equilibrium; this short coarse run does not establish a converged TAE frequency or growth rate.',
      },
    },
  },
  'ordinary-mode-dispersion': {
    category: 'Plasma waves', setupTitle: 'Electromagnetic waves above the plasma cutoff', plotTitle: 'Ordinary-mode frequencies and the plasma cutoff', plotAlt: 'Four measured ordinary-wave frequencies on the cold-plasma dispersion curve, with their electric-field oscillations',
    figures: {
      'space-time': {
        alt: 'Space-time map of four superposed ordinary electromagnetic waves',
        caption: 'The longer waves oscillate near the plasma frequency; shorter waves oscillate faster.',
      },
    },
  },
  'faraday-rotation': {
    category: 'Plasma wave polarization', setupTitle: 'Two circular waves rotate a linear polarization', plotTitle: 'Faraday rotation along a magnetized plasma', plotAlt: 'Animated three-dimensional electric-field wave train alongside the measured rotation of its polarization axis',
    figures: {
      'polarization': {
        alt: 'Electric-field hodographs forming lines at three different angles',
        caption: 'At each position the electric vector traces a line. Its orientation changes with distance along the magnetic field.',
      },
    },
  },
  'grad-b-drift': {
    category: 'Particle drifts', setupTitle: 'Gyration in a magnetic-field gradient', plotTitle: 'Grad-B drift of full-orbit test ions', plotAlt: 'Three gyrating ions drifting across a magnetic-field gradient, with drift velocities compared against perpendicular energy',
    figures: {
      'drift': {
        alt: 'Approximate guiding centers drifting steadily across the magnetic field',
        caption: 'Subtracting the leading gyration reveals the slow drift. Small oscillations remain because the guiding-center formula neglects finite-Larmor-radius corrections.',
      },
    },
  },
  'linear-dissipative-alfven-wave': {
    category: 'Dissipative MHD', setupTitle: 'A standing wave with viscosity and resistivity', plotTitle: 'Damped Alfvén modes and wave-energy decay', plotAlt: 'Velocity and magnetic modes oscillating inside an exponential envelope with the expected decay of wave energy',
    figures: {
      'space-time': {
        alt: 'Standing Alfvén wave fading under viscosity and resistivity',
        caption: 'The nodes remain fixed while viscosity and resistivity damp the wave.',
      },
    },
  },
  'pressureless-transport': {
    category: 'Pressureless flow', setupTitle: 'A density ripple carried at constant speed', plotTitle: 'Exact transport around a periodic box', plotAlt: 'Transported density profiles compared with the exact solution, with profile, mass and kinetic-energy errors',
    figures: {
      'space-time': {
        alt: 'Density ripple translating at constant speed',
        caption: 'The diagonal bands travel at speed 0.5 and wrap through the periodic boundary.',
      },
    },
  },
  'beltrami-sph': {
    category: 'Pressureless particle flow', setupTitle: 'Markers following a stationary cellular flow', plotTitle: 'SPH markers circulating on Beltrami streamlines', plotAlt: 'Pressureless SPH markers circulating along exact streamlines beside velocity and energy error histories',
    tutorial: 'https://struphy-hub.github.io/struphy/_collections/tutorials/tutorial_beltrami_sph.html',
    figures: {
      'density': {
        alt: 'Animated SPH density rho in the stationary Beltrami flow',
        caption: 'Kernel-reconstructed SPH mass density ρ. The exact divergence-free Beltrami flow preserves the initially uniform ρ = 1; the visible variation measures finite-particle and kernel-reconstruction error. Drag the slider or press Play to follow the evolution.',
      },
      'compression': {
        alt: 'Animated compression and rarefaction field rho minus one',
        caption: 'The signed density perturbation ρ − 1 separates compression (red) from rarefaction (blue). The exact solution is zero everywhere. To keep the bulk structure visible, the colour scale is clipped symmetrically at the 99th percentile; hover values remain unclipped.',
      },
      'area-deformation': {
        alt: 'Animated Lagrangian marker-cell area ratio',
        caption: 'Signed area A(t)/A(0) of each cell in the original marker tessellation, displayed at its initial position. Exact incompressible transport keeps the ratio at one. The colour scale is clipped symmetrically at the 99th percentile; hover values remain unclipped.',
      },
      'trajectories': {
        alt: 'Selected SPH marker paths over exact Beltrami streamlines',
        caption: 'Six markers starting near the positive x-axis sample nested streamline families. Their full numerical paths are drawn over the exact streamlines; circles mark the starts and crosses their positions at tmax.',
      },
    },
  },
  'bump-on-tail': {
    category: 'Kinetic electrostatics', setupTitle: 'A minority beam driving Langmuir waves', plotTitle: 'Electric field energy growth in the bump-on-tail instability', plotAlt: 'Electric field energy growing unsteadily over time as the bump-on-tail instability develops',
    figures: {
      'phasespace': {
        alt: 'Phase-space density of the bump-on-tail instability',
        caption: 'The phase-space density f(x, v) of the run above, resolved on 64 spatial cells and displayed in 128 × 128 bins; drag the slider or press Play. Initially the bulk (v ≈ 3) and the bump (v ≈ −4.5) are almost uniform in x. The frame shown is from the middle of the run, where both populations have developed strong structure in x and spread far beyond their initial velocity widths.',
      },
      'velocity-time': {
        alt: 'Space-averaged velocity distribution as a function of time',
        caption: 'The distribution averaged over space, f(v, t) (the mean of the binned f over the 64 cells). The bulk (v ≈ 3, peak f ≈ 0.36) and the much smaller bump (v ≈ −4.5, peak f ≈ 0.08) start as separate populations. As the wave grows, both broaden and the region between them fills in.',
      },
    },
  },
  'coaxial-waveguide': {
    category: 'Electromagnetism', setupTitle: 'A rotating mode between conducting cylinders', plotTitle: 'Axial magnetic field and error against the exact coaxial waveguide mode', plotAlt: 'Axial magnetic field and error against the exact coaxial waveguide mode',
    figures: {
      'frequency': {
        alt: 'Axial magnetic field at a probe against the exact coaxial mode',
        caption: 'The axial magnetic field at a fixed probe in the middle of the gap, from Struphy and from the exact mode, which is proportional to cos(3θ − t). A sinusoid fitted to the Struphy signal measures the mode\'s frequency against the exact value 1; the small phase drift it reveals is the source of the field error of the animation.',
      },
      'energy': {
        alt: 'Electric, magnetic and total energy of the coaxial mode against time',
        caption: 'The electric and magnetic energies of the rotating mode are equal and constant to round-off, since the energy density pattern rotates with the mode, and so is their sum. The relative change of the total energy, below, stays tiny over all time steps: the implicit time integrator conserves the energy to round-off.',
      },
    },
  },
  'dam-break': {
    category: 'Free-surface flow', setupTitle: 'A fluid column released in a closed box', plotTitle: 'Animated dam break simulated with SPH', plotAlt: 'SPH markers and their density estimate during the collapse of a fluid column',
    figures: {
      'front': {
        alt: 'Position of the fluid front and height of the centre of mass over time',
        caption: 'The front of the fluid (the largest marker x) and the height of its centre of mass. The front reaches the far wall early in the collapse and stays there. The centre of mass falls from 0.5 to about 0.15 by t ≈ 0.4, close to the free-fall time of 0.32, rises slightly as the fluid rebounds, and then settles slowly towards a layer at the bottom.',
      },
    },
  },
  'diocotron-instability': {
    category: 'Kinetic drift dynamics', setupTitle: 'Shear instability in a rotating charged ring', plotTitle: "Animated diocotron instability ring density", plotAlt: 'Density of a charged ring developing a rippled pattern as the diocotron instability grows',
    figures: {
      'mode-growth': {
        alt: 'Growth of the diocotron instability\'s azimuthal modes',
        caption: 'The seeded m = 4 perturbation grows above the other azimuthal modes. The dashed line is an exponential fit over the linear-growth interval, providing a quantitative companion to the animated density.',
      },
      'ring-interfaces': {
        alt: 'Inner and outer diocotron ring interfaces evolving in physical space',
        caption: 'The inner and outer density interfaces plotted in physical x-y space. Their four-lobed distortion reveals the seeded diocotron mode more directly than the radial-angular density map; drag the slider or press Play to follow the rotation.',
      },
    },
  },
  'gas-expansion': {
    category: 'Gas dynamics', setupTitle: 'A gas released into vacuum', plotTitle: 'Animated isothermal gas expansion into vacuum', plotAlt: 'SPH density estimate and marker velocities of an expanding gas',
    figures: {
      'particles': {
        alt: 'Animation of SPH particles expanding into vacuum, colored by velocity',
        caption: 'The 768 SPH particles move from the initially filled region into the vacuum; color shows velocity. The dashed line marks the initial gas edge. Vertical lanes only separate the particles visually: this simulation is one-dimensional. Press Play, Pause, or drag the time slider.',
      },
      'similarity': {
        alt: 'Density and velocity against the similarity variable at four times, on the exact curve',
        caption: 'The density (top) and the marker velocities (bottom) at four times against the similarity variable (x − x₀) / t, with the exact solution dashed. From ξ = −1, where the rarefaction starts, up to ξ ≈ 2 the four times fall on one curve. The earliest time is smoothed around ξ = −1 because the release is smoothed over the kernel width. Beyond ξ ≈ 2, where the density is below about 0.05, only a few markers are left: they move more slowly than the exact solution and their density estimate is noisy.',
      },
      'error': {
        alt: 'Density and velocity error of the SPH result against time',
        caption: 'The error of the SPH result against the exact solution for t ≥ 0.1. The density error is the L1 difference relative to the exact density. The rms error of the marker velocity grows with time, but the median marker is off by much less at the end: the rms is dominated by the few fast markers in the low-density tail.',
      },
    },
  },
  'gvec-equilibrium': {
    category: 'Stellarator particle orbits', setupTitle: 'Guiding centers in a generated GVEC stellarator', plotTitle: 'Guiding-center trajectories on 3D stellarator flux surfaces', plotAlt: 'Six guiding-center trajectories on nested magnetic flux surfaces of a five-field-period GVEC stellarator',
    figures: {
      'flux-surfaces': {
        alt: 'Nested GVEC flux surfaces in a poloidal cut beside pressure and rotational-transform profiles',
        caption: 'A poloidal cut through the nested magnetic surfaces generated by GVEC. The pressure and rotational transform on the right are evaluated directly from the newly generated equilibrium state.',
      },
      'poloidal-orbits': {
        alt: 'One panel per marker showing its guiding-center orbit projected on the poloidal plane',
        caption: 'Each marker\'s orbit projected on the (R, Z) plane. The grey curves are the flux surfaces at five toroidal angles across one field period, whose envelope is the region the projection can reach. The trapped markers stay within a narrow band of flux surfaces and retrace it, while the passing ones sweep the whole cross-section as they circulate. Unlike a tokamak, the stellarator cross-section rotates with the toroidal angle, so these projections are not closed banana curves: the width of the band is the radial excursion, the filling-in is the toroidal motion.',
      },
      'orbit-diagnostics': {
        alt: 'Parallel velocities and total-energy conservation of guiding-center orbits in the GVEC stellarator',
        caption: 'A sign change in parallel velocity identifies a magnetic-mirror reflection: the markers launched with a small parallel velocity are caught in the helical wells of the stellarator, while the faster ones circulate. The magnetic moment is a coordinate of the model and is conserved exactly; the total energy drifts by a few percent, which is set by the accuracy of the coarse FEEC projection of the GVEC field rather than by the time integrator — the drift does not fall when the time step is reduced.',
      },
    },
  },
  'guiding-center-orbits': {
    category: 'Particle orbits', setupTitle: 'Passing and trapped guiding centers', plotTitle: 'Animated guiding-center trajectories in a circular tokamak', plotAlt: 'Guiding-center trajectories in the poloidal plane of a circular tokamak',
    figures: {
      'panels': {
        alt: 'Poloidal-plane orbits of eight guiding centers, one panel each',
        caption: 'The eight orbits start at the same position and speed, with different parallel velocity fractions. Four reflect and form banana-shaped paths; four pass around the magnetic axis. Grey curves mark flux surfaces and the open circle marks the initial position.',
      },
      'velocity': {
        alt: 'Parallel velocity of the guiding centers against time',
        caption: 'The parallel velocity changes sign at each mirror reflection for trapped particles (solid curves). Passing particles (dotted curves) retain their initial sign. The normalization uses the common initial speed, not the instantaneous parallel speed.',
      },
      'conservation': {
        alt: 'Relative change of energy and canonical toroidal momentum against time',
        caption: 'Changes in energy and canonical toroidal momentum evaluated using the analytic equilibrium along the computed orbits. Energy is normalized to its initial value. Momentum is normalized to the sum of the magnitudes of its initial mechanical and magnetic-flux contributions, since their sum can nearly cancel. Values below 1e-12 are clipped for the logarithmic display. These diagnostics include equilibrium projection and time-integration errors.',
      },
    },
  },
  'hybrid-alfven-ion-coupling': {
    category: 'Hybrid kinetic-MHD', setupTitle: 'Two-way kinetic-MHD coupling', plotTitle: 'Energy exchange between energetic ions and the Alfvén wave', plotAlt: 'Field and fluid energy oscillating as they exchange energy with the energetic ion population',
  },
  'hybrid-current-coupling': {
    category: 'Hybrid kinetic-MHD', setupTitle: 'An Alfvén wave coupled to full-orbit ions', plotTitle: 'Current-coupling energy exchange and conservation', plotAlt: 'MHD kinetic and magnetic energies, ion and pressure energy changes, and total-energy conservation error',
    figures: {
      'wave-animation': {
        alt: 'Animated transverse velocity and magnetic perturbation along the periodic hybrid plasma slab',
        caption: 'Play or scrub through the wave evolution from t = 0 to 20. The upper panel shows the fluid velocity Uₓ and the lower panel the magnetic perturbation Bₓ, with fixed vertical scales to show their changing amplitudes as the wave exchanges energy with kinetic ions.',
      },
      'space-time': {
        alt: 'Space-time map of the transverse MHD velocity coupled to kinetic ions',
        caption: 'The initial sinusoidal transverse velocity evolves in a periodic slab along B₀. Full-orbit ions feed their current back into the MHD wave; finite marker sampling adds noise.',
      },
    },
  },
  'itg-drift-wave': {
    category: 'Drift-kinetic turbulence', setupTitle: 'Drift waves from a temperature gradient', plotTitle: 'ITG density perturbation energy growth', plotAlt: 'Density perturbation energy growing exponentially as the ITG drift wave develops',
    figures: {
      'potential': {
        alt: 'Animated electrostatic potential in radius and poloidal angle',
        caption: 'The electrostatic potential at z = 0 against radius and poloidal angle. The colour scale is fixed and symmetric about zero, so growth of the perturbation shows up as deepening colour.',
      },
      'mode-growth': {
        alt: 'Amplitudes of the poloidal Fourier modes against time and their fitted growth rates',
        caption: 'Left: radial rms amplitude of the potential\'s poloidal Fourier modes at the seeded axial mode number, with the seeded m = 5 in bold. Right: exponential growth rate of each, fitted for 25 < t < 175, against the poloidal wavenumber.',
      },
      'spectrum': {
        alt: 'Poloidal and axial Fourier spectrum of the potential at the first and last time',
        caption: 'Radial rms of the potential\'s Fourier amplitudes in the poloidal (m) and axial (n) mode numbers, on a logarithmic colour scale, at the first and last saved time.',
      },
      'radial-structure': {
        alt: 'Radial profile of the seeded potential mode at several times',
        caption: 'Radial profile of the seeded Fourier mode of the potential at evenly spaced times, on a logarithmic axis.',
      },
      'radial-time': {
        alt: 'Root-mean-square potential over flux surfaces as a function of radius and time',
        caption: 'The root-mean-square of the potential over each flux surface (poloidal angle and axis), against radius and time, on a logarithmic colour scale.',
      },
      'profile-change': {
        alt: 'Change of the flux-surface-averaged density against radius and time',
        caption: 'The density projected on the field grid, averaged over each flux surface, minus its initial value: the change of the radial profile as the wave grows.',
      },
    },
  },
  'hasegawa-wakatani': {
    category: 'Drift-wave turbulence', setupTitle: 'Eddies self-organizing into zonal flow', plotTitle: 'Animated vorticity and density in Hasegawa–Wakatani turbulence', plotAlt: 'Vorticity and density eddies evolving in a periodic plasma slab',
    figures: {
      'zonal-energy': {
        alt: 'Stacked kinetic energy in drift-wave and zonal-flow modes',
        caption: 'The E×B kinetic energy split by poloidal Fourier mode. The ky = 0 part is the banded zonal flow; everything with ky ≠ 0 is assigned to the turbulent drift-wave field.',
      },
    },
  },
  'maxwell-structure-preservation': {
    category: 'Numerical methods', setupTitle: 'One wave, three time integrators', plotTitle: 'Energy error of implicit and explicit time integrators', plotAlt: 'Relative energy error against time for Crank-Nicolson, RK4 and Heun time integrators on the same Maxwell wave',
    figures: {
      'profile': {
        alt: 'The electric field profile after thirty wave periods for the three time integrators',
        caption: 'The field after 30 periods, for the schemes that stayed stable. Both still sit on the exact standing-wave profile, so the energy differences above are not visible here: the cost of the explicit scheme is a slow, one-directional loss rather than a wrong shape. Heun\'s method is left out because its solution has diverged by this time. All runs share the same FEEC space discretization and the same time step — only the time integrator differs.',
      },
    },
  },
  'maxwell-wave': {
    category: 'Electromagnetic waves', setupTitle: 'A broadband vacuum light wave', plotTitle: 'Power spectrum of the Struphy Maxwell simulation', plotAlt: 'Power spectrum of a simulated Struphy Maxwell simulation',
    figures: {
      'space-time': {
        alt: 'Space-time map of the electric field of vacuum light waves',
        caption: 'The electric field of the run above along z, over time. The broadband noise launches waves in both directions, which appear as criss-crossing diagonal stripes; their slope is the wave speed, c = 1 in these units, as measured in the dispersion plot.',
      },
    },
  },
  'orszag-tang-vortex': {
    category: 'Nonlinear MHD', setupTitle: 'Nonlinear evolution of crossed vortices', plotTitle: 'Density and magnetic field lines in the Orszag–Tang vortex', plotAlt: 'Density and magnetic field lines in the Orszag–Tang vortex',
    figures: {
      'conservation': {
        alt: 'MHD energy channels and conservation diagnostics over time',
        caption: 'Kinetic, magnetic, internal and total energy, followed by the relative total-energy change and the L2 norm of the discrete magnetic divergence. The latter is the square root of tot_div_B, which stores the squared norm. Values below 1e-16 are clipped for display. Finite solver tolerances and time splitting affect energy conservation.',
      },
      'pressure-cut': {
        alt: 'Initial and final gas pressure along the midplane',
        caption: 'Gas pressure along y = π at the beginning and end of this run. This is a diagnostic of this finite-resolution evolution, not a comparison with reference data or a convergence result.',
      },
    },
  },
  'poisson-source': {
    category: 'Electrostatics', setupTitle: 'A driven electrostatic potential', plotTitle: 'Poisson potential compared with its exact solution', plotAlt: "Struphy's FEEC Poisson potential closely tracking the exact cosine-mode solution",
  },
  'resistive-x-point': {
    category: 'Magnetic reconnection', setupTitle: 'Resistive relaxation of a driven magnetic null', plotTitle: 'Current density and magnetic flux around an X-point', plotAlt: 'Animated out-of-plane current density beneath magnetic flux contours around a central X-point',
    figures: {
      'reconnection': {
        alt: 'Resistive electric field and connected magnetic flux at the X-point',
        caption: 'The resistive contribution eta Jz at the central null is the reconnection electric field in this symmetric two-dimensional setup. The lower panel tracks the flux difference between the neighbouring O-point and the X-point.',
      },
      'conservation': {
        alt: 'MHD energy conversion, total-energy error and magnetic-divergence norm',
        caption: 'Resistivity converts magnetic energy into internal energy. Total-energy change and the discrete L2 norm of div B monitor the numerical evolution.',
      },
    },
  },
  'shear-alfven-wave': {
    category: 'MHD waves', setupTitle: 'A broadband shear-Alfvén wave', plotTitle: 'Power spectrum of the Struphy shear-Alfvén simulation', plotAlt: 'Power spectrum of a simulated Struphy shear-Alfvén wave',
    figures: {
      'space-time': {
        alt: 'Space-time map of the transverse velocity of shear-Alfvén waves',
        caption: 'The transverse velocity of the run above along z, over time. The broadband noise launches shear-Alfvén waves in both directions, which appear as criss-crossing diagonal stripes; their slope is the Alfvén speed, v_A = 1 in these units. The colors show the first logical component of the velocity.',
      },
    },
  },
  'sph-velocity-diffusion': {
    category: 'Viscous particle flow', setupTitle: 'A velocity wave smoothed by viscosity', plotTitle: 'SPH velocity diffusion compared with the exact decay', plotAlt: 'A sinusoidal SPH velocity profile decreasing in amplitude while tracking the exact viscous solution',
  },
  'strong-landau-damping': {
    category: 'Kinetic electrostatics', setupTitle: 'Particle trapping in a large-amplitude wave', plotTitle: 'Electric field energy in strong Landau damping', plotAlt: 'Electric field energy damping and bouncing as trapped particles oscillate',
  },
  'two-stream-instability': {
    category: 'Kinetic electrostatics', setupTitle: 'Free energy in counter-streaming beams', plotTitle: 'Electric field energy growth in the two-stream instability', plotAlt: 'Electric field energy growing exponentially before saturating',
    figures: {
      'velocity-time': {
        alt: 'Space-averaged velocity distribution as a function of time',
        caption: 'The distribution averaged over space, f(v, t). The two beams stay narrow and separate until about t = 20, when the instability saturates; over the following ten time units they merge into a single broad distribution that fills the gap between them.',
      },
      'phasespace': {
        alt: 'Phase-space density rolled up into the classic two-stream \'cat\'s eye\' vortex pattern',
        caption: 'The classic two-stream picture: the phase-space density f(x, v); drag the slider or press Play. The two beams, initially flat bands at v = ±3, are bent by the growing wave and roll up into a trapped-particle “cat’s eye” hole. The frame shown is from the middle of the run (t = 25).',
      },
    },
  },
  'vlasov-tokamak': {
    category: 'Kinetic particle orbits', setupTitle: 'Test particles in a fixed tokamak field', plotTitle: 'Full-orbit particle trajectories in a tokamak', plotAlt: "Full-orbit particle trajectories gyrating around a tokamak's torus",
    figures: {
      'projections': {
        alt: 'Projected particle trajectories',
        caption: 'Additional projections of the simulated trajectories.',
        aspect: '1500/560',
      },
    },
  },
  'vortex-merger': {
    category: 'Electrostatic drift', setupTitle: 'Two charge blobs in an annulus', plotTitle: 'Animated charge density showing two blobs winding into one core', plotAlt: 'Animated charge density showing two blobs winding into one core',
    figures: {
      'energy': {
        alt: 'Electrostatic energy change after the first Poisson solve',
        caption: 'Electrostatic energy relative to the first solved field, at t = 0.02. The t = 0 scalar is an uninitialized zero and is omitted. The drift measures the error of this finite-resolution particle and field calculation; it is not a convergence study.',
      },
    },
  },
  'weak-landau-damping': {
    category: 'Kinetic electrostatics', setupTitle: 'Phase mixing in a uniform plasma', plotTitle: 'Electric field energy decay in weak Landau damping', plotAlt: 'Electric field energy decaying exponentially over time',
    figures: {
      'space-time': {
        alt: 'Space-time map of the electric field of the damped Langmuir wave',
        caption: 'The electric field E(x, t) of the run above. The single cosine mode is a standing wave: its sign alternates in time with a period of about 4.4 (ω ≈ 1.42) around nodes that stay in place, while its amplitude decays.',
      },
    },
  },
  'weibel-instability': {
    category: 'Kinetic electromagnetics', setupTitle: 'Spontaneous field generation from temperature anisotropy', plotTitle: 'Magnetic field energy growth in the Weibel instability', plotAlt: 'Magnetic field energy growing exponentially as the Weibel instability develops',
    figures: {
      'space-time': {
        alt: 'Space-time map of the magnetic field of the Weibel instability',
        caption: 'The magnetic field B₃(x, t). The seeded cosine mode, one wavelength across the box, grows in place around fixed nodes. It remains small until t ≈ 100 and reaches |B₃| ≈ 0.1 by the end of the run.',
      },
      'anisotropy': {
        alt: 'Temperature anisotropy as a function of time',
        caption: 'The temperature anisotropy: the ratio of the velocity variances along v₂ and v₁, computed from the binned (v₁, v₂) distribution with velocity_moments. It starts near 12, the ratio set by the initial temperatures, stays there while the field is small, and falls to about 3 by t = 200 as the magnetic field grows.',
      },
    },
  },
  'mhd-slab-waves': {
    category: 'MHD waves', setupTitle: 'Three waves from one noise spectrum', plotTitle: 'Power spectra of velocity and pressure MHD waves in a magnetized slab', plotAlt: 'Power spectra of velocity and pressure MHD waves in a magnetized slab',
  },
  'incompressible-shear-relaxation': {
    category: 'Incompressible flow', setupTitle: 'Removing compression, keeping the shear', plotTitle: 'Shear flow and compressive wave between no-slip walls', plotAlt: 'Shear flow and compressive wave of an incompressible fluid between no-slip walls',
  },
  'zeldovich-caustic': {
    category: 'Pressureless flow', setupTitle: 'A gas that falls onto itself', plotTitle: "Density and phase space of a pressureless Zel'dovich collapse", plotAlt: "Density and phase space of a pressureless Zel'dovich collapse",
  },
  'alfven-standing-wave': {
    category: 'MHD waves', setupTitle: 'One mode, two counter-propagating waves', plotTitle: 'Kinetic and magnetic energy exchanging in a standing Alfvén wave', plotAlt: 'Kinetic and magnetic energy oscillating out of phase with a constant sum in a standing Alfvén wave',
    figures: {
      'space-time': {
        alt: 'Space-time map of the transverse velocity of a standing Alfvén wave, with fixed nodes',
        caption: 'The transverse velocity along z, over time. The nodes stay at the same z while the profile changes sign every half period: the signature of a standing wave, in contrast with the criss-crossing diagonal stripes of the travelling waves in the broadband example. The relative drift of the total energy over 40 time units stays at the level of the linear-solver tolerance.',
      },
    },
  },
  'cold-plasma-waves': {
    category: 'Electromagnetic waves', setupTitle: 'Circularly polarized waves in a magnetized plasma', plotTitle: 'Power spectrum of cold-plasma R, L and whistler waves', plotAlt: 'Power spectrum of the transverse electric field with the analytic R, L and whistler branches of a cold magnetized plasma',
    figures: {
      'energy': {
        alt: 'Electric, magnetic and electron-current energy of the cold plasma against time, with their constant sum',
        caption: 'The initial noise is purely electric. Within a cyclotron period it is shared with the magnetic field and the kinetic energy of the electron current, while the total stays constant up to a tiny relative change over the run. Each propagator of the splitting is a Crank–Nicolson step that conserves its share of the energy up to the tolerance of the linear solver.',
      },
    },
  },
  'hall-mhd-waves': {
    category: 'MHD waves', setupTitle: 'The Alfvén wave splits at the ion inertial length', plotTitle: 'Power spectra of Hall-MHD whistler, ion-cyclotron and sound waves', plotAlt: 'Power spectra of the magnetic field and pressure with the analytic whistler, ion-cyclotron and sound branches of Hall MHD',
    figures: {
      'phase-velocity': {
        alt: 'Phase velocity against wavenumber for the whistler, ion-cyclotron and sound waves, measured and exact',
        caption: 'Phase velocities read off the spectra above (markers) against the exact Hall-MHD values (lines). At long wavelength both transverse branches travel at the Alfvén speed, as in ideal MHD. Near k d_i = 1 they separate: the whistler speeds up and the ion-cyclotron wave slows down. The sound wave along the field is not affected by the Hall term. The median relative frequency error is measured for each of the three branches.',
      },
    },
  },
  'diffusion-methods': {
    category: 'Diffusion', setupTitle: 'Diffusion with random and deterministic particles', plotTitle: 'Density mode diffusion by random-walk and deterministic particles', plotAlt: 'Density mode diffusing by random-walk and deterministic particle methods',
  },
  'cold-plasma-oscillation': {
    category: 'Electromagnetic waves', setupTitle: 'A cold plasma rings at its plasma frequency', plotTitle: 'Plasma oscillation: field and flow energy exchange', plotAlt: 'Electric field and electron kinetic energy of a plasma oscillation trading places as cos² and sin²',
    figures: {
      'frequency-scan': {
        alt: 'Measured oscillation frequency of a cold plasma against density, on the square-root law',
        caption: 'The oscillation frequency of the same run at four densities, measured from the zero crossings of the field, on the analytic plasma frequency ω_p = √n₀ (α = 1).',
      },
    },
  },
  'cold-plasma-wave-packet': {
    category: 'Electromagnetic waves', setupTitle: 'A packet splits into a fast and a slow wave', plotTitle: 'Space-time map of an L and an R wave packet in a cold plasma', plotAlt: 'Energy of a transverse electric field packet in space and time, splitting into a fast L-wave packet and a slow R-wave packet',
    figures: {
      'centroid': {
        alt: 'Distance travelled by the L-wave and whistler packets against time, on straight lines of the analytic group velocity',
        caption: 'The position of the energy peak of each of the two packets, against time, on the straight line of the group velocity d ω / d k of the cold-plasma dispersion relation. The L-wave packet is fast; the R-type packet is slow and disperses as it goes, and its two branches, the whistler and the upper R wave, have almost the same group velocity here (0.45 and 0.43); the fit is compared with the whistler value.',
      },
    },
  },
  'maxwell-cavity-resonances': {
    category: 'Electromagnetic waves', setupTitle: 'The resonances of a box', plotTitle: 'Power spectrum of E_z in a rectangular box with the exact resonances', plotAlt: 'Power spectrum of the electric field of a rectangular box with peaks at the exact resonance frequencies',
    figures: {
      'frequency-error': {
        alt: 'Relative error of the measured resonance frequencies of a Maxwell box',
        caption: 'The relative difference between each measured peak and the exact resonance frequency, against that frequency. It is set by the frequency resolution of the run (about 2π/T) and by the numerical dispersion of the cubic splines, which lowers the higher frequencies.',
      },
    },
  },
  'maxwell-curved-mesh': {
    category: 'Electromagnetism', setupTitle: 'A pulse on a bent mesh', plotTitle: 'Electric field pulse on a distorted Colella mesh', plotAlt: 'A pulse of the out-of-plane electric field spreading across a box on a distorted mesh',
    figures: {
      'diagnostics': {
        alt: 'Error of the electric field against the exact solution and change of the total energy over time',
        caption: 'Top: the root-mean-square difference between the computed E_z and the exact solution at the points of the distorted mesh, relative to the rms of the initial pulse. Bottom: the relative change of the total energy, which stays small: the scheme conserves it whatever the shape of the mesh.',
      },
    },
  },
  'gyromotion': {
    category: 'Particle orbits', setupTitle: 'Circles in a uniform field', plotTitle: 'Gyrating particles compared with the exact orbits', plotAlt: 'Orbits of four charged particles circling a uniform magnetic field, on the exact circles of different Larmor radii',
    figures: {
      'helices': {
        alt: 'Helical orbits of four charged particles around a uniform magnetic field',
        caption: 'The same four orbits in three dimensions: circles in the plane perpendicular to B, stretched into helices by the free streaming along the field.',
      },
      'conservation': {
        alt: 'Relative change of the speed and the perpendicular speed of gyrating particles',
        caption: 'A magnetic field does no work, so the speed and the perpendicular speed of each marker are constant. The relative changes stay at round-off level.',
      },
    },
  },
  'langmuir-wave-dispersion': {
    category: 'Kinetic electrostatics', setupTitle: 'Frequency and damping of Langmuir waves', plotTitle: 'Langmuir wave frequency and Landau damping against wavenumber', plotAlt: 'Measured frequency and Landau damping rate of Langmuir waves against wavenumber, on the kinetic dispersion relation',
    figures: {
      'signals': {
        alt: 'Electric field of Langmuir waves at four wavenumbers, damped inside their Landau envelopes',
        caption: 'The amplitude of the electric field mode in the four runs, with the Landau envelope exp(γt) (γ from the kinetic dispersion relation, scaled to the first peak) as dashed lines. The wave oscillates faster and dies out sooner as k grows. The first time units hold a transient of phase-mixing modes that are not the Langmuir wave, so the frequencies and rates above are read from t = 1 to t = 12.',
      },
    },
  },
  'resistive-diffusion': {
    category: 'Resistive MHD', setupTitle: 'A field that leaks away', plotTitle: 'Resistive decay of a magnetic field and its Ohmic heating', plotAlt: 'A sinusoidal magnetic field decaying at the resistive rate while its energy becomes heat',
    figures: {
      'decay': {
        alt: 'Decay of the amplitude of a magnetic field mode at three resistivities, on the exact exponentials',
        caption: 'The amplitude of the field mode at three resistivities, on log axes where the exact decay exp(−η k² t) is a straight line. The fitted rates are compared with η k² for each resistivity (η = 0.05, 0.1, 0.2).',
      },
    },
  },
  'damped-alfven-wave': {
    category: 'MHD waves', setupTitle: 'An Alfvén wave that loses its field lines', plotTitle: 'A resistively damped standing Alfvén wave', plotAlt: 'A standing Alfvén wave oscillating inside a decaying exponential envelope',
    figures: {
      'decay': {
        alt: 'Peaks of the amplitude of a damped Alfvén wave at three resistivities, on the exact exponentials',
        caption: 'The successive amplitude peaks of the wave at three resistivities, on log axes where the exact decay exp(−η k² t / 2) is a straight line. The fitted rates are compared with η k² / 2 for each resistivity (η = 0.05, 0.1, 0.2).',
      },
    },
  },
  'acoustic-pulse': {
    category: 'Gas dynamics', setupTitle: 'A pulse that splits in two', plotTitle: 'An acoustic pulse splitting and travelling around a periodic box', plotAlt: 'A density pulse splitting into two travelling pulses, with kinetic and thermodynamic energy trading places',
    figures: {
      'space-time': {
        alt: 'Space-time map of the density change of an acoustic pulse splitting into two',
        caption: 'The density change of the run above along x, over time. The pulse splits into two, and the two travelling pulses leave straight lines whose slope is the speed of sound, c = 1; because the box is periodic, they meet again on the opposite side at t = L / (2c).',
      },
    },
  },
  'poisson-convergence': {
    category: 'Electrostatics', setupTitle: 'Error against resolution on a bent mesh', plotTitle: 'Convergence of the Poisson solver at spline degrees 1 to 3', plotAlt: 'Error of the Poisson potential against resolution for splines of degree 1 to 3 on straight and distorted meshes',
    figures: {
      'maps': {
        alt: 'Computed potential and its error on a distorted mesh of 8 by 12 cells with splines of degree 2',
        caption: 'The potential (left) and its error against the exact solution (right) for splines of degree 2 on the distorted mesh of 8 × 12 cells. The largest error is a small fraction of the peak of the potential, and it follows the pattern of the mesh.',
      },
    },
  },
};
