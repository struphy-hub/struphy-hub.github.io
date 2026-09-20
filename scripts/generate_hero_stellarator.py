"""Generate the landing-page hero: a five-period stellarator with magnetic field lines.

Runs GVEC on the compact stellarator of ``docs/src/examples/gvec-equilibrium.py`` and writes VTK
PolyData files for the interactive viewer (``VtkHeroViewer.astro``) into ``docs/public/hero/``:

* ``stellarator-surface.vtp``: the last closed flux surface over the full torus, with |B| as point data;
* ``stellarator-lines.vtp``: magnetic field lines on nested flux surfaces, drawn as thin tubes (WebGL
  cannot draw lines wider than one pixel), with |B| along each line.

* ``stellarator-particles.vtp`` / ``stellarator-trails.vtp``: full-orbit ions from Struphy's ``Vlasov`` model,
  launched over the plasma volume, as spheres at their final position and their most recent orbit as thin tubes.
  The orbits are helices: the gyration around the field lines is resolved.

It also renders ``docs/public/images/stellarator-hero.webp``, the still shown until the viewer has loaded.

All are small and committed, so the site build does not need GVEC. Rerun this script only when the hero
should change. It needs compiled Struphy kernels and ``pip install "./submodules/struphy[phys]" pyvista``.

    python scripts/generate_hero_stellarator.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "docs" / "src" / "examples" / "gvec-equilibrium.py"
OUTPUT = ROOT / "docs" / "public" / "hero"
IMAGES = ROOT / "docs" / "public" / "images"

# |B| colour ramp; VtkHeroViewer.astro uses the same stops.
B_STOPS = ("#2f6bd8", "#55e6e1", "#ffd166")
B_RANGE = (1.0, 1.42)
TUBE_RADIUS, TUBE_SIDES = 0.016, 6
PARTICLE_COLOR = "#ff5a5f"
# Full-orbit ions. |B| ~ 1.2 in these units, so v_perp = 0.14 is a gyroradius of ~0.12. The cross-section is only
# ~0.3 wide from the axis in its narrow direction, so this is as large as it can be without markers hitting the
# wall, and small enough that most of them stay confined for the whole trail.
N_MARKERS, V_PERP, V_PAR, RHO = 60, 0.14, (0.1, 0.3), (0.2, 0.55)
T_END, DT, SAVE_STEP, TRAIL_STEPS = 13.0, 0.02, 5, 131  # the trail is the whole run: 13 time units, 2.4 gyrations
MAPPING_ELEMENTS, MAPPING_DEGREE = (8, 16, 80), (2, 2, 2)  # the full torus: five field periods, 16 elements each
MARKER_RADIUS, TRAIL_RADIUS = 0.12, 0.024

TWO_PI = 2.0 * np.pi
N_THETA, N_ZETA = 720, 1000  # dense sampling of a flux surface, for the surface mesh and the field lines
# (rho, number of field lines on that flux surface, toroidal turns followed by each)
FIELD_LINES = ((1.0, 3, 2.0), (0.7, 3, 3.0), (0.4, 2, 4.0), (0.15, 2, 5.0))
POINTS_PER_TURN = 300


def example_namespace() -> dict:
    """The gallery example's definitions (GVEC parameters, mapping, `make_simulation`), without running it."""
    namespace: dict = {"__name__": "gvec_equilibrium_example"}
    exec(EXAMPLE.read_text().split("# Metadata generation imports")[0], namespace)
    return namespace


def solve(namespace: dict):
    """Run GVEC; returns its state and the state files that Struphy's `GVECequilibrium` reads."""
    import gvec

    run = gvec.run(namespace["gvec_parameters"](), runpath=Path(tempfile.mkdtemp()), quiet=True, redirect_gvec_stdout=True)
    print(f"GVEC converged in {run.GVEC_iter_used} iterations, |force| = {run.max_force:.2e}")
    return run.state, {"param_file": str(run.state.parameterfile), "dat_file": str(run.state.statefile)}


def periodic(values: np.ndarray) -> np.ndarray:
    """Append the first (theta, zeta) row/column so that interpolation wraps around."""
    return np.pad(values, [(0, 1), (0, 1)], mode="wrap")


class Surface:
    """One flux surface sampled on a dense periodic (theta, zeta) grid."""

    def __init__(self, state, rho: float):
        self.theta = np.linspace(0.0, TWO_PI, N_THETA, endpoint=False)
        self.zeta = np.linspace(0.0, TWO_PI, N_ZETA, endpoint=False)
        data = state.evaluate("pos", "mod_B", "LA", "iota", rho=np.array([rho]), theta=self.theta, zeta=self.zeta)
        self.pos = np.asarray(data["pos"])[:, 0]  # (3, theta, zeta)
        self.mod_b = np.asarray(data["mod_B"])[0]
        self.la = np.asarray(data["LA"])[0]
        self.iota = float(np.asarray(data["iota"]).ravel()[0])
        self._axes = (np.append(self.theta, TWO_PI), np.append(self.zeta, TWO_PI))

    def interpolate(self, values: np.ndarray, theta: np.ndarray, zeta: np.ndarray) -> np.ndarray:
        fn = RegularGridInterpolator(self._axes, periodic(values), method="linear")
        return fn(np.column_stack([np.mod(theta, TWO_PI), np.mod(zeta, TWO_PI)]))

    def field_line(self, alpha: float, turns: float) -> tuple[np.ndarray, np.ndarray]:
        """Follow theta* = alpha + iota * zeta, where theta* = theta + LA is GVEC's straight-field-line angle."""
        zeta = np.linspace(0.0, TWO_PI * turns, int(POINTS_PER_TURN * turns))
        target = alpha + self.iota * zeta
        theta = np.empty_like(zeta)
        theta_ext = np.append(self.theta, TWO_PI)
        for k, ze in enumerate(zeta):
            j = int(round((np.mod(ze, TWO_PI) / TWO_PI) * N_ZETA)) % N_ZETA
            star = self.theta + self.la[:, j]  # increases monotonically with theta
            wrapped = np.mod(target[k] - star[0], TWO_PI) + star[0]
            theta[k] = np.interp(wrapped, np.append(star, star[0] + TWO_PI), theta_ext)
        points = np.column_stack([self.interpolate(self.pos[i], theta, zeta) for i in range(3)])
        return points, self.interpolate(self.mod_b, theta, zeta)


def surface_mesh(surface: Surface) -> pv.PolyData:
    # The dense grid is only needed for the field lines; every 5th poloidal and 4th toroidal sample is smooth
    # enough for the surface and keeps the file small.
    wrap = [(0, 0), (0, 1), (0, 1)]
    pos = np.pad(surface.pos[:, ::5, ::4], wrap, mode="wrap")  # repeat the first row/column to close the seams
    mod_b = periodic(surface.mod_b[::5, ::4])
    grid = pv.StructuredGrid(
        np.moveaxis(pos[0], 0, 1), np.moveaxis(pos[1], 0, 1), np.moveaxis(pos[2], 0, 1)
    )  # index order (zeta, theta)
    grid.point_data["B"] = mod_b.T.ravel(order="F")
    mesh = grid.extract_surface(algorithm=None).triangulate()
    # Merge the repeated seam points so the surface is closed.
    return mesh.clean(tolerance=1e-7)


def lines_mesh(state) -> pv.PolyData:
    points, strengths, cells = [], [], []
    offset = 0
    for rho, count, turns in FIELD_LINES:
        surface = Surface(state, rho)
        for index in range(count):
            line, strength = surface.field_line(TWO_PI * (index + 0.5) / count, turns)
            points.append(line)
            strengths.append(strength)
            cells.append(np.concatenate([[len(line)], offset + np.arange(len(line))]))
            offset += len(line)
    mesh = pv.PolyData(np.vstack(points))
    mesh.lines = np.concatenate(cells).astype(np.int64)
    mesh.point_data["B"] = np.concatenate(strengths)
    return mesh.tube(radius=TUBE_RADIUS, n_sides=TUBE_SIDES, capping=False)


def particle_meshes(state_files: dict) -> tuple[pv.PolyData, pv.PolyData]:
    """Run Struphy's full-orbit `Vlasov` model; return (spheres at the end, recent orbit tubes)."""
    import h5py

    from struphy import (
        BoundaryParameters,
        DerhamOptions,
        EnvironmentOptions,
        LoadingParameters,
        SavingParameters,
        Simulation,
        Time,
        WeightsParameters,
        equils,
        grids,
        maxwellians,
    )
    from struphy.models import Vlasov

    # Full-orbit velocities are Cartesian, so the whole torus is mapped (`use_nfp=False`): a marker leaving a
    # single field period would need its velocity rotated when it re-enters.
    equilibrium = equils.GVECequilibrium(
        rel_path=False, use_nfp=False, num_elements=MAPPING_ELEMENTS, degree=MAPPING_DEGREE, **state_files
    )

    rng = np.random.default_rng(7)
    rho = rng.uniform(*RHO, N_MARKERS)
    eta2, eta3 = rng.uniform(0.0, 1.0, (2, N_MARKERS))
    v_par = rng.uniform(*V_PAR, N_MARKERS) * rng.choice([-1.0, 1.0], N_MARKERS)
    initial = []
    for r, e2, e3, vp in zip(rho, eta2, eta3, v_par):
        b_hat = np.ravel(equilibrium.unit_b_cart(r, e2, e3, squeeze_out=True))[:3]  # unit vector, then position
        perp = np.cross(b_hat, [0.0, 0.0, 1.0])
        perp /= np.linalg.norm(perp)
        initial.append((r, e2, e3, *(vp * b_hat + V_PERP * perp)))

    model = Vlasov()
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(Np=N_MARKERS, seed=1, specific_markers=tuple(initial)),
        weights_params=WeightsParameters(),
        boundary_params=BoundaryParameters(bc=("remove", "periodic", "periodic")),
        saving_params=SavingParameters(n_markers=1.0),
        bufsize=2.0,
    )
    model.propagators.push_vxb.options = model.propagators.push_vxb.Options()
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    run = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=tempfile.mkdtemp(), sim_folder="hero_markers", save_step=SAVE_STEP),
        time_opts=Time(dt=DT, Tend=T_END, split_algo="Strang"),
        domain=equilibrium.numerical_domain,
        equil=equilibrium,
        grid=grids.TensorProductGrid(num_elements=MAPPING_ELEMENTS),
        derham_opts=DerhamOptions(degree=MAPPING_DEGREE, bcs=(("free", "free"), None, None)),
    )
    run.run()

    with h5py.File(Path(run.env.path_out) / "data/data_proc0.hdf5") as data:
        saved = np.asarray(data["kinetic/kinetic_ions/markers"])
    history = np.full((len(saved), N_MARKERS, 3), np.nan)
    for step, rows in enumerate(saved):
        active = rows[(rows[:, -1] >= 0) & (rows[:, 0] >= 0)]  # removed markers keep their id but get eta1 = -1
        history[step, active[:, -1].astype(int)] = active[:, :3]
    alive = np.isfinite(history[-TRAIL_STEPS:]).all(axis=(0, 2))  # markers that stayed inside for the whole trail
    print(f"markers still confined: {alive.sum()} of {N_MARKERS}")
    logical = history[-TRAIL_STEPS:, alive]

    x, y, z = np.moveaxis(
        equilibrium.numerical_domain(logical.reshape(-1, 3), change_out_order=True).reshape(logical.shape), -1, 0
    )
    positions = np.stack([x, y, z], axis=-1)  # (step, marker, xyz)

    steps = positions.shape[0]
    trails = pv.PolyData(positions.transpose(1, 0, 2).reshape(-1, 3))
    trails.lines = np.concatenate(
        [np.concatenate([[steps], m * steps + np.arange(steps)]) for m in range(positions.shape[1])]
    ).astype(np.int64)
    trails = trails.tube(radius=TRAIL_RADIUS, n_sides=5, capping=False)
    spheres = pv.PolyData(positions[-1]).glyph(
        geom=pv.Sphere(radius=MARKER_RADIUS, theta_resolution=10, phi_resolution=10), scale=False, orient=False
    )
    for mesh in (trails, spheres):
        mesh.clear_data()
        mesh.points = mesh.points.astype(np.float32)
    return spheres, trails


def render_poster(surface: pv.PolyData, lines: pv.PolyData, spheres: pv.PolyData, trails: pv.PolyData) -> None:
    """The static frame shown before (and without) the interactive viewer, from the viewer's default camera."""
    cmap = LinearSegmentedColormap.from_list("B", B_STOPS)
    plotter = pv.Plotter(off_screen=True, window_size=(1440, 1120))
    plotter.set_background("#07131d")
    plotter.add_mesh(surface, scalars="B", cmap=cmap, clim=B_RANGE, opacity=0.32, show_scalar_bar=False)
    plotter.add_mesh(lines, scalars="B", cmap=cmap, clim=B_RANGE, show_scalar_bar=False, smooth_shading=True)
    plotter.add_mesh(trails, color=PARTICLE_COLOR, opacity=0.9, show_scalar_bar=False)
    plotter.add_mesh(spheres, color=PARTICLE_COLOR, show_scalar_bar=False, smooth_shading=True)
    b = surface.bounds
    center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
    plotter.camera_position = [tuple(center + span * np.array([0.35, -0.95, 0.7])), tuple(center), (0, 0, 1)]
    plotter.reset_camera()  # fit the whole torus, keeping the viewing direction
    plotter.camera.zoom(1.35)
    png = IMAGES / "stellarator-hero.png"
    plotter.screenshot(png)
    plotter.close()
    Image.open(png).convert("RGB").save(IMAGES / "stellarator-hero.webp", "WEBP", quality=80, method=6)
    png.unlink()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    namespace = example_namespace()
    state, state_files = solve(namespace)

    surface = surface_mesh(Surface(state, 1.0))
    surface.points = surface.points.astype(np.float32)
    surface.save(OUTPUT / "stellarator-surface.vtp", binary=True)
    print(f"surface: {surface.n_points} points, |B| in [{surface['B'].min():.3f}, {surface['B'].max():.3f}]")

    lines = lines_mesh(state)
    lines.points = lines.points.astype(np.float32)
    lines.save(OUTPUT / "stellarator-lines.vtp", binary=True)
    print(f"field lines: {lines.n_points} points, |B| in [{lines['B'].min():.3f}, {lines['B'].max():.3f}]")

    spheres, trails = particle_meshes(state_files)
    spheres.save(OUTPUT / "stellarator-particles.vtp", binary=True)
    trails.save(OUTPUT / "stellarator-trails.vtp", binary=True)
    print(f"particles: {spheres.n_points} sphere points, {trails.n_points} trail points")

    IMAGES.mkdir(parents=True, exist_ok=True)
    render_poster(surface, lines, spheres, trails)


if __name__ == "__main__":
    main()
