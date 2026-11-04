"""MSR core model: design description and the two-step solution sequence.

Sequence (``solve_core``):

1. Lattice calculation per core zone (45 groups, self-shielding, CP cell).
2. 1D radial (cylinder) and 1D axial (slab) 45-group spectral-zone diffusion
   calculations over the full layout -> region-wise fine spectra.
3. Condensation of every region onto the few-group structure with its own
   spectrum (zones, reflectors, plena, downcomer, vessel, control rod).
4. 2D R-Z few-group forward + adjoint diffusion -> k_eff, flux, power,
   adjoint-weighted kinetics parameters, graphite fast flux, and salt-
   inventory-averaged one-group cross sections for depletion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from .constants import MEV_TO_J, SECONDS_PER_YEAR
from .data.delayed_neutrons import mix_delayed
from .data.nuclides import NUCLIDES
from .diffusion import DiffusionResult, DiffusionSolver, Mesh, uniform_edges
from .lattice import CellGeometry, LatticeResult, run_lattice
from .macro import MacroXS, build_macro, condense, homogeneous_sig0
from .materials import (Material, b4c, graphite, hastelloy_n, mix, molten_salt,
                        salt_at_temperature)
from .xs.energy import FINE45, GroupStructure, collapse_map, few_group
from .xs.library import Library, load_library

GRAPHITE_FLUENCE_LIMIT = 3.0e22   # n/cm2 (E > 50 keV), ORNL MSBR design basis


@dataclass
class CoreZone:
    name: str
    outer_radius: float            # cm
    salt_fraction: float
    channel_radius: float = 1.0    # cm
    lattice: str = "hex"


@dataclass
class ControlRod:
    radius: float = 5.0            # cm (central absorber channel)
    insertion: float = 0.0         # cm inserted from the top of the core
    absorber_fraction: float = 0.3 # absorber volume fraction in the rod channel
    b10_fraction: float = 0.199


@dataclass
class MSRDesign:
    name: str = "MSR"
    salt_composition: dict = field(default_factory=lambda: {"LiF": 0.717, "BeF2": 0.16,
                                                            "ThF4": 0.12, "UF4": 0.003})
    uranium: Union[dict, float] = field(default_factory=lambda: {"U233": 1.0})
    plutonium: Optional[dict] = None
    li7_enrichment: float = 0.99995
    cl37_enrichment: Optional[float] = None
    salt_density: Union[None, float, tuple] = None
    zones: list = field(default_factory=lambda: [CoreZone("core", 150.0, 0.15, 1.0)])
    core_height: float = 300.0
    radial_reflector: float = 50.0
    downcomer: float = 5.0
    vessel: float = 5.0
    lower_plenum: float = 15.0
    upper_plenum: float = 15.0
    axial_reflector: float = 40.0
    reflector_material: str = "graphite"
    power_MW: float = 250.0
    T_fuel: float = 900.0
    T_graphite: float = 920.0
    graphite_density: float = 1.86
    graphite_boron_ppm: float = 0.0
    flow_rate_m3s: float = 1.0
    external_loop_volume_m3: float = 10.0
    void_fraction: float = 0.0
    control_rod: Optional[ControlRod] = None
    capacity_factor: float = 0.9

    # ---- materials ----------------------------------------------------
    def fuel_salt(self, T: Optional[float] = None) -> Material:
        T = self.T_fuel if T is None else T
        s = molten_salt(self.salt_composition, T, self.li7_enrichment, self.uranium,
                        self.plutonium, self.cl37_enrichment, self.salt_density,
                        name=f"{self.name} fuel salt")
        if self.void_fraction > 0:
            s = s.scaled(1.0 - self.void_fraction)
            s.meta["void_fraction"] = self.void_fraction
        return s

    def moderator(self, T: Optional[float] = None) -> Material:
        return graphite(self.graphite_density, self.T_graphite if T is None else T,
                        self.graphite_boron_ppm)

    def reflector(self) -> Material:
        if self.reflector_material == "graphite":
            return self.moderator()
        return hastelloy_n(self.T_fuel)

    # ---- geometry -----------------------------------------------------
    @property
    def core_radius(self) -> float:
        return self.zones[-1].outer_radius

    @property
    def vessel_outer_radius(self) -> float:
        return self.core_radius + self.radial_reflector + self.downcomer + self.vessel

    def zone_volumes_cm3(self) -> list:
        out, r0 = [], 0.0
        for z in self.zones:
            out.append(np.pi * (z.outer_radius ** 2 - r0 ** 2) * self.core_height)
            r0 = z.outer_radius
        return out

    def salt_volumes_m3(self) -> dict:
        zones = sum(v * z.salt_fraction for v, z in zip(self.zone_volumes_cm3(), self.zones))
        R_ref = self.core_radius + self.radial_reflector
        plena = np.pi * R_ref ** 2 * (self.lower_plenum + self.upper_plenum)
        H_tot = (self.core_height + self.lower_plenum + self.upper_plenum
                 + 2 * self.axial_reflector)
        dc = np.pi * ((R_ref + self.downcomer) ** 2 - R_ref ** 2) * H_tot
        d = {"core_zones": zones * 1e-6, "plena": plena * 1e-6, "downcomer": dc * 1e-6,
             "external_loop": self.external_loop_volume_m3}
        d["in_vessel"] = d["core_zones"] + d["plena"] + d["downcomer"]
        d["total"] = d["in_vessel"] + d["external_loop"]
        return d

    def transit_times(self, include_plena: bool = False) -> tuple:
        """(in-core transit time, out-of-core loop time) in seconds.

        With ``include_plena`` the fuelled plena are counted as part of the
        core flow path (they carry fission source and importance).
        """
        v = self.salt_volumes_m3()
        core = v["core_zones"] + (v["plena"] if include_plena else 0.0)
        tau_c = core / self.flow_rate_m3s
        tau_l = (v["total"] - core) / self.flow_rate_m3s
        return tau_c, tau_l

    def inventory_kg(self, salt: Optional[Material] = None) -> dict:
        salt = salt or self.fuel_salt()
        V = self.salt_volumes_m3()["total"] * 1e6  # cm3
        from .constants import AVOGADRO
        out = {}
        for k, n in salt.N.items():
            out[k] = n * 1e24 * V * NUCLIDES[k].mass / AVOGADRO / 1000.0
        return out

    def copy(self, **changes) -> "MSRDesign":
        d = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(d, k, v)
        return d


# ---------------------------------------------------------------------------
@dataclass
class CoreResult:
    design: MSRDesign
    k_eff: float
    diffusion: DiffusionResult
    groups: GroupStructure
    region_names: list
    region_xs: list
    lattice: list
    fine_spectra: dict
    beta_eff: float = 0.0
    beta_i: np.ndarray = None
    lambda_i: np.ndarray = None
    beta_static: float = 0.0
    generation_time: float = 0.0
    one_group: dict = field(default_factory=dict)      # nuclide -> rx -> barns
    salt_flux_avg: float = 0.0       # inventory-averaged flux (n/cm2/s)
    fast_flux_graphite_peak: float = 0.0
    graphite_lifetime_years: float = 0.0
    conversion_ratio: float = 0.0
    leakage_fraction: float = 0.0        # escaping the system (vacuum boundary)
    core_leakage_fraction: float = 0.0   # net leakage out of the fuelled zones
    power_fraction_outside_zones: float = 0.0
    k_inf_zones: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    axial_fission_shape: np.ndarray = None
    axial_importance_shape: np.ndarray = None
    tritium_rate: float = 0.0        # atoms/s

    @property
    def rho_pcm(self) -> float:
        return (self.k_eff - 1.0) / self.k_eff * 1e5

    def summary(self) -> dict:
        pk = self.diffusion.peaking(region_ids=list(range(len(self.design.zones))))
        return {
            "k_eff": self.k_eff, "reactivity_pcm": self.rho_pcm,
            "k_inf_zone1": self.k_inf_zones[0] if self.k_inf_zones else float("nan"),
            "dominance_ratio": self.diffusion.dominance_ratio,
            "beta_eff_static_pcm": self.beta_static * 1e5,
            "generation_time_us": self.generation_time * 1e6,
            "conversion_ratio": self.conversion_ratio,
            "leakage_fraction": self.leakage_fraction,
            "core_leakage_fraction": self.core_leakage_fraction,
            "power_peaking": pk["total_peaking"],
            "radial_peaking": pk.get("radial_peaking", float("nan")),
            "axial_peaking": pk.get("axial_peaking", float("nan")),
            "q_avg_W_cm3": pk["q_avg"], "q_max_W_cm3": pk["q_max"],
            "graphite_fast_flux_peak": self.fast_flux_graphite_peak,
            "graphite_lifetime_years": self.graphite_lifetime_years,
            "power_outside_zones": self.power_fraction_outside_zones,
            "salt_avg_flux": self.salt_flux_avg,
            "tritium_Ci_per_day": self.tritium_Ci_day,
        }

    @property
    def tritium_Ci_day(self) -> float:
        lam = np.log(2) / (12.32 * SECONDS_PER_YEAR)
        # atoms produced per day * lambda -> Bq; 1 Ci = 3.7e10 Bq (activity added per day)
        return self.tritium_rate * 86400.0 * lam / 3.7e10


# ---------------------------------------------------------------------------
def _edges_for(lengths_names, h):
    segs = []
    for L, _ in lengths_names:
        if L > 0:
            segs.append((L, max(1, int(np.ceil(L / h)))))
    return uniform_edges(segs)


def _region_spectra_1d(result: DiffusionResult, n_regions: int):
    V = result.mesh.volumes()
    out = {}
    for rid in range(n_regions):
        m = result.mesh.region == rid
        if np.any(m):
            out[rid] = (result.flux[:, m] * V[m][None]).sum(axis=1)
    return out


def solve_core(design: MSRDesign, few_groups: Union[str, GroupStructure] = "6g",
               mesh_size: float = 6.0, lib: Optional[Library] = None,
               salt: Optional[Material] = None, perturb: Optional[dict] = None,
               adjoint: bool = True, n_modes: int = 2, T_fuel: Optional[float] = None,
               T_graphite: Optional[float] = None) -> CoreResult:
    """Run the full lattice -> spectral-zone -> 2D R-Z sequence for a design."""
    import time
    t0 = time.time()
    lib = lib or load_library()
    coarse = few_group(few_groups) if isinstance(few_groups, str) else few_groups
    Tf = design.T_fuel if T_fuel is None else T_fuel
    Tg = design.T_graphite if T_graphite is None else T_graphite
    if salt is None:
        salt = design.fuel_salt(Tf)
    elif abs(salt.T - Tf) > 1e-6:
        salt = salt_at_temperature(salt, Tf)
    grap = graphite(design.graphite_density, Tg, design.graphite_boron_ppm)
    # ---------------- lattice per zone ----------------
    lat = []
    for z in design.zones:
        cell = CellGeometry(z.channel_radius, z.salt_fraction, z.lattice)
        lat.append(run_lattice(salt, grap if z.salt_fraction < 0.999 else None, cell, lib,
                               perturb=perturb))
    t1 = time.time()
    # ---------------- other region materials (fine group) ----------------
    nZ = len(design.zones)
    salt_macro = build_macro(salt, lib, homogeneous_sig0(salt), perturb)
    refl_mat = grap if design.reflector_material == "graphite" else hastelloy_n(Tf)
    refl_macro = build_macro(refl_mat, lib, None, perturb)
    vessel_macro = build_macro(hastelloy_n(Tf), lib, None, perturb)
    ax_refl_macro = refl_macro
    rod = design.control_rod
    rod_macro = None
    if rod is not None and rod.insertion > 0:
        absorber = b4c(b10_fraction=rod.b10_fraction, T=Tg)
        rod_mat = mix([absorber, grap], [rod.absorber_fraction, 1 - rod.absorber_fraction],
                      name="control rod")
        rod_macro = build_macro(rod_mat, lib, None, perturb)
    names = [z.name for z in design.zones] + ["radial reflector", "downcomer", "vessel",
                                              "lower plenum", "upper plenum",
                                              "lower axial reflector", "upper axial reflector",
                                              "control rod"]
    fine_xs = [l.hom for l in lat] + [refl_macro, salt_macro, vessel_macro, salt_macro,
                                      salt_macro, ax_refl_macro, ax_refl_macro,
                                      rod_macro if rod_macro is not None else refl_macro]
    iR, iDC, iV, iLP, iUP, iLA, iUA, iROD = range(nZ, nZ + 8)
    # ---------------- 1D radial spectral-zone calculation ----------------
    h1 = max(mesh_size / 2.0, 1.0)
    rseg = []
    r0 = 0.0
    for z in design.zones:
        rseg.append((z.outer_radius - r0, z.name))
        r0 = z.outer_radius
    rseg += [(design.radial_reflector, "refl"), (design.downcomer, "dc"), (design.vessel, "v")]
    r_edges = _edges_for(rseg, h1)
    rc = 0.5 * (r_edges[1:] + r_edges[:-1])
    bounds = np.cumsum([0.0] + [L for L, _ in rseg])
    reg1 = np.zeros(len(rc), dtype=int)
    ids_r = list(range(nZ)) + [iR, iDC, iV]
    for k, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        reg1[(rc >= lo) & (rc < hi)] = ids_r[k]
    H_eff = design.core_height + design.lower_plenum + design.upper_plenum + 20.0
    mesh_r = Mesh("cylinder", r_edges, region=reg1[:, None], transverse_B2=(np.pi / H_eff) ** 2,
                  bc={"r_max": "vacuum"})
    res_r = DiffusionSolver(mesh_r, fine_xs).solve()
    spec_r = _region_spectra_1d(res_r, len(fine_xs))
    # ---------------- 1D axial spectral-zone calculation ----------------
    zseg = [(design.axial_reflector, iLA), (design.lower_plenum, iLP),
            (design.core_height, 0), (design.upper_plenum, iUP), (design.axial_reflector, iUA)]
    z_edges = _edges_for(zseg, h1)
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])
    zb = np.cumsum([0.0] + [L for L, _ in zseg])
    reg_z = np.zeros(len(zc), dtype=int)
    for (L, rid), lo, hi in zip(zseg, zb[:-1], zb[1:]):
        reg_z[(zc >= lo) & (zc < hi)] = rid
    R_eff = design.core_radius + 0.5 * design.radial_reflector
    mesh_z = Mesh("slab", z_edges, region=reg_z[:, None], transverse_B2=(2.405 / R_eff) ** 2,
                  bc={"r_min": "vacuum", "r_max": "vacuum"})
    res_z = DiffusionSolver(mesh_z, fine_xs).solve()
    spec_z = _region_spectra_1d(res_z, len(fine_xs))
    t2 = time.time()
    # ---------------- condensation ----------------
    spectra = {}
    for rid in range(len(fine_xs)):
        if rid in (iLP, iUP, iLA, iUA) and rid in spec_z:
            spectra[rid] = spec_z[rid]
        elif rid in spec_r:
            spectra[rid] = spec_r[rid]
        elif rid == iROD:
            spectra[rid] = spec_r[0]
        else:
            spectra[rid] = spec_r.get(iR, spec_r[0])
    coarse_xs = [condense(fine_xs[rid], spectra[rid], coarse) for rid in range(len(fine_xs))]
    for rid, x in enumerate(coarse_xs):
        x.name = names[rid]
    # ---------------- 2D R-Z mesh ----------------
    h = mesh_size
    rseg2 = []
    r0 = 0.0
    if rod_macro is not None:
        rseg2.append((rod.radius, "rod"))
        r0 = rod.radius
    for z in design.zones:
        rseg2.append((z.outer_radius - r0, z.name))
        r0 = z.outer_radius
    rseg2 += [(design.radial_reflector, "refl"), (design.downcomer, "dc"),
              (design.vessel, "v")]
    r2 = _edges_for(rseg2, h)
    z_core0 = design.axial_reflector + design.lower_plenum
    z_core1 = z_core0 + design.core_height
    zseg2 = [(design.axial_reflector, "la"), (design.lower_plenum, "lp")]
    if rod_macro is not None and 0 < rod.insertion < design.core_height:
        zseg2 += [(design.core_height - rod.insertion, "core-lo"), (rod.insertion, "core-hi")]
    else:
        zseg2 += [(design.core_height, "core")]
    zseg2 += [(design.upper_plenum, "up"), (design.axial_reflector, "ua")]
    z2 = _edges_for(zseg2, h)
    rc2 = 0.5 * (r2[1:] + r2[:-1])
    zc2 = 0.5 * (z2[1:] + z2[:-1])
    RR, ZZ = np.meshgrid(rc2, zc2, indexing="ij")
    region = np.full(RR.shape, iV, dtype=int)
    R_core = design.core_radius
    R_ref = R_core + design.radial_reflector
    R_dc = R_ref + design.downcomer
    in_core_z = (ZZ >= z_core0) & (ZZ < z_core1)
    r0 = 0.0
    for zi, z in enumerate(design.zones):
        region[(RR >= r0) & (RR < z.outer_radius) & in_core_z] = zi
        r0 = z.outer_radius
    region[(RR >= R_core) & (RR < R_ref) & in_core_z] = iR
    region[(RR < R_ref) & (ZZ >= design.axial_reflector) & (ZZ < z_core0)] = iLP
    region[(RR < R_ref) & (ZZ >= z_core1) & (ZZ < z_core1 + design.upper_plenum)] = iUP
    region[(RR < R_ref) & (ZZ < design.axial_reflector)] = iLA
    region[(RR < R_ref) & (ZZ >= z_core1 + design.upper_plenum)] = iUA
    region[(RR >= R_ref) & (RR < R_dc)] = iDC
    if rod_macro is not None:
        tip = z_core1 - min(rod.insertion, design.core_height)
        region[(RR < rod.radius) & (ZZ >= tip) & (ZZ < z_core1)] = iROD
    mesh2 = Mesh("rz", r2, z2, region=region,
                 bc={"r_max": "vacuum", "z_min": "vacuum", "z_max": "vacuum"})
    solver = DiffusionSolver(mesh2, coarse_xs)
    dres = solver.solve(n_modes=n_modes)
    if adjoint:
        dres.adjoint = solver.solve(adjoint=True).flux
    dres.normalize(design.power_MW * 1e6)
    t3 = time.time()
    res = CoreResult(design, dres.k, dres, coarse, names, coarse_xs, lat,
                     {"radial": spec_r, "axial": spec_z, "used": spectra})
    res.k_inf_zones = [l.k_inf for l in lat]
    res.timings = {"lattice": t1 - t0, "spectral_1d": t2 - t1, "core_2d": t3 - t2,
                   "total": t3 - t0}
    _postprocess(res, salt, lat, salt_macro, spectra, coarse, nZ,
                 (iR, iDC, iV, iLP, iUP, iLA, iUA, iROD), lib)
    return res


def _postprocess(res: CoreResult, salt, lat, salt_macro, spectra, coarse, nZ, ids, lib):
    iR, iDC, iV, iLP, iUP, iLA, iUA, iROD = ids
    d = res.diffusion
    design = res.design
    mesh = d.mesh
    V = mesh.volumes()
    fine = lib.groups
    cmap = collapse_map(fine, coarse.bounds)
    # ---- power distribution ----
    q = d.power_density()
    P_tot = float((q * V).sum())
    in_zones = mesh.region < nZ
    res.power_fraction_outside_zones = 1.0 - float((q * V)[in_zones].sum()) / P_tot
    # ---- leakage fraction ----
    prod = float(((d.field("nu_fission") * d.flux).sum(axis=0) * V).sum()) / d.k
    absr = 0.0
    for rid, xs in enumerate(res.region_xs):
        m = mesh.region == rid
        if np.any(m):
            phi_r = (d.flux[:, m] * V[m][None]).sum(axis=1)
            absr += float(((xs.total - xs.scatter.sum(axis=1)) * phi_r).sum())
    res.leakage_fraction = max(1.0 - absr / prod, 0.0)
    # net leakage out of the fuelled core zones (fraction of neutrons born there)
    zmask = mesh.region < nZ
    born_z = float(((d.field("nu_fission") * d.flux).sum(axis=0) * V)[zmask].sum()) / d.k
    abs_z = 0.0
    for rid in range(nZ):
        m = mesh.region == rid
        if np.any(m):
            xs = res.region_xs[rid]
            phi_r = (d.flux[:, m] * V[m][None]).sum(axis=1)
            abs_z += float(((xs.total - xs.scatter.sum(axis=1)) * phi_r).sum())
    res.core_leakage_fraction = (born_z - abs_z) / born_z if born_z > 0 else 0.0
    # ---- fine-group salt flux integrals for one-group XS ----
    salt_regions = list(range(nZ)) + [iDC, iLP, iUP]
    rr = {}
    phi_salt_tot = 0.0
    hm_nucs = [n for n in salt.N if NUCLIDES[n].fissionable]
    prod_by_nuc = {}
    for rid in salt_regions:
        m = mesh.region == rid
        if not np.any(m):
            continue
        Phi_G = (d.flux[:, m] * V[m][None]).sum(axis=1)              # coarse, cell-averaged
        s = spectra[rid]
        sG = np.bincount(cmap, weights=s, minlength=coarse.G)
        Phi_g = Phi_G[cmap] * s / np.maximum(sG[cmap], 1e-300)       # fine, cell-averaged
        if rid < nZ:
            L = lat[rid]
            Phi_salt = Phi_g * L.fuel_flux_ratio * L.V_fuel
            micro = dict(L.fuel_macro.micro)
        else:
            Phi_salt = Phi_g
            micro = dict(salt_macro.micro)
        # trace nuclides absent from the salt (fission products, minor actinides):
        # infinitely dilute cross sections at the salt temperature
        for nuc in lib.nuclides:
            if nuc not in micro:
                micro[nuc] = lib[nuc].micro(salt.T, 1e10)
        phi_salt_tot += Phi_salt.sum()
        for nuc, mic in micro.items():
            acc = rr.setdefault(nuc, {"capture": 0.0, "fission": 0.0, "nalpha": 0.0, "n2n": 0.0,
                                      "tritium": 0.0, "absorption": 0.0, "nu_fission": 0.0})
            for rx in ("capture", "fission", "nalpha", "n2n", "tritium", "absorption"):
                acc[rx] += float(mic[rx] @ Phi_salt)
            acc["nu_fission"] += float((mic["nu"] * mic["fission"]) @ Phi_salt)
    res.one_group = {nuc: {rx: v / phi_salt_tot for rx, v in acc.items()} for nuc, acc in rr.items()}
    V_salt_total = design.salt_volumes_m3()["total"] * 1e6
    res.salt_flux_avg = phi_salt_tot / V_salt_total
    # tritium production: Li-6(n,t) + Li-7(n,n't)
    Nli6, Nli7 = salt.N.get("Li6", 0.0), salt.N.get("Li7", 0.0)
    og = res.one_group
    t_rate = 0.0
    if "Li6" in og:
        t_rate += Nli6 * og["Li6"]["nalpha"]
    if "Li7" in og:
        t_rate += Nli7 * og["Li7"]["tritium"]
    res.tritium_rate = t_rate * phi_salt_tot  # atoms/s  (N[1/b-cm]*sigma[b] = 1/cm)
    # ---- conversion ratio from one-group rates ----
    N = salt.N

    def R(nuc, rx):
        return N.get(nuc, 0.0) * og.get(nuc, {}).get(rx, 0.0)

    gain = R("Th232", "capture") + R("U238", "capture") + R("Pu240", "capture") + R("U234", "capture")
    loss = sum(R(n, "absorption") for n in ("U233", "U235", "Pu239", "Pu241"))
    res.conversion_ratio = gain / loss if loss > 0 else 0.0
    # ---- delayed neutron data & adjoint weighting ----
    nf_rates = {n: N.get(n, 0.0) * og[n]["nu_fission"] for n in og if og[n]["nu_fission"] > 0}
    dd = mix_delayed(nf_rates)
    res.beta_i = dd.beta_i
    res.lambda_i = dd.lam_i
    if d.adjoint is not None:
        adj = d.adjoint
        chi = d.field("chi")
        chid = d.field("chi_d")
        S = (d.field("nu_fission") * d.flux).sum(axis=0) * V
        Ip = float((adj * chi * S[None]).sum())
        Id = float((adj * chid * S[None]).sum())
        ratio = Id / Ip if Ip > 0 else 1.0
        res.beta_static = dd.beta * ratio
        invv = d.field("invv")
        res.generation_time = float((adj * invv * d.flux * V[None]).sum()) / Ip
        # axial shapes over the core zones (for circulating-fuel kinetics)
        zone_mask = (mesh.region < nZ) | (mesh.region == iLP) | (mesh.region == iUP)
        fz = np.where(zone_mask, S, 0.0).sum(axis=0)
        Iz = np.where(zone_mask, (adj * chi).sum(axis=0) * S, 0.0).sum(axis=0)
        keep = fz > 0
        res.axial_fission_shape = np.vstack([mesh.zc[keep], fz[keep] / fz[keep].sum()])
        imp = np.where(fz[keep] > 0, Iz[keep] / np.maximum(fz[keep], 1e-300), 0.0)
        res.axial_importance_shape = np.vstack([mesh.zc[keep], imp / imp.max()])
    else:
        res.beta_static = dd.beta
    res.beta_eff = res.beta_static
    # ---- graphite fast flux (E > 50 keV) ----
    frac_fast = np.zeros(fine.G)
    frac_fast[fine.lower >= 50e3] = 1.0
    gi = fine.group_of(50e3)
    frac_fast[gi] = np.log(fine.upper[gi] / 50e3) / np.log(fine.upper[gi] / fine.lower[gi])
    fast_peak = 0.0
    for rid in list(range(nZ)) + [iR, iLA, iUA]:
        m = mesh.region == rid
        if not np.any(m):
            continue
        s = spectra[rid]
        sG = np.bincount(cmap, weights=s, minlength=coarse.G)
        fG = np.bincount(cmap, weights=s * frac_fast, minlength=coarse.G) / np.maximum(sG, 1e-300)
        ff = (d.flux[:, m] * fG[:, None]).sum(axis=0)
        fast_peak = max(fast_peak, float(ff.max()))
    res.fast_flux_graphite_peak = fast_peak
    if fast_peak > 0:
        res.graphite_lifetime_years = GRAPHITE_FLUENCE_LIMIT / (
            fast_peak * design.capacity_factor * SECONDS_PER_YEAR)


# ---------------------------------------------------------------------------
# Reference designs
# ---------------------------------------------------------------------------
def msre_design() -> MSRDesign:
    """Molten Salt Reactor Experiment-like core (8 MWth, U-235 fuel)."""
    return MSRDesign(
        name="MSRE-like",
        salt_composition={"LiF": 0.65, "BeF2": 0.291, "ZrF4": 0.05, "UF4": 0.009},
        uranium=0.33, li7_enrichment=0.99993,
        zones=[CoreZone("graphite lattice", 70.2, 0.225, 0.99, "square")],
        core_height=166.0, radial_reflector=0.0, downcomer=3.0, vessel=2.5,
        lower_plenum=15.0, upper_plenum=15.0, axial_reflector=0.0,
        power_MW=8.0, T_fuel=922.0, T_graphite=930.0, graphite_density=1.86,
        flow_rate_m3s=0.0757, external_loop_volume_m3=0.60, capacity_factor=0.8)


def msbr_design() -> MSRDesign:
    """Two-zone single-fluid Molten Salt Breeder Reactor-like core (2250 MWth)."""
    return MSRDesign(
        name="MSBR-like",
        salt_composition={"LiF": 0.7175, "BeF2": 0.16, "ThF4": 0.12, "UF4": 0.0025},
        uranium={"U233": 1.0}, li7_enrichment=0.99995,
        zones=[CoreZone("zone I", 215.0, 0.13, 1.0, "hex"),
               CoreZone("zone II", 260.0, 0.37, 2.0, "hex")],
        core_height=400.0, radial_reflector=76.0, downcomer=5.0, vessel=5.0,
        lower_plenum=15.0, upper_plenum=15.0, axial_reflector=60.0,
        power_MW=2250.0, T_fuel=900.0, T_graphite=920.0,
        flow_rate_m3s=3.4, external_loop_volume_m3=20.0, capacity_factor=0.8)


def tmsr_leu_design() -> MSRDesign:
    """Generic small thermal MSR burning 5 % LEU in FLiBe (250 MWth, IMSR-like)."""
    return MSRDesign(
        name="TMSR-LEU",
        salt_composition={"LiF": 0.69, "BeF2": 0.30, "UF4": 0.010},
        uranium=0.05, li7_enrichment=0.99995,
        zones=[CoreZone("core", 150.0, 0.20, 2.5, "hex")],
        core_height=300.0, radial_reflector=50.0, downcomer=5.0, vessel=5.0,
        lower_plenum=15.0, upper_plenum=15.0, axial_reflector=40.0,
        power_MW=250.0, T_fuel=900.0, T_graphite=920.0,
        flow_rate_m3s=1.0, external_loop_volume_m3=8.0)


def fast_chloride_design() -> MSRDesign:
    """Unmoderated chloride fast-spectrum MSR (NaCl-UCl3, Cl-37 enriched)."""
    return MSRDesign(
        name="Fast chloride MSR",
        salt_composition={"NaCl": 0.58, "UCl3": 0.42}, uranium=0.16,
        cl37_enrichment=0.97, li7_enrichment=0.99995,
        zones=[CoreZone("fuel salt", 160.0, 1.0, 1.0)],
        core_height=320.0, radial_reflector=60.0, reflector_material="steel",
        downcomer=0.0, vessel=5.0, lower_plenum=0.0, upper_plenum=0.0, axial_reflector=60.0,
        power_MW=1000.0, T_fuel=950.0, T_graphite=950.0,
        flow_rate_m3s=2.0, external_loop_volume_m3=15.0)


PRESETS = {"msre": msre_design, "msbr": msbr_design, "tmsr": tmsr_leu_design,
           "fast_chloride": fast_chloride_design}
