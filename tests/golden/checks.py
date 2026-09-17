"""Dispatch layer for the golden vector suite: input dict in, output dict out.

Each function here takes the ``inputs`` mapping of one vector file in
``data/golden/`` and returns a flat mapping of output name to float. It calls
the platform code and does no arithmetic of its own beyond unit stripping, so
that the comparison performed in ``test_golden_vectors.py`` is between a value
this repository computes and a value derived by hand in
``docs/golden-vectors.md``.

Two rules hold for every function below.

First, no function may reproduce a formula that the module under test also
implements. Where a check needs a second route to the same number (the
round trip of a kinetic inversion, for example) it uses the module's OWN
forward function, so a wrong formula cannot be cancelled by the same wrong
formula written twice.

Second, the returned keys are the contract. A vector file that names an output
this function does not return fails on a missing key rather than passing
silently, which is checked by a dedicated control.
"""
from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ae.agent.decisions import Action, DecisionProblem, evpi
from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.site import Currency, LabourRates, PowerSupply, ReagentPrices, Site
from ae.core.units import Q_
from ae.econ.capex import Equipment, escalate_cost, estimate_capex, scale_cost
from ae.econ.uncertainty import Uncertain, monte_carlo, sobol_analysis
from ae.econ.unit_economics import InputDemand, cash_cost
from ae.econ.valuation import Project, breakeven_price, levelized_cost, npv, payback_period
from ae.ml.surrogate import TrainingSet, evaluate_surrogate, leave_one_deposit_out_splits
from ae.physics.chlorination import CHLORIDES, gibbs_of_reaction, vapour_pressure
from ae.physics.comminution import (
    TonConvention,
    bond_specific_energy,
    bond_work_index_from_energy,
    convert_ton_convention,
)
from ae.physics.impurity_location import (
    ElementPartition,
    PartitionModel,
    floor_concentration,
    implied_removal_rate,
    lattice_ceiling_sio2_percent,
    location_concentrations,
    removable_ppm,
)
from ae.physics.leaching import (
    Arrhenius,
    LeachSystem,
    Regime,
    conversion,
    g_of_conversion,
    tau_film,
    tau_from_single_point,
    tau_product_layer,
    tau_surface_reaction,
)
from ae.physics.liberation import enclosed_fraction, exposure, leachable_fraction, liberation_size
from ae.physics.packing import (
    SizeClass,
    furnas_max_packing,
    furnas_optimal_composition,
    krieger_dougherty_relative_viscosity,
    volume_to_mass_fraction,
)
from ae.physics.phases import Polymorph
from ae.physics.psd import (
    LogNormalPSD,
    RosinRammlerPSD,
    Weighting,
    specific_surface_area,
)
from ae.physics.separation import (
    flotation_recovery,
    logistic_scale_from_ep,
    partition_curve,
    two_product,
)
from ae.physics.thermal import (
    integrated_enthalpy,
    landau_excess_heat_capacity,
    molar_heat_capacity,
)
from ae.plant.capacity import OEE, UnitCapacity, assess_line
from ae.plant.scheduling import allen_cunneen_waiting_time, mm1_waiting_time
from ae.plant.streams import Flowsheet, Stream, UnitOp
from ae.plant.yield_cascade import (
    capability,
    cascade_yield,
    off_spec_fraction,
    required_process_mean,
    stage_throughput_factors,
)

# A fixture source. It carries no measurement and must never be cited as
# evidence for any deposit property. Every vector that needs a Feedstock or a
# Site needs provenance objects to exist, and inventing a real citation for a
# test fixture would put an unverified reference into the repository.
FIXTURE_SRC = Source(
    citation="Golden vector fixture, not a real reference",
    tier=Tier.T2,
    url="https://example.invalid/golden-vector-fixture",
    accessed=dt.date(2026, 9, 17),
    note="Synthetic fixture for the golden vector suite. Carries no real "
         "measurement. The Vikarabad deposit is uncharacterized in the citable "
         "record and no number in this file is a property of that ore.",
)


def _v(q: Any, tag: Tag = Tag.ASSUMED) -> Value:
    if tag in (Tag.SOURCED, Tag.MEASURED):
        return Value(quantity=q, tag=tag, source=FIXTURE_SRC)
    return Value(quantity=q, tag=tag, basis="golden vector fixture input")


def _mag(q: Any, unit: str) -> float:
    return float(q.to(unit).magnitude)


# ---------------------------------------------------------------------------
# comminution
# ---------------------------------------------------------------------------

def bond_energy(i: Mapping[str, Any]) -> dict[str, float]:
    conv = TonConvention(i["convention"])
    w = bond_specific_energy(
        work_index=Q_(i["work_index"], i["work_index_unit"]),
        f80=Q_(i["f80_um"], "um"),
        p80=Q_(i["p80_um"], "um"),
        convention=conv,
    )
    metric = convert_ton_convention(w, TonConvention.METRIC)
    return {
        "specific_energy_kwh_per_short_ton": _mag(w, "kWh/short_ton"),
        "specific_energy_kwh_per_metric_tonne": _mag(metric, "kWh/tonne"),
    }


def bond_inversion(i: Mapping[str, Any]) -> dict[str, float]:
    wi = bond_work_index_from_energy(
        specific_energy=Q_(i["specific_energy"], i["energy_unit"]),
        f80=Q_(i["f80_um"], "um"),
        p80=Q_(i["p80_um"], "um"),
        convention=TonConvention(i["convention"]),
    )
    # Round trip through the forward function, which is the module's own
    # separate code path, not a restatement of the inversion.
    back = bond_specific_energy(
        work_index=wi,
        f80=Q_(i["f80_um"], "um"),
        p80=Q_(i["p80_um"], "um"),
        convention=TonConvention(i["convention"]),
    )
    return {
        "operating_work_index": _mag(wi, i["work_index_unit"]),
        "round_trip_specific_energy": _mag(back, i["energy_unit"]),
    }


# ---------------------------------------------------------------------------
# particle size distribution
# ---------------------------------------------------------------------------

def rosin_rammler(i: Mapping[str, Any]) -> dict[str, float]:
    psd = RosinRammlerPSD(
        d_prime=Q_(i["d_prime_um"], "um"),
        n=i["n"],
        density=Q_(i["density_kg_m3"], "kg/m**3"),
    )
    return {
        "d10_um": _mag(psd.quantile(0.10), "um"),
        "d50_um": _mag(psd.median(), "um"),
        "d90_um": _mag(psd.quantile(0.90), "um"),
        "span": psd.span(),
        "sauter_d32_um": _mag(psd.sauter_d32(), "um"),
        "volume_weighted_mean_um": _mag(psd.volume_weighted_mean(), "um"),
        "specific_surface_area_m2_per_kg": _mag(psd.specific_surface_area(), "m**2/kg"),
        "cumulative_undersize_at_d_prime": psd.cumulative_undersize(Q_(i["d_prime_um"], "um")),
    }


def lognormal_psd(i: Mapping[str, Any]) -> dict[str, float]:
    psd = LogNormalPSD(
        d_gn=Q_(i["d_gn_um"], "um"),
        sigma_g=i["sigma_g"],
        density=Q_(i["density_kg_m3"], "kg/m**3"),
    )
    d32 = psd.sauter_d32()
    ssa = specific_surface_area(d32, Q_(i["density_kg_m3"], "kg/m**3"))
    return {
        "s": psd.s,
        "number_median_um": _mag(psd.median(Weighting.NUMBER), "um"),
        "area_median_um": _mag(psd.median(Weighting.AREA), "um"),
        "volume_median_um": _mag(psd.median(Weighting.VOLUME), "um"),
        "sauter_d32_um": _mag(d32, "um"),
        "specific_surface_area_m2_per_kg": _mag(ssa, "m**2/kg"),
        "volume_span": psd.span(Weighting.VOLUME),
    }


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------

def furnas_packing(i: Mapping[str, Any]) -> dict[str, float]:
    classes = tuple(SizeClass(diameter=Q_(d, "um")) for d in i["diameters_um"])
    r = furnas_max_packing(classes)
    comp = furnas_optimal_composition(len(classes))
    out: dict[str, float] = {
        "n_classes": float(r.n_classes),
        "phi_monomodal": r.phi_monomodal,
        "phi_max": r.phi_max,
        "composition_sum": float(sum(comp)),
    }
    for k, x in enumerate(comp):
        out[f"composition_{k}"] = x
    for k, x in enumerate(r.size_ratios):
        out[f"size_ratio_{k}"] = x
    return out


def krieger_dougherty(i: Mapping[str, Any]) -> dict[str, float]:
    eta = krieger_dougherty_relative_viscosity(
        phi=i["phi"], phi_max=i["phi_max"],
        intrinsic_viscosity=_v(Q_(i["intrinsic_viscosity"], "dimensionless")),
    )
    mass_fraction = volume_to_mass_fraction(
        phi=i["phi"],
        filler_density=Q_(i["filler_density_kg_m3"], "kg/m**3"),
        matrix_density=Q_(i["matrix_density_kg_m3"], "kg/m**3"),
    )
    return {"relative_viscosity": eta, "mass_fraction": mass_fraction}


# ---------------------------------------------------------------------------
# liberation
# ---------------------------------------------------------------------------

def liberation_geometry(i: Mapping[str, Any]) -> dict[str, float]:
    dp = Q_(i["particle_size_um"], "um")
    di = Q_(i["inclusion_size_um"], "um")
    return {
        "enclosed_fraction": enclosed_fraction(dp, di),
        "exposure": exposure(dp, di),
        "closure": enclosed_fraction(dp, di) + exposure(dp, di),
        "liberation_size_um": _mag(liberation_size(di, i["target_exposure"]), "um"),
    }


def liberation_leachable(i: Mapping[str, Any]) -> dict[str, float]:
    f = leachable_fraction(
        partition=i["partition"],
        particle_size=Q_(i["particle_size_um"], "um"),
        fluid_inclusion_size=Q_(i["fluid_inclusion_size_um"], "um"),
        mineral_inclusion_size=Q_(i["mineral_inclusion_size_um"], "um"),
    )
    dp = Q_(i["particle_size_um"], "um")
    return {
        "exposure_fluid": exposure(dp, Q_(i["fluid_inclusion_size_um"], "um")),
        "exposure_mineral": exposure(dp, Q_(i["mineral_inclusion_size_um"], "um")),
        "leachable_fraction": f,
        "ceiling_one_minus_lattice": 1.0 - i["partition"]["lattice"],
    }


# ---------------------------------------------------------------------------
# impurity partition
# ---------------------------------------------------------------------------

def _feedstock_with_partition(i: Mapping[str, Any]) -> tuple[Feedstock, PartitionModel]:
    el = i["element"]
    prof = ImpurityProfile(
        total={el: _v(Q_(i["bulk_ppm"], "ppm_mass"), Tag.SOURCED)},
        method="LA_ICP_MS",
        basis_material="single_grain",
    )
    fs = Feedstock(
        sample_id=i["sample_id"], ore_type=OreType(i["ore_type"]),
        deposit_name="Golden vector fixture", country="IN", impurities=prof,
    )
    p = i["partition"]
    model = PartitionModel(
        sample_id=i["sample_id"],
        partitions={el: ElementPartition(
            element=el,
            surface=_v(Q_(p["surface"], "dimensionless"), Tag.MEASURED),
            fluid=_v(Q_(p["fluid"], "dimensionless"), Tag.MEASURED),
            mineral=_v(Q_(p["mineral"], "dimensionless"), Tag.MEASURED),
            lattice=_v(Q_(p["lattice"], "dimensionless"), Tag.MEASURED),
            method="LA_ICP_MS_inclusion_free_domains",
        )},
    )
    return fs, model


def impurity_floor(i: Mapping[str, Any]) -> dict[str, float]:
    fs, model = _feedstock_with_partition(i)
    el = i["element"]
    conc = location_concentrations(fs, el, model=model)
    floor = floor_concentration(fs, el, model=model, efficiencies=i["efficiencies"])
    rem = removable_ppm(fs, el, model=model)
    return {
        "conc_surface_ppm": conc["surface"],
        "conc_fluid_ppm": conc["fluid"],
        "conc_mineral_ppm": conc["mineral"],
        "conc_lattice_ppm": conc["lattice"],
        "conc_sum_ppm": float(math.fsum(conc.values())),
        "floor_ppm": floor,
        "removable_ppm": rem,
        "removable_plus_lattice_ppm": rem + conc["lattice"],
    }


def purity_ceiling(i: Mapping[str, Any]) -> dict[str, float]:
    return {
        "implied_removal_fraction": implied_removal_rate(i["feed_sum_ppm"], i["product_sum_ppm"]),
        "implied_removal_percent": implied_removal_rate(
            i["feed_sum_ppm"], i["product_sum_ppm"]) * 100.0,
        "sio2_percent_from_product": lattice_ceiling_sio2_percent(i["product_sum_ppm"]),
        "sio2_percent_from_floor": lattice_ceiling_sio2_percent(i["floor_sum_ppm"]),
    }


# ---------------------------------------------------------------------------
# leach kinetics
# ---------------------------------------------------------------------------

def _leach_system(i: Mapping[str, Any]) -> LeachSystem:
    prof = ImpurityProfile(
        total={"Fe": _v(Q_(i["bulk_fe_ppm"], "ppm_mass"), Tag.SOURCED)},
        lattice_fraction={"Fe": _v(Q_(i["fe_lattice_fraction"], "dimensionless"), Tag.MEASURED)},
        method="LA_ICP_MS",
    )
    # characterized is left FALSE deliberately. The fixture carries one element
    # (Fe) with a lattice split, which puts it at tier 'screened'; the flag
    # requires tier 'located', a full element suite by a spatially resolved
    # method. Claiming characterization here would assert evidence the fixture
    # does not have, and the Vikarabad deposit is uncharacterized in the
    # citable record in any case. The leach arithmetic does not read the flag.
    fs = Feedstock(
        sample_id=i["sample_id"], ore_type=OreType.VEIN_QUARTZ,
        deposit_name="Golden vector fixture", country="IN", impurities=prof,
        characterized=False,
    )
    t_k = Q_(i["temperature_K"], "K")
    return LeachSystem(
        feedstock=fs, element="Fe", reagent=i["reagent"],
        particle_radius=_v(Q_(i["particle_radius_m"], "m")),
        reagent_concentration=_v(Q_(i["concentration_mol_m3"], "mol/m**3")),
        solid_molar_density=_v(Q_(i["solid_molar_density_mol_m3"], "mol/m**3")),
        stoich_b=i["stoich_b"],
        temperature=_v(t_k),
        # Zero activation energy makes k(T) equal the prefactor exactly, so the
        # tau arithmetic is traceable without evaluating an exponential.
        film_coefficient=Arrhenius(
            prefactor=_v(Q_(i["k_film_m_s"], "m/s")),
            activation_energy=_v(Q_(0.0, "kJ/mol"))),
        product_layer_diffusivity=Arrhenius(
            prefactor=_v(Q_(i["diffusivity_m2_s"], "m**2/s")),
            activation_energy=_v(Q_(0.0, "kJ/mol"))),
        surface_rate_constant=Arrhenius(
            prefactor=_v(Q_(i["k_surface_m_s"], "m/s")),
            activation_energy=_v(Q_(0.0, "kJ/mol"))),
    )


def leach_characteristic_times(i: Mapping[str, Any]) -> dict[str, float]:
    s = _leach_system(i)
    return {
        "tau_film_s": _mag(tau_film(s), "s"),
        "tau_product_layer_s": _mag(tau_product_layer(s), "s"),
        "tau_surface_reaction_s": _mag(tau_surface_reaction(s), "s"),
    }


def leach_conversion(i: Mapping[str, Any]) -> dict[str, float]:
    theta = i["t_over_tau"]
    tau = Q_(1.0, "s")
    t = Q_(theta, "s")
    out: dict[str, float] = {}
    for name, reg in (("film", Regime.FILM),
                      ("product_layer", Regime.PRODUCT_LAYER),
                      ("surface_reaction", Regime.SURFACE_REACTION)):
        x = conversion(reg, t, tau)
        out[f"x_{name}"] = x
        # Round trip through g(X), which is the module's separate forward
        # relation: g(X(theta)) must return theta. g_of_conversion returns a
        # 1-element ndarray for a scalar argument, so np.asarray(...) is shape
        # (1,) and float() on it raises; ravel()[0] is the scalar. A first
        # draft used float(np.asarray(...)) and failed with "only
        # 0-dimensional arrays can be converted to Python scalars".
        out[f"g_round_trip_{name}"] = float(
            np.asarray(g_of_conversion(reg, x)).ravel()[0]
        )
    x_ref = i["conversion_x"]
    for name, reg in (("film", Regime.FILM),
                      ("product_layer", Regime.PRODUCT_LAYER),
                      ("surface_reaction", Regime.SURFACE_REACTION)):
        out[f"g_at_x_{name}"] = float(
            np.asarray(g_of_conversion(reg, x_ref)).ravel()[0]
        )
    return out


def leach_tau_from_point(i: Mapping[str, Any]) -> dict[str, float]:
    reg = Regime(i["regime"])
    tau = tau_from_single_point(reg, i["conversion_x"], Q_(i["t_s"], "s"))
    # Forward check: at t = t_s the module's conversion must return X again.
    x_back = conversion(reg, Q_(i["t_s"], "s"), tau)
    return {"tau_s": _mag(tau, "s"), "round_trip_conversion": x_back}


# ---------------------------------------------------------------------------
# chlorination
# ---------------------------------------------------------------------------

def chlorination_volatility(i: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in i["species"]:
        sp = CHLORIDES[key]
        p, capped = vapour_pressure(
            sp, Q_(i["temperature_K"], "K"), Q_(i["total_pressure_Pa"], "Pa")
        )
        out[f"{key}_pressure_Pa"] = _mag(p, "Pa")
        out[f"{key}_capped"] = 1.0 if capped else 0.0
        # Named transition_enthalpy and not trouton_enthalpy: TiCl4 carries a
        # MEASURED PubChem heat of vaporization (36.2 kJ/mol), while AlCl3 and
        # NaCl fall back to a Trouton estimate. Calling the key trouton would
        # assert an estimate where a measurement exists.
        out[f"{key}_transition_enthalpy_J_per_mol"] = _mag(sp.enthalpy(), "J/mol")
        out[f"{key}_enthalpy_is_trouton"] = 1.0 if sp.enthalpy_is_trouton else 0.0
        out[f"{key}_transition_T_K"] = _mag(sp.transition_T.quantity, "K")
    return out


def chlorination_gibbs(i: Mapping[str, Any]) -> dict[str, float]:
    table = {
        k: {"Hf": _v(Q_(v["Hf_kJ_per_mol"], "kJ/mol"), Tag.SOURCED),
            "S": _v(Q_(v["S_J_per_mol_K"], "J/(mol*K)"), Tag.SOURCED)}
        for k, v in i["thermo"].items()
    }
    dg = gibbs_of_reaction(i["stoichiometry"], Q_(i["temperature_K"], "K"), thermo=table)
    return {"delta_g_kJ_per_mol": _mag(dg, "kJ/mol")}


# ---------------------------------------------------------------------------
# thermal
# ---------------------------------------------------------------------------

def thermal_enthalpy(i: Mapping[str, Any]) -> dict[str, float]:
    dh = integrated_enthalpy(
        Polymorph(i["polymorph"]),
        Q_(i["t_from_K"], "K"),
        Q_(i["t_to_K"], "K"),
        include_landau=i["include_landau"],
        allow_extrapolation=i.get("allow_extrapolation", False),
    )
    return {
        "delta_h_J_per_mol": _mag(dh, "J/mol"),
        "delta_h_kJ_per_kg": _mag(dh / Q_(i["molar_mass_kg_per_mol"], "kg/mol"), "kJ/kg"),
    }


def thermal_heat_capacity(i: Mapping[str, Any]) -> dict[str, float]:
    t = Q_(i["temperature_K"], "K")
    ph = Polymorph(i["polymorph"])
    base = molar_heat_capacity(ph, t, include_landau=False)
    total = molar_heat_capacity(ph, t, include_landau=True)
    excess = landau_excess_heat_capacity(t)
    return {
        "cp_base_J_per_mol_K": _mag(base, "J/(mol*K)"),
        "landau_excess_J_per_mol_K": _mag(excess, "J/(mol*K)"),
        "cp_total_J_per_mol_K": _mag(total, "J/(mol*K)"),
        "sum_check_J_per_mol_K": _mag(base, "J/(mol*K)") + _mag(excess, "J/(mol*K)"),
    }


# ---------------------------------------------------------------------------
# separation
# ---------------------------------------------------------------------------

def separation_recovery(i: Mapping[str, Any]) -> dict[str, float]:
    r = flotation_recovery(
        time=Q_(i["time_s"], "s"),
        rate_constant=Q_(i["k_per_s"], "1/s"),
        ultimate_recovery=i["ultimate_recovery"],
    )
    y, rec = two_product(i["feed_grade"], i["concentrate_grade"], i["tailing_grade"])
    s = logistic_scale_from_ep(i["ep"])
    curve = dict(partition_curve([i["x50"], i["x50"] + i["ep"]], i["x50"], i["ep"]))
    return {
        "flotation_recovery": r,
        "mass_yield": y,
        "element_recovery": rec,
        "logistic_scale": s,
        "partition_at_x50": curve[i["x50"]],
        "partition_at_x50_plus_ep": curve[i["x50"] + i["ep"]],
    }


# ---------------------------------------------------------------------------
# plant
# ---------------------------------------------------------------------------

def stream_balance_recycle(i: Mapping[str, Any]) -> dict[str, float]:
    el = i["element"]
    fs = Flowsheet(i["name"])
    for u in i["units"]:
        fs.add(UnitOp(name=u["name"], mass_yield=u["mass_yield"],
                      element_removal={el: u["element_removal"]}))
    for src, kind, tgt in i["links"]:
        fs.connect(src, kind, tgt)
    feed = Stream("feed", Q_(i["feed_kg_s"], "kg/s"), {el: i["feed_ppm"] * 1e-6})
    res = fs.solve(feed, damping=i["damping"], tol=i["tol"], max_iter=i["max_iter"])
    names = [u["name"] for u in i["units"]]
    out = {
        "converged": 1.0 if res.converged else 0.0,
        "overall_yield": res.overall_yield,
        "closure_error": res.closure_error,
        "element_closure": res.element_closure(el),
        "recycle_kg_s": res.rejects[names[1]].kg_s,
        "rougher_feed_kg_s": res.unit_feeds[names[0]].kg_s,
        "rougher_product_kg_s": res.products[names[0]].kg_s,
        "cleaner_product_kg_s": res.products[names[1]].kg_s,
        "rougher_reject_kg_s": res.rejects[names[0]].kg_s,
        "rougher_feed_ppm": res.unit_feeds[names[0]].ppm(el),
        "rougher_product_ppm": res.products[names[0]].ppm(el),
        "cleaner_product_ppm": res.products[names[1]].ppm(el),
        "rougher_reject_ppm": res.rejects[names[0]].ppm(el),
        "recycle_ppm": res.rejects[names[1]].ppm(el),
    }
    for u in i["units"]:
        op = fs.units[u["name"]]
        out[f"max_feasible_feed_fraction_{u['name']}"] = op.max_feasible_feed_fraction(el)
    return out


def capacity_oee(i: Mapping[str, Any]) -> dict[str, float]:
    oee = OEE(availability=i["availability"], performance=i["performance"],
              quality=i["quality"])
    units = [UnitCapacity(name=u["name"], nameplate_rate=Q_(u["rate_t_per_h"], "tonne/hour"),
                          planned_hours=i["planned_hours"], oee=oee,
                          tonnes_per_tonne_product=u["tonnes_per_tonne_product"])
             for u in i["units"]]
    line = assess_line(units)
    losses = oee.loss_breakdown
    out = {
        "oee": oee.value,
        "availability_loss": losses["availability"],
        "performance_loss": losses["performance"],
        "quality_loss": losses["quality"],
        "loss_sum": float(math.fsum(losses.values())),
        "one_minus_oee": 1.0 - oee.value,
        "line_rate_t_per_yr": _mag(line.line_rate, "tonne"),
        "bottleneck_margin": line.bottleneck_margin,
    }
    out["bottleneck_is_" + line.bottleneck] = 1.0
    for u in units:
        out[f"effective_capacity_{u.name}"] = _mag(u.effective_capacity, "tonne")
        out[f"product_capacity_{u.name}"] = _mag(u.product_capacity, "tonne")
        out[f"utilisation_{u.name}"] = line.utilisation[u.name]
        out[f"slack_{u.name}"] = _mag(line.slack[u.name], "tonne")
    return out


def yield_cascade_vector(i: Mapping[str, Any]) -> dict[str, float]:
    ys = i["stage_yields"]
    out = {"cascade_yield": cascade_yield(ys)}
    for k, f in enumerate(stage_throughput_factors(ys)):
        out[f"throughput_factor_{k}"] = f
    return out


def spc_vector(i: Mapping[str, Any]) -> dict[str, float]:
    cap = capability(i["assays"], usl=i["usl"], confidence=i["confidence"])
    return {
        "mean": cap.mean,
        "sigma": cap.sigma,
        "n_lots": float(cap.n_lots),
        "cpk": cap.cpk,
        "ci_low": cap.ci_low,
        "ci_high": cap.ci_high,
        "off_spec_normal": off_spec_fraction(cap.mean, cap.sigma, i["usl"]),
        "off_spec_lognormal": off_spec_fraction(cap.mean, cap.sigma, i["usl"],
                                                model="lognormal"),
        "required_mean_at_target_cpk": required_process_mean(
            i["usl"], cap.sigma, i["cpk_target"]),
    }


def queueing_vector(i: Mapping[str, Any]) -> dict[str, float]:
    return {
        "mm1_waiting_hours": mm1_waiting_time(i["arrival_rate"], i["service_rate"]),
        "allen_cunneen_waiting_hours": allen_cunneen_waiting_time(
            i["arrival_rate"], i["service_rate"], i["cv_arrival"], i["cv_service"]),
        "allen_cunneen_at_unit_cv": allen_cunneen_waiting_time(
            i["arrival_rate"], i["service_rate"], 1.0, 1.0),
    }


# ---------------------------------------------------------------------------
# economics
# ---------------------------------------------------------------------------

def _site(i: Mapping[str, Any]) -> Site:
    return Site(
        site_id=i["site_id"], name="Golden vector fixture site",
        country=i["country"], region=i["region"], currency=Currency(i["currency"]),
        power=PowerSupply(
            energy_price=_v(Q_(i["energy_price"], f"{i['currency']}/kWh"), Tag.SOURCED),
            rate_basis="state_average"),
        labour=LabourRates(fully_loaded_operator=_v(
            Q_(i["operator_rate"], f"{i['currency']}/hour"), Tag.SOURCED)),
        reagents=ReagentPrices(prices={
            k: _v(Q_(x, f"{i['currency']}/kg"), Tag.SOURCED)
            for k, x in i["reagent_prices"].items()}),
        construction_cost_index=i.get("construction_cost_index", 1.0),
    )


def cash_cost_vector(i: Mapping[str, Any]) -> dict[str, float]:
    site = _site(i["site"])
    cur = i["site"]["currency"]
    build = cash_cost(
        site=site,
        demands=[InputDemand(d["name"], Q_(d["per_tonne_feed"], d["unit"]),
                             basis=d.get("basis", "feed"))
                 for d in i["demands"]],
        cascade_yield=i["cascade_yield"],
        product_tonnes_per_year=i["product_tonnes_per_year"],
        annual_labour=_v(Q_(i["annual_labour"], f"{cur}/year"), Tag.SOURCED),
        annual_maintenance=_v(Q_(i["annual_maintenance"], f"{cur}/year"), Tag.SOURCED),
        credits=[(c["name"], _v(Q_(c["value"], f"{cur}/tonne")))
                 for c in i["credits"]],
        freight_waived=True,
        capex=_v(Q_(i["capex"], cur), Tag.SOURCED),
        fixed_charge_rate=i["fixed_charge_rate"],
    )
    lines = {ln.name: ln.magnitude for ln in build.lines}
    out = {
        "gross_cost": build.gross_cost.magnitude,
        "cash_cost": build.cash_cost.magnitude,
        "full_cost": build.full_cost.magnitude,
        "capital_recovery": build.full_cost.magnitude - build.cash_cost.magnitude,
        "breakdown_closes": 1.0 if build.breakdown_sums_to_cash_cost() else 0.0,
    }
    for k, x in lines.items():
        out[f"line_{k}"] = x
    return out


def capex_vector(i: Mapping[str, Any]) -> dict[str, float]:
    site = _site(i["site"])
    eq = [Equipment(name=e["name"],
                    purchased_cost=_v(Q_(e["purchased_cost"], i["site"]["currency"]),
                                      Tag.SOURCED),
                    installation_factor=e["installation_factor"])
          for e in i["equipment"]]
    est = estimate_capex(
        equipment=eq, site=site, indirect_factor=i["indirect_factor"],
        contingency_fraction=i["contingency_fraction"],
        base_index=i["base_index"], target_index=i["target_index"],
    )
    lo, hi = est.accuracy_band()
    cur = i["site"]["currency"]
    scaled = scale_cost(
        known_cost=Q_(i["scaling"]["known_cost"], cur),
        known_size=Q_(i["scaling"]["known_size"], i["scaling"]["size_unit"]),
        target_size=Q_(i["scaling"]["target_size"], i["scaling"]["size_unit"]),
        exponent=i["scaling"]["exponent"],
    )
    escalated = escalate_cost(
        cost=Q_(i["escalation"]["cost"], cur),
        base_index=i["base_index"], target_index=i["target_index"],
    )
    return {
        "index_ratio": est.index_ratio,
        "location_factor": est.location_factor,
        "total_purchased": _mag(est.total_purchased, cur),
        "total_installed": _mag(est.total_installed, cur),
        "indirect_cost": _mag(est.indirect_cost, cur),
        "contingency": _mag(est.contingency, cur),
        "total_project_cost": _mag(est.total_project_cost, cur),
        "accuracy_low": _mag(lo, cur),
        "accuracy_high": _mag(hi, cur),
        "scaled_cost": _mag(scaled, cur),
        "escalated_cost": _mag(escalated, cur),
        "reconciles": 1.0 if est.reconciles() else 0.0,
    }


def valuation_vector(i: Mapping[str, Any]) -> dict[str, float]:
    proj = Project(
        capex_schedule=i["capex_schedule"],
        construction_periods=i["construction_periods"],
        ramp_fractions=i["ramp_fractions"],
        nameplate_tonnes=i["nameplate_tonnes"],
        price=i["price"],
        cash_cost_per_tonne=i["cash_cost_per_tonne"],
        life_periods=i["life_periods"],
    )
    cf = proj.cash_flows()
    rate = i["discount_rate"]
    out: dict[str, float] = {
        "npv": cf.npv(rate),
        "irr": float(cf.irr()),
        "payback_undiscounted": float(payback_period(cf.net_cash_flow, 0.0)),
        "levelized_cost": levelized_cost(proj, rate),
        "breakeven_price": breakeven_price(proj, rate),
        "npv_at_breakeven": npv(rate, __import__("dataclasses").replace(
            proj, price=breakeven_price(proj, rate)).cash_flows().net_cash_flow),
        "peak_funding_requirement": cf.peak_funding_requirement(),
    }
    for k, x in enumerate(cf.net_cash_flow):
        out[f"net_cash_flow_{k}"] = float(x)
    for k, x in enumerate(cf.output):
        out[f"output_{k}"] = float(x)
    return out


# ---------------------------------------------------------------------------
# uncertainty, sensitivity, surrogate, decision value
# ---------------------------------------------------------------------------

def monte_carlo_percentiles(i: Mapping[str, Any]) -> dict[str, float]:
    inputs = [Uncertain(name=i["name"], low=i["low"], high=i["high"], kind=i["kind"])]
    mc = monte_carlo(lambda **kw: kw[i["name"]], inputs,
                     n_draws=i["n_draws"], seed=i["seed"])
    y = mc.outputs["y"]
    out = {
        "n_draws": float(mc.n_draws),
        "n_failed": float(mc.n_failed),
        "mean": float(np.mean(y)),
    }
    for p in i["percentiles"]:
        out[f"p{int(p)}"] = float(np.percentile(y, p))
        out[f"bootstrap_se_p{int(p)}"] = mc.percentile_standard_error(
            "y", p, n_boot=i["n_boot"], seed=i["seed"])
    return out


def sobol_additive(i: Mapping[str, Any]) -> dict[str, float]:
    coef = i["coefficients"]
    names = [f"x{k + 1}" for k in range(len(coef))]
    inputs = [Uncertain(name=nm, low=0.0, high=1.0, kind="uniform") for nm in names]

    def model(**kw: float) -> float:
        return float(sum(c * kw[nm] for c, nm in zip(coef, names)))

    r = sobol_analysis(model, inputs, n_base=i["n_base"], seed=i["seed"])
    diag = r.diagnostics()
    # diagnostics() returns a mapping of named flags, never an empty dict, so
    # emptiness is not the pass condition. The pass condition is that the two
    # sum checks hold, no index is materially negative, no first-order index
    # exceeds its total, and the run is marked converged. A first draft tested
    # "not diag", which was always false and so reported every Sobol run as
    # failing its own diagnostics.
    passed = (
        bool(diag["sum_first_le_one"])
        and bool(diag["sum_total_ge_one"])
        and not diag["materially_negative_first"]
        and not diag["materially_negative_total"]
        and not diag["first_exceeds_total"]
        and bool(diag["converged"])
    )
    out: dict[str, float] = {
        "n_evaluations": float(r.n_evaluations),
        "output_variance": r.output_variance,
        "additive_fraction": r.additive_fraction,
        "diagnostics_pass": 1.0 if passed else 0.0,
        "diagnostics_sum_first_order": float(diag["sum_first_order"]),
        "diagnostics_sum_total_order": float(diag["sum_total_order"]),
    }
    for nm in names:
        out[f"first_{nm}"] = r.first_order[nm]
        out[f"total_{nm}"] = r.total_order[nm]
        out[f"interaction_{nm}"] = r.interaction_share[nm]
    return out


def surrogate_lodo(i: Mapping[str, Any]) -> dict[str, float]:
    X = np.asarray(i["X"], dtype=float)
    y = np.asarray(i["y"], dtype=float)
    groups = np.asarray(i["groups"])
    ts = TrainingSet(X=X, y=y, groups=groups,
                     feature_names=i["feature_names"], target_name=i["target_name"])
    splits = leave_one_deposit_out_splits(groups)
    res = evaluate_surrogate(ts, kind=i["kind"], seed=i["seed"])
    out: dict[str, float] = {
        "n_folds": float(len(splits)),
        "target_iqr": res.target_iqr,
        "global_mean": float(np.mean(y)),
    }
    for f in res.folds:
        out[f"n_train_{f.deposit}"] = float(f.n_train)
        out[f"n_test_{f.deposit}"] = float(f.n_test)
        out[f"baseline_mean_rmse_{f.deposit}"] = f.baseline_mean_rmse
        out[f"rmse_{f.deposit}"] = f.rmse
        out[f"baseline_ridge_rmse_{f.deposit}"] = f.baseline_ridge_rmse
    return out


def evpi_vector(i: Mapping[str, Any]) -> dict[str, float]:
    lo, hi = i["theta_low"], i["theta_high"]
    actions = [Action(name=a["name"], cost=a["cost"]) for a in i["actions"]]

    def value(action: str, params: Mapping[str, float]) -> float:
        return 0.0 if action == i["null_action"] else float(params["theta"])

    priors = {"theta": lambda rng, n: rng.uniform(lo, hi, n)}
    problem = DecisionProblem(actions=actions, value=value, priors=priors)
    r = evpi(problem, "theta", n_outer=i["n_outer"], n_inner=i["n_inner"], seed=i["seed"])
    return {
        "evpi": r.evpi,
        "baseline_value": r.baseline_value,
        "resolved_value": r.resolved_value,
        "switch_fraction": r.switch_fraction,
    }


DISPATCH: dict[str, Callable[[Mapping[str, Any]], dict[str, float]]] = {
    "bond_energy": bond_energy,
    "bond_inversion": bond_inversion,
    "rosin_rammler": rosin_rammler,
    "lognormal_psd": lognormal_psd,
    "furnas_packing": furnas_packing,
    "krieger_dougherty": krieger_dougherty,
    "liberation_geometry": liberation_geometry,
    "liberation_leachable": liberation_leachable,
    "impurity_floor": impurity_floor,
    "purity_ceiling": purity_ceiling,
    "leach_characteristic_times": leach_characteristic_times,
    "leach_conversion": leach_conversion,
    "leach_tau_from_point": leach_tau_from_point,
    "chlorination_volatility": chlorination_volatility,
    "chlorination_gibbs": chlorination_gibbs,
    "thermal_enthalpy": thermal_enthalpy,
    "thermal_heat_capacity": thermal_heat_capacity,
    "separation_recovery": separation_recovery,
    "stream_balance_recycle": stream_balance_recycle,
    "capacity_oee": capacity_oee,
    "yield_cascade": yield_cascade_vector,
    "spc": spc_vector,
    "queueing": queueing_vector,
    "cash_cost": cash_cost_vector,
    "capex": capex_vector,
    "valuation": valuation_vector,
    "monte_carlo_percentiles": monte_carlo_percentiles,
    "sobol_additive": sobol_additive,
    "surrogate_lodo": surrogate_lodo,
    "evpi": evpi_vector,
}


def run(check: str, inputs: Mapping[str, Any]) -> dict[str, float]:
    """Dispatch one vector's inputs to its check function.

    An unknown check name raises rather than returning an empty mapping, so a
    typo in a vector file fails loudly instead of passing with nothing checked.
    """
    if check not in DISPATCH:
        raise KeyError(
            f"unknown check {check!r}; the registered checks are {sorted(DISPATCH)}"
        )
    out = DISPATCH[check](inputs)
    return {k: float(v) for k, v in out.items()}
