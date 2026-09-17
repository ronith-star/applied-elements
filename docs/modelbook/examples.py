"""Worked examples for the model book, executed so the printed numbers are real.

Every number that appears in a worked example in the book comes from running
this file. Nothing is transcribed by hand.

Two sources of examples, kept separate on purpose:

1. DOCTEST examples already present in ``src/ae`` docstrings. Those are run by
   :mod:`build` through the :mod:`doctest` machinery, and the book prints the
   output the interpreter produced, not the output the docstring claims. A
   mismatch is recorded as a discrepancy rather than hidden.

2. The examples in this file, for the modules that carry no doctest. Ten of the
   twenty-seven model modules fall in that class, mostly because their entry
   points need a constructed ``Feedstock``, ``Site`` or ``TrainingSet`` that
   does not fit a three-line doctest. Each example here is written against the
   module's own documented API and its output is captured verbatim.

An example that raises aborts the build. A worked example that cannot be run is
worse than no worked example, because the reader has no way to tell.
"""
from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass, field


@dataclass
class Example:
    """One executable worked example.

    Attributes
    ----------
    module
        Dotted module path this example documents, matching the chapter.
    title
        Caption shown above the listing.
    code
        Source executed with :func:`exec` in a fresh namespace. Its stdout is
        captured and typeset directly below the source.
    note
        Prose printed after the output, for the reading of the numbers.
    shown
        The portion of ``code`` printed in the book. Examples that need a
        constructed Feedstock or Site prepend a shared fixture preamble which
        is executed but not reprinted in every chapter, since it is listed once
        in the fixtures appendix.
    """

    module: str
    title: str
    code: str
    note: str = ""
    shown: str = field(default="", init=False)
    stdout: str = field(default="", init=False)
    error: str = field(default="", init=False)

    def run(self) -> None:
        """Execute the example, capturing stdout and any traceback."""
        buf = io.StringIO()
        ns: dict[str, object] = {}
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(self.code, f"<example:{self.module}>", "exec"), ns)
        except Exception:
            self.error = traceback.format_exc()
        self.stdout = buf.getvalue()


EXAMPLES: list[Example] = []

#: Shared fixture preamble, executed before any example that needs a
#: constructed ``Feedstock`` or ``Site``. Declared once here and listed once in
#: the book's fixtures appendix rather than reprinted in every chapter.
#:
#: Everything in it is SYNTHETIC. The Vikarabad deposit is uncharacterized in
#: the citable record, so every impurity number below is a scenario input
#: tagged ASSUMED. None of them is a property of that ore, and none may be
#: quoted as one.
PREAMBLE = r"""
import datetime as dt
import numpy as np
from ae.core.units import Q_
from ae.core.provenance import Value, Source, Tag, Tier, Distribution
from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.site import (Currency, LabourRates, PermittingRegime, PowerSupply,
                          ReagentPrices, Site)

_SRC = Source(citation="Fixture placeholder for a synthetic worked example",
              tier=Tier.T2, url="https://example.invalid/not-a-real-source",
              accessed=dt.date(2026, 9, 17))


def _v(q, tag=Tag.SOURCED, **kw):
    if tag in (Tag.SOURCED, Tag.MEASURED):
        return Value(quantity=q, tag=tag, source=_SRC, **kw)
    kw.setdefault("basis", "scenario input for a worked example, not a measurement")
    return Value(quantity=q, tag=tag, **kw)


def scenario_feedstock():
    imp = ImpurityProfile(
        total={e: _v(Q_(v, "ppm_mass"), Tag.ASSUMED) for e, v in
               [("Al", 30.0), ("Ti", 10.0), ("Li", 5.0), ("Fe", 3.0),
                ("Na", 8.0), ("K", 8.0), ("B", 1.0)]},
        lattice_fraction={e: _v(Q_(f, "dimensionless"), Tag.ASSUMED) for e, f in
                          [("Al", 0.6), ("Ti", 0.9), ("Li", 0.8), ("Fe", 0.1),
                           ("Na", 0.3), ("K", 0.3), ("B", 0.5)]},
        method="LA_ICP_MS")
    return Feedstock(sample_id="AE-Q-IN-SYNTH-001", ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="Synthetic worked-example vein", country="IN",
                     impurities=imp, characterized=True)


def us_site():
    return Site(site_id="US-NM-ABQ", name="New Mexico", country="US", region="NM",
                currency=Currency.USD,
                power=PowerSupply(energy_price=_v(Q_(0.0541, "USD/kWh")),
                                  rate_basis="state_average"),
                labour=LabourRates(fully_loaded_operator=_v(Q_(36.92, "USD/hour"))),
                reagents=ReagentPrices(prices={"HF": _v(Q_(1.80, "USD/kg")),
                                               "HCl": _v(Q_(0.22, "USD/kg")),
                                               "Ca(OH)2": _v(Q_(0.12, "USD/kg"))}))


def india_site():
    return Site(site_id="IN-TG-VKB", name="Telangana", country="IN", region="Telangana",
                currency=Currency.INR,
                power=PowerSupply(energy_price=_v(Q_(7.0, "INR/kWh"), Tag.ASSUMED),
                                  rate_basis="published_tariff"),
                labour=LabourRates(
                    fully_loaded_operator=_v(Q_(300.0, "INR/hour"), Tag.ASSUMED)),
                reagents=ReagentPrices(
                    prices={"HF": _v(Q_(180.0, "INR/kg"), Tag.ASSUMED),
                            "HCl": _v(Q_(12.0, "INR/kg"), Tag.ASSUMED),
                            "Ca(OH)2": _v(Q_(6.0, "INR/kg"), Tag.ASSUMED)},
                    locally_available={"HF": False, "HCl": True, "Ca(OH)2": True}),
                permitting=PermittingRegime(
                    jurisdiction="Synthetic fixture jurisdiction",
                    effluent_limits={"F": _v(Q_(2.0, "mg/L"), Tag.ASSUMED)}))
"""


def ex(module: str, title: str, code: str, note: str = "", preamble: bool = False) -> None:
    """Register one example. ``preamble=True`` prepends the shared fixtures."""
    e = Example(module, title, (PREAMBLE + "\n" + code) if preamble else code, note)
    e.shown = code.strip("\n")
    EXAMPLES.append(e)


# --- core -------------------------------------------------------------------

ex(
    "ae.core.units",
    "Mass ratio versus mole ratio for 30 ppm Al in SiO2",
    """
from ae.core.units import Q_, to_ppm_mass, ratio_basis

q = Q_(30.0, "ug/g")
print("basis of ug/g           :", ratio_basis(q))
print("to_ppm_mass(30 ug/g)    :", to_ppm_mass(q).magnitude, "ppm by mass")

# The same material on a mole basis, computed explicitly rather than by unit
# conversion, so the stoichiometry is visible.
M_SiO2, M_Al = 60.08, 26.98
print("M(SiO2)/M(Al)           :", round(M_SiO2 / M_Al, 4))
print("30 ppm mass as umol/mol :", round(30.0 * M_SiO2 / M_Al, 2))
print("per mole of ATOMS       :", round(30.0 * M_SiO2 / M_Al / 3.0, 2))

# A mole ratio offered where a mass ratio is required must be refused.
try:
    to_ppm_mass(Q_(30.0, "umol/mol"))
except ValueError as e:
    print("mole ratio rejected     :", str(e)[:56])
""",
    "The three readings of one reported 30 ppm span a factor of three. pint "
    "reduces all three to dimensionless, so the basis is checked by inspecting "
    "unit composition rather than inferred from dimensionality.",
)

ex(
    "ae.core.provenance",
    "A sourced value, a rejected assumption, and a missing value",
    """
import datetime as dt
import numpy as np
from ae.core.units import Q_
from ae.core.provenance import (Distribution, MISSING, MissingValueError, Source,
                                Tag, Tier, Value)

v = Value(quantity=Q_(30.0, "ppm_mass"), tag=Tag.SOURCED,
          source=Source(citation="Goetze and Moeckel (eds) 2012, Quartz: Deposits, "
                                 "Mineralogy and Analytics",
                        tier=Tier.T1, url="https://doi.org/10.1007/978-3-642-22161-3",
                        accessed=dt.date(2026, 9, 16)))
print("sourced value           :", v)

# An ASSUMED value with no stated basis must fail construction, not review.
try:
    Value(quantity=Q_(0.7, "dimensionless"), tag=Tag.ASSUMED)
except Exception as e:
    print("assumed, no basis       :", type(e).__name__)

a = Value(quantity=Q_(0.7, "dimensionless"), tag=Tag.ASSUMED,
          basis="placeholder cascade yield pending the assay campaign",
          dist=Distribution.relative_normal(0.20))
s = a.sample(20000, np.random.default_rng(0))
print("nominal                 :", a.magnitude)
print("relative_normal(0.20) sd:", round(float(np.std(s.magnitude)), 5),
      " expected 0.7 x 0.20 =", 0.7 * 0.20)

try:
    _ = MISSING + 1.0
except MissingValueError as e:
    print("MISSING in arithmetic   :", type(e).__name__)
""",
    "A 20 percent relative normal is multiplicative on the nominal, so its "
    "standard deviation is 0.7 times 0.20. An absolute distribution with the "
    "same parameters would mean something entirely different, which is why the "
    "flag is explicit rather than inferred from a zero mean.",
)

ex(
    "ae.core.registry",
    "Coverage report over a three-parameter registry",
    """
import datetime as dt
from ae.core.units import Q_
from ae.core.provenance import Distribution, Source, Tag, Tier, Value
from ae.core.registry import ParameterNotFound, Registry

r = Registry()
r.add("comminution.work_index.quartzite",
      Value(quantity=Q_(13.57, "kWh/tonne"), tag=Tag.SOURCED,
            source=Source(citation="Bond work index compilation, quartzite",
                          tier=Tier.T2, url="https://www.911metallurgist.com/blog/bond-work-index/",
                          accessed=dt.date(2026, 9, 16))))
r.add("plant.cascade_yield",
      Value(quantity=Q_(0.70, "dimensionless"), tag=Tag.ASSUMED,
            basis="placeholder pending the AE-Q characterization campaign",
            dist=Distribution.relative_normal(0.15)))
r.add("site.power.energy_price",
      Value(quantity=Q_(0.0541, "USD/kWh"), tag=Tag.SOURCED,
            source=Source(citation="EIA Electric Power Monthly Table 5.6.B", tier=Tier.T1,
                          url="https://www.eia.gov/electricity/monthly/",
                          accessed=dt.date(2026, 9, 16))))

c = r.coverage()
print("n_parameters            :", c["n_parameters"])
print("by_tag                  :", dict(sorted(c["by_tag"].items())))
print("by_tier                 :", dict(sorted(c["by_tier"].items())))
print("assumed_fraction        :", round(c["assumed_fraction"], 6))
print("assumed_keys            :", c["assumed_keys"])
print("point_estimate_keys     :", len(c["point_estimate_keys"]), "of", c["n_parameters"])
print("uncertain_keys          :", r.uncertain_keys())
print("magnitude, named unit   :", r.magnitude("comminution.work_index.quartzite", "kWh/tonne"))

try:
    r.get("comminution.work_index.quartzit")
except ParameterNotFound as e:
    print("typo, near miss offered :", "quartzite" in str(e))
""",
    "One parameter of three is ASSUMED, so a third of this registry rests on "
    "assumption. Two of three carry no distribution and are therefore invisible "
    "to any sensitivity result, which is the second number a reviewer should ask "
    "about.",
)

ex(
    "ae.core.feedstock",
    "Lattice-bound versus leachable impurities, and the ceiling they set",
    """
f = scenario_feedstock()
imp = f.impurities
els = ("Al", "Ti", "Li", "Fe", "Na", "K", "B")
print(f"{'el':3s} {'total':>8s} {'lattice f':>10s} {'lattice':>9s} {'leachable':>10s}")
for el in els:
    tot = imp.total_ppm(el)
    lat = imp.lattice_ppm(el)
    lf = imp.lattice_fraction[el].magnitude
    print(f"{el:3s} {tot:8.1f} {lf:10.2f} {lat:9.2f} {tot - lat:10.2f}")
print()
print("sum of all traces       :", round(imp.sum_ppm(els), 2), "ppm")
print("lattice-bound subtotal  :",
      round(sum(imp.lattice_ppm(e) for e in els), 2), "ppm")
print("acid-accessible         :",
      round(imp.sum_ppm(els) - sum(imp.lattice_ppm(e) for e in els), 2), "ppm")
print()
print("characterization tier   :", f.characterization_tier)
print("characterization gap    :", f.characterization_gap())
print("permits a grade claim   :", f.permits_output("product_grade_claim"))
print("permits a ceiling claim :", f.permits_output("purification_ceiling"))
""",
    "The lattice-bound subtotal is the part no acid route reaches. It is a "
    "property of crystallisation conditions, not of the flowsheet, and it sets "
    "the ceiling grade before any route is chosen.",
    preamble=True,
)

ex(
    "ae.core.site",
    "Two sites, one flowsheet, and the currency discipline between them",
    """
us, india = us_site(), india_site()
for s in (us, india):
    print(f"{s.name:12s} {s.currency:4s} power {s.power.energy_price.magnitude:9.4f} "
          f"{s.power.energy_price.units:12s} tag {s.power.energy_price.tag.value}")
print()
for s in (us, india):
    hf = s.reagents.prices["HF"]
    print(f"{s.name:12s} HF {hf.magnitude:8.2f} {hf.units:10s} "
          f"locally available: {s.reagents.locally_available.get('HF', 'unstated')}")
print()
print("India fluoride limit    :", india.permitting.effluent_limits["F"])
print("US permitting regime    :", us.permitting.jurisdiction if us.permitting else "not stated")
print()
# A cross-currency comparison requires an explicit dated rate, never unit magic.
from ae.core.units import DimensionalityError
try:
    _ = us.power.energy_price.quantity + india.power.energy_price.quantity
except DimensionalityError:
    print("USD + INR              : refused, no implicit exchange rate")
""",
    "Currency is a pint base dimension, so adding USD to INR raises rather than "
    "silently reducing. An exchange rate is an economic assumption with a date "
    "and a source, handled explicitly and never by unit conversion.",
    preamble=True,
)

# --- physics ----------------------------------------------------------------

ex(
    "ae.physics.chlorination",
    "The three-condition removal screen at 1200 degC, with and without a reductant",
    """
from ae.core.units import Q_
from ae.physics.chlorination import (CHLORIDES, hertz_knudsen_flux, removal_screen,
                                     requires_reductant, trouton_enthalpy, vapour_pressure)

T = Q_(1200.0, "degC")
print(f"{'el':3s} {'reductant':>9s} {'thermo ok':>10s} {'volatile':>9s} {'T/Tb':>7s}")
for el in ("Na", "K", "Fe", "Al", "Ti", "B"):
    for red in (False, True):
        r = removal_screen(el, T, reductant_present=red)
        print(f"{el:3s} {str(red):>9s} {str(not r.thermodynamically_blocked):>10s} "
              f"{str(r.volatile):>9s} {r.T_over_Tb:7.2f}")
print()
r = removal_screen("Al", T, reductant_present=False)
print("Al verdict without C    :", r.verdict[:96])
print()
p, capped = vapour_pressure(CHLORIDES["TiCl4"], Q_(100.0, "degC"))
print("TiCl4 p_sat at 100 degC :", round(float(p.to("Pa").magnitude), 2), "Pa =",
      round(float(p.to("Pa").magnitude) / 101325.0, 6), "atm; capped:", capped)
p2, capped2 = vapour_pressure(CHLORIDES["TiCl4"], T)
print("TiCl4 p_sat at 1200 degC:", round(float(p2.to("Pa").magnitude), 2), "Pa; capped:", capped2)
j = hertz_knudsen_flux(CHLORIDES["NaCl"], Q_(1465.0, "degC"))
print("NaCl flux at its Tb     :", round(float(j.to("mol/(m**2*s)").magnitude), 2),
      "mol m^-2 s^-1")
print("Trouton dHvap at 1738 K :",
      round(float(trouton_enthalpy(Q_(1738.15, "K")).to("kJ/mol").magnitude), 4), "kJ/mol")
print("requires_reductant Al/Na:", requires_reductant("Al"), requires_reductant("Na"))
""",
    "Ti, Al and B come back thermodynamically blocked without a reductant at "
    "every temperature, which is the Liu et al. (2026) finding enforced "
    "structurally. The 1200 degC TiCl4 pressure is returned capped at the "
    "system total pressure, because a Clausius-Clapeyron extrapolation that far "
    "above the boiling point is not a physical pressure.",
)

ex(
    "ae.physics.diffusion",
    "Diffusion length, and the activation-energy bracket that makes the Al case undecidable",
    """
import math
from ae.core.units import Q_
from ae.physics.diffusion import (LIU_2026_EA_RANGE_KJ, MOST_MOBILE_BOUND,
                                  critical_activation_energy, diffusion_length,
                                  fourier_number, fractional_extraction_sphere)

d = MOST_MOBILE_BOUND.at(Q_(80.0, "degC"))
L = diffusion_length(d, Q_(6.0, "hour"))
print("D at 80 degC, upper bound:", f"{float(d.to('m**2/s').magnitude):.6e}", "m2/s")
print("L = sqrt(D t) over 6 h   :", f"{float(L.to('m').magnitude):.6e}", "m =",
      round(float(L.to("nm").magnitude), 2), "nm")
print("R/L on a 100 um grain    :", round(1.0e-4 / float(L.to("m").magnitude), 2),
      "=", round(math.log10(1.0e-4 / float(L.to("m").magnitude)), 4), "orders of magnitude")
print()
hi = critical_activation_energy(Q_(1e-4, "m**2/s"), Q_(100.0, "um"),
                                Q_(1200.0, "degC"), Q_(6.0, "hour"))
lo = critical_activation_energy(Q_(1e-10, "m**2/s"), Q_(100.0, "um"),
                                Q_(1200.0, "degC"), Q_(6.0, "hour"))
band = LIU_2026_EA_RANGE_KJ.quantity.to("kJ/mol").magnitude
print("Ea_crit at D0 = 1e-4     :", round(float(hi.magnitude), 4), "kJ/mol")
print("Ea_crit at D0 = 1e-10    :", round(float(lo.magnitude), 4), "kJ/mol")
print("Liu et al. 2026 Ea range :", list(band), "kJ/mol")
print("overlap                  :", round(max(float(lo.magnitude), float(band.min())), 4),
      "to", round(min(float(hi.magnitude), float(band.max())), 4), "kJ/mol")
print("either bracket contains? :",
      bool(float(lo.magnitude) <= band.min() and float(hi.magnitude) >= band.max()))
print()
fo = fourier_number(d, Q_(6.0, "hour"), Q_(100.0, "um"))
print("Fourier number at 80 degC:", f"{fo:.6e}")
print("fractional extraction    :", f"{fractional_extraction_sphere(fo):.6e}")
""",
    "The two prefactor brackets straddle the reported barrier range without "
    "either containing the other, so the overlap region is where a true Al "
    "barrier permits depletion at one prefactor and forbids it at another. That "
    "is undecidable on present data, not merely uncertain.",
)

ex(
    "ae.physics.leaching",
    "Shrinking-core time constants and the geometric factor of three",
    """
from ae.core.units import Q_
from ae.core.provenance import Tag
from ae.physics.leaching import (Arrhenius, LeachSystem, Regime, conversion,
                                 removal_fraction_from_assay, tau_film,
                                 tau_product_layer, tau_surface_reaction)

k = Arrhenius(prefactor=_v(Q_(1.0e-4, "m/s"), Tag.ASSUMED),
              activation_energy=_v(Q_(15.0, "kJ/mol"), Tag.ASSUMED))
system = LeachSystem(
    feedstock=scenario_feedstock(), element="Fe", reagent="HCl",
    particle_radius=_v(Q_(100.0, "um"), Tag.ASSUMED),
    reagent_concentration=_v(Q_(1000.0, "mol/m**3"), Tag.ASSUMED),
    solid_molar_density=_v(Q_(30000.0, "mol/m**3"), Tag.ASSUMED),
    stoich_b=1.0,
    temperature=_v(Q_(80.0, "degC"), Tag.ASSUMED),
    film_coefficient=k,
    surface_rate_constant=k,
    product_layer_diffusivity=Arrhenius(
        prefactor=_v(Q_(1.0e-9, "m**2/s"), Tag.ASSUMED),
        activation_energy=_v(Q_(25.0, "kJ/mol"), Tag.ASSUMED)))

tf, ts, tp = tau_film(system), tau_surface_reaction(system), tau_product_layer(system)
print("tau_film                :", round(float(tf.to("s").magnitude), 4), "s =",
      round(float(tf.to("min").magnitude), 4), "min")
print("tau_surface             :", round(float(ts.to("s").magnitude), 4), "s")
print("tau_surface / tau_film  :", round(float((ts / tf).to("dimensionless").magnitude), 10))
print("tau_product_layer       :", round(float(tp.to("s").magnitude), 4), "s")
print()
# Evaluated at a FRACTION of tau: at t = tau every regime reaches X = 1 by
# definition of tau, so the regimes are only distinguishable before that.
for frac in (0.25, 0.5, 0.75):
    row = [f"{conversion(reg, frac * tf, tf):.6f}"
           for reg in (Regime.FILM, Regime.SURFACE_REACTION, Regime.PRODUCT_LAYER)]
    print(f"X at t = {frac:.2f} tau: film {row[0]}  surface {row[1]}  product layer {row[2]}")
print()
print("Xia et al. 2024, 128.86 to 24.23 ug/g:",
      round(removal_fraction_from_assay(128.86, 24.23), 7))
""",
    "With equal film and surface rate constants the surface time constant is "
    "exactly three times the film one. That factor is the ratio of particle "
    "volume to surface area in the shrinking-core derivation, and reproducing it "
    "is a check on the implementation rather than on the physics.",
    preamble=True,
)

ex(
    "ae.physics.reagents",
    "HF demand set by silica attack, not by the assay",
    """
from ae.core.units import Q_
from ae.physics.reagents import (Acid, Base, M_SIO2, hf_silica_demand,
                                 impurity_acid_demand, impurity_moles,
                                 neutralization_demand)

f = scenario_feedstock()
ore = Q_(1000.0, "kg")
n_hf, m_diss = hf_silica_demand(ore, 0.001)
n_hf4, _ = hf_silica_demand(ore, 0.001, route="SiF4_gas")
print("M_SiO2                  :", M_SIO2, "kg/mol")
print("silica dissolved at 0.1%:", float(m_diss.to("kg").magnitude), "kg per tonne")
print("n_SiO2                  :", round(1.0 / M_SIO2, 6), "mol")
print("HF, H2SiF6 route (6:1)  :", round(float(n_hf.to("mol").magnitude), 6), "mol/t")
print("HF, SiF4 route   (4:1)  :", round(float(n_hf4.to("mol").magnitude), 6), "mol/t")
print("ratio of routes         :", round(float((n_hf4 / n_hf).magnitude), 8))
print()
for el, q in sorted(impurity_moles(f, ore).items()):
    print(f"leachable {el:2s}            : {float(q.to('mol').magnitude):10.6f} mol/t")
tot, _per = impurity_acid_demand(f, ore, Acid.HF)
print("all impurities, HF      :", round(float(tot.to("mol").magnitude), 6), "mol/t")
print("silica / impurity HF    :", round(float((n_hf / tot).magnitude), 2), "times")
print()
n = neutralization_demand({Acid.HF: Q_(100.0, "mol")}, Base.LIME)
print("100 mol HF, lime        :", n["base_moles_mol"], "mol =",
      round(n["base_mass_kg"], 4), "kg")
print("CaF2 dry sludge         :", n["CaF2_moles_mol"], "mol =",
      round(n["CaF2_sludge_dry_kg"], 4), "kg")
print("fluoride removed by base:", n["fluoride_removed_by_base"])
""",
    "Dissolving one part in a thousand of the silica consumes roughly two orders "
    "of magnitude more HF than every impurity cation in the feed combined. "
    "Reagent cost is therefore set by silica selectivity, and an HF budget built "
    "from the assay alone is wrong by that factor.",
    preamble=True,
)

# --- plant ------------------------------------------------------------------

ex(
    "ae.plant.streams",
    "A two-unit flowsheet whose mass and element balances must close",
    """
from ae.core.units import Q_
from ae.plant.streams import Flowsheet, Stream, UnitOp

feed = Stream(name="feed", mass_flow=Q_(10.0, "tonne/hour"),
              composition={"Al": 1164e-6, "Fe": 140e-6, "Ti": 50e-6})
print("feed Al                 :", feed.ppm("Al"), "ppm")

flot = UnitOp(name="flotation", mass_yield=0.92,
              element_removal={"Al": 0.55, "Fe": 0.70, "Ti": 0.30})
leach = UnitOp(name="leach", mass_yield=0.97,
               element_removal={"Al": 0.40, "Fe": 0.85}, kind="chemical")
flot.check_feasible(feed.composition)
print("max feasible Al feed    :", round(flot.max_feasible_feed_fraction("Al"), 8),
      "(feed is", feed.composition["Al"], ")")
print()
fs_ = Flowsheet("two stage").add(flot).add(leach)
fs_.connect("flotation", "product", "leach")
res = fs_.solve(feed, first_unit="flotation")
print("overall mass yield      :", round(res.overall_yield, 8))
print("closure error           :", f"{res.closure_error:.3e}")
print("converged in            :", res.iterations, "iterations by", res.method)
print()
for el in sorted(feed.composition):
    print(f"element closure {el:3s}     : {res.element_closure(el):.3e}")
print()
for name, s in sorted(res.products.items()):
    print(f"product {name:12s}    : {float(s.mass_flow.to('tonne/hour').magnitude):8.5f} t/h",
          {k: round(v * 1e6, 3) for k, v in sorted(s.composition.items())}, "ppm")
""",
    "Closure is enforced to floating-point residual on total mass and on every "
    "tracked element, so a reported recovery cannot be the result of mass "
    "appearing or disappearing between units. An infeasible mass-yield and "
    "removal pair is rejected before the solve, naming the parameter rather than "
    "the symptom.",
)

ex(
    "ae.plant.capacity",
    "Nameplate to shippable tonnes through OEE, and where the bottleneck sits",
    """
from ae.core.units import Q_
from ae.plant.capacity import OEE, UnitCapacity, assess_line

oee = OEE(availability=0.85, performance=0.95, quality=0.99)
print("OEE components          :", oee)
print("OEE product             :", round(oee.availability * oee.performance * oee.quality, 8))
print()
units = [
    UnitCapacity(name="crush", nameplate_rate=Q_(12.0, "tonne/hour"),
                 planned_hours=8760.0, oee=oee, tonnes_per_tonne_product=1.45),
    UnitCapacity(name="flotation", nameplate_rate=Q_(6.0, "tonne/hour"),
                 planned_hours=8000.0, oee=oee, tonnes_per_tonne_product=1.30),
    UnitCapacity(name="leach", nameplate_rate=Q_(5.0, "tonne/hour"),
                 planned_hours=7500.0, oee=oee, tonnes_per_tonne_product=1.05),
]
line = assess_line(units)
print("bottleneck              :", line.bottleneck)
print("line rate               :", round(float(line.line_rate.to("tonne").magnitude), 3), "t/yr")
print()
for name in sorted(line.product_capacity):
    print(f"{name:10s} capacity {float(line.product_capacity[name].to('tonne').magnitude):10.2f} t/yr"
          f"  utilisation {line.utilisation[name]:6.4f}"
          f"  slack {float(line.slack[name].to('tonne').magnitude):9.2f} t/yr")
""",
    "Capacity is stated per tonne of PRODUCT, so each unit's nameplate is "
    "divided by the tonnes of its own throughput needed per product tonne. The "
    "bottleneck is the unit with the least product capacity, not the smallest "
    "nameplate.",
)

ex(
    "ae.plant.yield_cascade",
    "The multiplicative cascade, and the throughput each stage must carry",
    """
from ae.plant.yield_cascade import (capability, cascade_yield, joint_off_spec_fraction,
                                    off_spec_fraction, required_process_mean,
                                    stage_throughput_factors)

stages = [0.95, 0.97, 0.88, 0.93, 0.90]
y = cascade_yield(stages)
print("stage yields            :", stages)
print("overall cascade yield   :", round(y, 8))
print("product per 1000 t feed :", round(1000.0 * y, 4), "t")
print()
print("throughput factors, relative to product out:")
for s, fx in zip(stages, stage_throughput_factors(stages)):
    print(f"  stage yield {s:.2f} must process {fx:.6f} t per t of its own output")
print()
# Specification conformance on a single element against an upper spec limit.
print("off-spec at mean 22, sigma 4, USL 30:")
for model in ("normal", "lognormal"):
    print(f"  {model:10s}: {off_spec_fraction(22.0, 4.0, 30.0, model=model):.8f}")
print("required mean for Cpk 1.33, sigma 4, USL 30:",
      round(required_process_mean(30.0, 4.0, 1.33), 6))
print()
per_el = {"Al": 0.02, "Ti": 0.03, "Fe": 0.01}
print("joint off-spec across three elements:", 
      {k: round(v, 8) for k, v in joint_off_spec_fraction(per_el).items()})
""",
    "A cascade multiplies, so the upstream stages must process more than the "
    "product tonnage by the reciprocal of everything downstream of them. The "
    "joint off-spec calculation is reported under both independence and perfect "
    "correlation because the truth lies between and the assay campaign has not "
    "measured which.",
)

ex(
    "ae.plant.scheduling",
    "M/M/1 closed form against the discrete-event simulation",
    """
from ae.plant.scheduling import (PlantSchedule, Station, allen_cunneen_waiting_time,
                                 mm1_waiting_time)
import numpy as np

lam, mu = 0.8, 1.0
wq = mm1_waiting_time(lam, mu)
print("rho                     :", lam / mu)
print("M/M/1 Wq, closed form   :", round(wq, 8))
print("Allen-Cunneen at cv=1   :", round(allen_cunneen_waiting_time(lam, mu, 1.0, 1.0), 8))
print("L = lambda W (Little)   :", round(lam * (wq + 1.0 / mu), 8))
print()
# Exponential service is cv = 1, which is the M/M/1 assumption.
res = PlantSchedule([Station(name="leach", service_hours=1.0 / mu, capacity=1, cv_service=1.0)]
                    ).run(arrival_rate=lam, horizon_hours=200000.0, seed=3, cv_arrival=1.0,
                          warmup_hours=2000.0)
print("lots started / shipped  :", res.lots_started, "/", res.lots_shipped)
print("station busy fraction   :", {k: round(v, 6) for k, v in res.station_busy_fraction.items()})
qt = float(np.mean(res.queue_times["leach"]))
print("simulated mean queue t  :", round(qt, 6))
print("relative error vs M/M/1 :", f"{abs(qt - wq) / wq:.4%}")
print()
lc = res.littles_law_check()
print("Little's law check      :", {k: round(v, 8) for k, v in lc.items()})
print("cycle time p50 / p95    :", round(res.cycle_time_percentile(50), 6), "/",
      round(res.cycle_time_percentile(95), 6), "h")
""",
    "The simulation is checked against the M/M/1 waiting time and against "
    "Little's law. Both are analytic identities with no literature datapoint "
    "behind them, and the validation record labels them analytic rather than "
    "literature for that reason.",
)

# --- econ -------------------------------------------------------------------

ex(
    "ae.econ.unit_economics",
    "Cash cost per tonne of product at a 0.70 cascade yield",
    """
from ae.core.units import Q_
from ae.core.provenance import Tag, Value
from ae.econ.unit_economics import InputDemand, cash_cost

b = cash_cost(
    site=us_site(),
    demands=[InputDemand("electricity", Q_(250.0, "kWh/tonne")),
             InputDemand("HF", Q_(12.0, "kg/tonne"))],
    cascade_yield=0.70,
    product_tonnes_per_year=5000.0,
    annual_labour=Value(quantity=Q_(3_000_000.0, "USD/year"), tag=Tag.ASSUMED,
                        basis="worked example input"),
    credits=[("silica fines", Value(quantity=Q_(20.0, "USD/tonne"), tag=Tag.ASSUMED,
                                    basis="worked example input"))],
    freight=Value(quantity=Q_(95.0, "USD/tonne"), tag=Tag.ASSUMED,
                  basis="worked example input"))

print("per tonne of FEED, then divided by the 0.70 yield:")
print("  electricity 250 kWh/t x 0.0541 USD/kWh =", round(250.0 * 0.0541, 4), "USD/t feed")
print("  HF          12 kg/t   x 1.80   USD/kg  =", round(12.0 * 1.80, 4), "USD/t feed")
print()
for ln in b.lines:
    print(f"{ln.name:16s} {ln.magnitude:10.4f} USD/t product")
print(f"{'gross':16s} {b.gross_cost.magnitude:10.4f} USD/t")
print(f"{'cash cost':16s} {b.cash_cost.magnitude:10.4f} USD/t")
print()
print("breakdown closes        :", b.breakdown_sums_to_cash_cost())
print("yield penalty factor    :", round(1.0 / 0.70, 6))
""",
    "Demands are quoted per tonne of feed and the cost is reported per tonne of "
    "product, so each variable line is divided by the cascade yield. Omitting "
    "that division understates variable cost by the reciprocal of the yield, "
    "here a factor of 1.428571.",
    preamble=True,
)

ex(
    "ae.econ.capex",
    "Six-tenths scaling, and the accuracy class the estimate is entitled to",
    """
from ae.core.units import Q_
from ae.core.provenance import Tag
from ae.econ.capex import Equipment, escalate_cost, estimate_capex, scale_cost

base, known, target = Q_(50.0e6, "USD"), Q_(5000.0, "tonne/year"), Q_(10000.0, "tonne/year")
for n in (0.5, 0.6, 0.7, 0.9):
    s = scale_cost(base, known, target, n)
    print(f"exponent {n:.1f}: 2x capacity costs "
          f"{float(s.to('USD').magnitude) / 1e6:9.4f} MUSD")
print()
print("cost per annual tonne at n = 0.6:")
for cap in (1.0, 2.0, 5.0, 10.0):
    s = float(scale_cost(base, known, known * cap, 0.6).to("USD").magnitude)
    print(f"  x{cap:5.1f} capacity: {s / (cap * 5000.0):10.2f} USD per annual tonne")
print()
print("escalated 2019 to 2026 on a CEPCI-style ratio:",
      round(float(escalate_cost(Q_(10.0e6, "USD"), 607.5, 800.0).to("USD").magnitude) / 1e6, 4),
      "MUSD")
print()
est = estimate_capex(
    equipment=[Equipment(name="jaw crusher",
                         purchased_cost=_v(Q_(450_000.0, "USD"), Tag.ASSUMED),
                         installation_factor=1.4),
               Equipment(name="leach train",
                         purchased_cost=_v(Q_(2_200_000.0, "USD"), Tag.ASSUMED),
                         installation_factor=2.1),
               Equipment(name="rotary calciner",
                         purchased_cost=_v(Q_(1_800_000.0, "USD"), Tag.ASSUMED),
                         installation_factor=1.9)],
    site=us_site(), indirect_factor=0.35, contingency_fraction=0.30,
    capacity=Q_(5000.0, "tonne/year"))
print("purchased               :", round(float(est.total_purchased.to("USD").magnitude) / 1e6, 4), "MUSD")
print("installed               :", round(float(est.total_installed.to("USD").magnitude) / 1e6, 4), "MUSD")
print("indirects               :", round(float(est.indirect_cost.to("USD").magnitude) / 1e6, 4), "MUSD")
print("contingency             :", round(float(est.contingency.to("USD").magnitude) / 1e6, 4), "MUSD")
print("total project cost      :", round(float(est.total_project_cost.to("USD").magnitude) / 1e6, 4), "MUSD")
lo, hi = est.accuracy_band()
print("estimate class          :", est.estimate_class, "accuracy band",
      f"{float(lo.magnitude):+.0%} to {float(hi.magnitude):+.0%}")
print("accuracy note           :", est.accuracy_note()[:92])
print("reconciles              :", est.reconciles())
""",
    "The exponent is the whole content of a scaled estimate: at ten times "
    "capacity, n = 0.6 gives 40 percent of the linear number. The estimate class "
    "is reported with the total, because a factored estimate carries an accuracy "
    "band wide enough to change the investment decision.",
    preamble=True,
)

ex(
    "ae.econ.valuation",
    "NPV, IRR and the discount-rate sensitivity of a two-year build",
    """
import numpy as np
from ae.econ.valuation import irr, modified_irr, npv, payback_period

flows = [-60.0, -40.0, 12.0, 18.0, 22.0, 25.0, 25.0, 25.0, 25.0, 25.0]
print("flows, MUSD             :", flows)
print("undiscounted sum        :", round(float(np.sum(flows)), 4), "MUSD")
print()
for r in (0.08, 0.10, 0.12, 0.15, 0.20):
    print(f"discount {r:.0%}: NPV {npv(r, flows):9.4f} MUSD")
print()
print("IRR                     :", round(irr(flows), 8))
print("MIRR (fin 8%, reinv 6%) :", round(modified_irr(flows, 0.08, 0.06), 8))
print("payback at 0%           :", payback_period(flows), "periods")
print("payback at 10%          :", payback_period(flows, rate=0.10), "periods")
print()
print("NPV sensitivity, 8% to 20%:",
      round(npv(0.08, flows) - npv(0.20, flows), 4), "MUSD swing")
""",
    "Two periods of outflow before first revenue make the NPV strongly "
    "discount-rate dependent. The swing across a plausible rate range exceeds "
    "the base-case NPV itself, which is the arithmetic reason a pre-assay "
    "project cannot be presented at a point discount rate.",
)

ex(
    "ae.econ.uncertainty",
    "Sobol indices against a closed form, and the convergence of the estimator",
    """
import numpy as np
from ae.econ.uncertainty import (Uncertain, convergence_check, monte_carlo,
                                 sobol_analysis, spearman_screening, tornado)

def f(a, b, c):
    return 2.0 * a + 3.0 * b + 4.0 * c

ins = [Uncertain(n, 0.0, 1.0) for n in ("a", "b", "c")]
exact = {"a": 4 / 29, "b": 9 / 29, "c": 16 / 29}
r = sobol_analysis(f, ins, n_base=8192, seed=7)
print("closed form: c_i^2 / sum(c_i^2), with 4 + 9 + 16 = 29")
print(f"{'param':6s} {'S1':>10s} {'ST':>10s} {'exact':>10s} {'abs err':>10s}")
for k in ("a", "b", "c"):
    print(f"{k:6s} {r.first_order[k]:10.6f} {r.total_order[k]:10.6f} "
          f"{exact[k]:10.6f} {abs(r.first_order[k] - exact[k]):10.6f}")
print("model evaluations       :", r.n_evaluations)
print("S1 sums to              :", round(sum(r.first_order.values()), 6), "(additive model)")
print()
mc = monte_carlo(f, ins, n_draws=40000, seed=1)
y = mc.outputs["y"]
print("MC mean                 :", round(float(np.mean(y)), 6),
      " closed form:", (2 + 3 + 4) / 2)
print("MC sd                   :", round(float(np.std(y)), 6),
      " closed form:", round(float(np.sqrt(29 / 12)), 6))
print("failed draws            :", mc.n_failed)
print()
print("Spearman screening      :", {k: round(v, 4) for k, v in
                                    spearman_screening(mc, "y").items()})
print()
t = tornado(f, ins, nominal={"a": 0.5, "b": 0.5, "c": 0.5})
for k in ("a", "b", "c"):
    print(f"tornado {k}: low {t[k]['low']:7.4f} high {t[k]['high']:7.4f} "
          f"swing {t[k]['swing']:7.4f}")
""",
    "For a linear model with independent uniform inputs the first-order indices "
    "are the squared coefficients normalised by their sum, and first order "
    "equals total order because there is no interaction. Recovering both, and "
    "the closed-form mean and standard deviation, is the analytic benchmark for "
    "the estimator.",
)

# --- ml and agent -----------------------------------------------------------

ex(
    "ae.ml.surrogate",
    "Leave-one-deposit-out error, the optimism of random k-fold, and an OOD refusal",
    """
import numpy as np
from ae.ml.surrogate import (Surrogate, TrainingSet, evaluate_surrogate,
                             random_kfold_optimism)

# Five synthetic deposits. Each carries its own offset, which is what makes
# within-deposit samples near-replicates and random k-fold optimistic.
rng = np.random.default_rng(0)
X, y, g = [], [], []
for i, dep in enumerate(["D1", "D2", "D3", "D4", "D5"]):
    x = rng.normal(loc=i * 0.5, scale=1.0, size=(40, 3))
    offset = rng.normal(0.0, 2.0)
    X.append(x)
    y.append(2.0 * x[:, 0] - 1.0 * x[:, 1] + 0.5 * x[:, 2] + offset + rng.normal(0, 0.3, 40))
    g += [dep] * 40
data = TrainingSet(X=np.vstack(X), y=np.concatenate(y), groups=np.array(g),
                   feature_names=["Al_ppm", "Ti_ppm", "Fe_ppm"], target_name="recovery")
print("deposits / samples      :", data.n_deposits, "/", data.n_samples)
print("deposit sizes           :", data.deposit_sizes())
print()
res = evaluate_surrogate(data, kind="gbm", seed=0)
print(f"{'fold':5s} {'rmse':>8s} {'mean base':>10s} {'ridge base':>11s} "
      f"{'>mean':>6s} {'>ridge':>7s}")
for fo in res.folds:
    print(f"{fo.deposit:5s} {fo.rmse:8.4f} {fo.baseline_mean_rmse:10.4f} "
          f"{fo.baseline_ridge_rmse:11.4f} {str(fo.beats_mean):>6s} "
          f"{str(fo.beats_ridge):>7s}")
print()
print("pooled LODO RMSE        :", round(res.rmse, 6))
print("mean absolute error     :", round(res.mae, 6))
print("normalised RMSE (/IQR)  :", round(res.normalised_rmse, 6))
print("skill over mean         :", round(res.skill_over_mean, 6))
print("folds beating mean      :", res.folds_beating_mean, "of", len(res.folds))
print("is_useful(min_skill 0.1):", res.is_useful())
print()
o = random_kfold_optimism(data, kind="gbm", seed=0)
print("random k-fold RMSE      :", round(o["random_kfold_rmse"], 6))
print("leave-one-deposit-out   :", round(o["leave_one_deposit_out_rmse"], 6))
print("optimism ratio          :", round(o["optimism_ratio"], 6))
print("optimism, absolute      :", round(o["optimism_absolute"], 6))
print()
s = Surrogate(kind="gbm", seed=0).fit(data)
far = [50.0, 50.0, 50.0]
rep = s.ood_report(far)
print("OOD on a far query      :", rep.is_ood, "| method:", rep.method)
print("reason                  :", rep.reason()[:88])
try:
    s.predict(far)
except Exception as e:
    print("predict refuses OOD     :", type(e).__name__)
val, rep2 = s.predict_with_report(far)
print("predict_with_report     :", round(val, 6), "| flagged:", rep2.is_ood)
print()
print("permutation importance on held-out deposits:",
      {k: round(v, 5) for k, v in s.permutation_importance(data, n_repeats=5, seed=0).items()})
""",
    "The optimism ratio is the number that matters. Random k-fold reports a "
    "smaller error than leave-one-deposit-out on identical data because it puts "
    "near-replicates on both sides of the split. Only the grouped figure is an "
    "estimate of performance on new geology.",
)

ex(
    "ae.agent.decisions",
    "EVPI ranking, EVPI per unit cost, and the inner-sample bias made visible",
    """
from ae.agent.decisions import (Action, DecisionProblem, evpi, evpi_convergence,
                                measurement_priority, rank_measurements)

# Two real actions. Value depends on a lattice Al ceiling, which is
# decision-relevant, and on a freight rate, which carries large variance but
# does not change which action is better.
def value(action, p):
    if action == "build leach circuit":
        return 400.0 - 3.0 * p["lattice_al_ppm"] - 0.5 * p["freight_usd_t"]
    return 120.0 - 0.5 * p["freight_usd_t"]

problem = DecisionProblem(
    actions=[Action("build leach circuit"), Action("sell lump quartz")],
    value=value,
    priors={"lattice_al_ppm": lambda rng, n: rng.uniform(20.0, 140.0, n),
            "freight_usd_t": lambda rng, n: rng.uniform(40.0, 200.0, n)})

print("E[V | a] under the prior:",
      {k: round(v, 4) for k, v in problem.expected_values(n_draws=8192, seed=0).items()})
best, v = problem.best_action_now(n_draws=8192, seed=0)
print("prior-best action       :", best, "at", round(v, 4))
print()
for iv in rank_measurements(problem, n_outer=256, n_inner=256, seed=0):
    print(f"{iv.parameter:16s} EVPI {iv.evpi:9.4f}  as fraction "
          f"{iv.evpi_fraction:8.5f}  switch fraction {iv.switch_fraction:.4f}")
print()
costs = {"lattice_al_ppm": 2500.0, "freight_usd_t": 50.0}
for row in measurement_priority(problem, costs, n_outer=256, n_inner=256, seed=0):
    print(f"{row['parameter']:16s} cost {row['cost']:8.1f}  "
          f"EVPI/cost {row['evpi_per_cost']:.6e}")
print()
print("inner-sample convergence, upward bias falling as n_inner grows:")
for n_inner, e in evpi_convergence(problem, "lattice_al_ppm",
                                   inner_sizes=(16, 64, 256), n_outer=128, seed=0):
    print(f"  n_inner {n_inner:4d}: EVPI {e:9.4f}")
""",
    "The freight rate carries large output variance and near-zero EVPI, because "
    "the decision is the same at every value it takes. A characterization budget "
    "ranked on variance rather than on decision relevance would buy the wrong "
    "measurement first.",
)
