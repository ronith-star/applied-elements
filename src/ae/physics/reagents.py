r"""Reagent stoichiometry, mass balance, neutralization demand and effluent load.

Scope: the consumption side of a hot acid leach on a silica feedstock. Given an
impurity load and a reagent recipe, this module computes moles of acid consumed
by each impurity phase and by the silica matrix, the lime or caustic required to
neutralize the spent liquor, the fluoride and metal load in the effluent
compared against the site's discharge limits, and the resulting neutralization
sludge. It takes SITE for reagent prices and effluent limits, and FEEDSTOCK for
the impurity inventory, so a change of jurisdiction or ore is a change of
argument rather than a change of code.

Why this module exists separately from :mod:`ae.physics.leaching`: the kinetic
model says how fast an impurity dissolves, but reagent cost and effluent
liability are driven by TOTAL consumption, and for HF on quartz the dominant
consumer is usually the silica matrix rather than the impurities. A flowsheet
that budgets HF against the impurity load alone understates consumption by
orders of magnitude, and that error is the main thing this module exists to
prevent.

EQUATIONS
=========

Symbols
-------
=========================  =======================  =========================
Symbol                     Unit                     Valid range
=========================  =======================  =========================
:math:`m_{ore}`            kg                       > 0
:math:`C_i`                kg kg^-1 (as ppm mass)   >= 0
:math:`M_i`                kg mol^-1                > 0
:math:`n_i`                mol                      >= 0
:math:`\nu_{i,r}`          dimensionless            > 0 (mol acid per mol i)
:math:`f_{diss}`           dimensionless            [0, 1] silica dissolved
:math:`z_i`                dimensionless            ion charge, any sign
:math:`n_{H^+}`            mol                      >= 0
:math:`\eta_{excess}`      dimensionless            >= 1 (reagent excess)
=========================  =======================  =========================

Impurity dissolution stoichiometry
----------------------------------
Moles of impurity element ``i`` in a charge of ore:

.. math:: n_i = \frac{m_{ore}\,C_i}{M_i}

with :math:`C_i` the mass fraction (ppm by mass divided by 1e6). Acid demand for
that impurity is :math:`\nu_{i,r} n_i`, where :math:`\nu_{i,r}` comes from the
balanced dissolution reaction. The reactions used, each balanced for mass and
charge (:func:`check_reaction_balance` verifies both):

.. math:: \mathrm{Fe_2O_3} + 6\,\mathrm{HCl} \rightarrow
          2\,\mathrm{FeCl_3} + 3\,\mathrm{H_2O}
          \quad (\nu = 3\ \mathrm{mol\ HCl\ per\ mol\ Fe})

.. math:: \mathrm{Al_2O_3} + 6\,\mathrm{HCl} \rightarrow
          2\,\mathrm{AlCl_3} + 3\,\mathrm{H_2O}
          \quad (\nu = 3\ \mathrm{mol\ HCl\ per\ mol\ Al})

.. math:: \mathrm{CaO} + 2\,\mathrm{HCl} \rightarrow
          \mathrm{CaCl_2} + \mathrm{H_2O}
          \quad (\nu = 2\ \mathrm{mol\ HCl\ per\ mol\ Ca})

.. math:: \mathrm{K_2O} + 2\,\mathrm{HCl} \rightarrow
          2\,\mathrm{KCl} + \mathrm{H_2O}
          \quad (\nu = 1\ \mathrm{mol\ HCl\ per\ mol\ K})

The general rule these follow: a cation of charge :math:`z` in an oxide consumes
:math:`z` moles of a monoprotic acid, because the acid must both supply
:math:`z` anions to the cation and :math:`2` protons per oxide oxygen to form
water. :func:`acid_demand_per_cation` applies the charge rule, so a cation not
tabulated above is handled by its charge rather than by a missing entry.

Silica attack by HF
-------------------
HF dissolves the silica matrix itself, which no other mineral acid in this set
does appreciably:

.. math:: \mathrm{SiO_2} + 6\,\mathrm{HF} \rightarrow
          \mathrm{H_2SiF_6} + 2\,\mathrm{H_2O}

Six moles of HF per mole of SiO2 dissolved. Because SiO2 is essentially the
whole charge rather than a trace, a silica dissolution of even 0.1 percent by
mass consumes

.. math:: n_{HF} = 6\,\frac{0.001\,m_{ore}}{M_{SiO_2}}
        = 6 \times \frac{0.001 \times 1000\ \mathrm{kg}}
          {0.060083\ \mathrm{kg\,mol^{-1}}}
        = 99.9\ \mathrm{mol\ per\ tonne}

against an impurity demand of 3.34 mol per tonne for 30 ppm Al
(:math:`1000 \times 30\times 10^{-6} / 0.0269815 = 1.112` mol Al, times
:math:`\nu = 3`), so the matrix consumes 30 times more HF than the impurity it
is being used to remove. That 30 ppm is taken as fully leachable for the
illustration; the functions here work on the LEACHABLE inventory only
(total minus lattice), which makes the ratio larger still, 44.7 for a feedstock
whose leachable load is 2.23 mol HF per tonne. The ratio scales linearly with
the silica dissolution:
3.0 at 0.01 percent dissolution, 30 at 0.1 percent, 299 at 1 percent, and it
rises further once the lattice-bound fraction is excluded from the leachable
impurity inventory. :func:`hf_silica_demand` computes the silica term and
:func:`reagent_balance` reports the ratio explicitly as
``silica_to_impurity_demand_ratio``, so an HF budget can never be quoted against
the impurity load alone.

An important consequence for the alternative reaction stoichiometry sometimes
quoted, :math:`\mathrm{SiO_2} + 4\,\mathrm{HF} \rightarrow \mathrm{SiF_4} +
2\,\mathrm{H_2O}`: which one applies depends on whether the product is
fluorosilicic acid in solution (6 HF) or silicon tetrafluoride gas (4 HF). Both
are provided as :data:`SILICA_HF_STOICH` options, defaulting to the 6 HF
solution route for an aqueous leach, and the choice is reported in the balance
output because it changes HF demand by 50 percent.

Neutralization demand
---------------------
The neutralization basis is the TOTAL acid charged, not the unreacted excess.
Acid consumed by a dissolution reaction is not destroyed: its anion leaves with
the liquor as a dissolved metal salt (FeCl3, AlCl3, CaCl2) or as fluorosilicic
acid, and that anion still has to be neutralized or precipitated before
discharge. Charging neutralization against the excess alone would report zero
lime and zero sludge at the stoichiometric floor (``excess_factor = 1.0``) for a
liquor carrying every mole of fluoride that was fed, which is the error this
basis exists to avoid. ``reagent_balance`` reports ``unreacted_acid_mol`` and
``acid_neutralized_mol`` separately so the two are never confused.

Using lime:

.. math:: \mathrm{Ca(OH)_2} + 2\,\mathrm{HCl} \rightarrow
          \mathrm{CaCl_2} + 2\,\mathrm{H_2O}

and for fluoride, precipitation as the sparingly soluble fluorite:

.. math:: \mathrm{Ca(OH)_2} + 2\,\mathrm{HF} \rightarrow
          \mathrm{CaF_2}(s) + 2\,\mathrm{H_2O}

so one mole of lime per two moles of monoprotic acid in both cases, and one mole
of CaF2 sludge per two moles of fluoride. Using caustic,
:math:`\mathrm{NaOH} + \mathrm{HCl} \rightarrow \mathrm{NaCl} + \mathrm{H_2O}`,
one mole per mole, but NaF is far more soluble than CaF2 so caustic does NOT
remove fluoride from the water: :func:`neutralization_demand` returns zero
fluoride precipitation for the caustic route and flags it, because treating
caustic as a fluoride treatment is a permit-level error.

Charge balance
--------------
The neutralized liquor must satisfy electroneutrality:

.. math:: \sum_i z_i n_i^{cation} + n_{H^+}
          = \sum_j |z_j| n_j^{anion} + n_{OH^-}

:func:`charge_balance` evaluates both sides and
:func:`test_charge_balance_closes` asserts closure to 1e-9 relative. A charge
imbalance means a species was double counted or omitted, which is the most
common defect in a hand-built effluent balance.

Mass balance
------------
Total mass in equals total mass out:

.. math:: m_{ore} + m_{acid} + m_{base} + m_{water}
        = m_{product} + m_{dissolved} + m_{sludge} + m_{liquor} + m_{gas}

:func:`reagent_balance` closes this to a relative tolerance of 1e-9 and raises
if it does not, rather than reporting an unbalanced flowsheet.

LIMITATIONS
===========
1. The stoichiometric demand computed here is a LOWER BOUND on real consumption.
   Real circuits run an excess (typically expressed as a multiple of
   stoichiometric), lose acid to entrainment and vapour, and regenerate
   incompletely. ``excess_factor`` defaults to 1.0, which is the stoichiometric
   floor and is NOT a plant number: a realistic value must come from a specific
   flowsheet and is an input, not a default this module can supply.
2. Impurity phase speciation is assumed. The acid demand for an element depends
   on the mineral it sits in: iron as hematite, as goethite, as a sulphide or as
   a silicate all consume different amounts of acid and dissolve at different
   rates. This module computes demand from the ELEMENTAL inventory and the
   cation charge, which is exact only if every impurity is present as the simple
   oxide. Mineral inclusion data (feldspar, mica, rutile) would refine it, and
   the direction of the error is not general: a silicate host can consume MORE
   acid than the oxide, an unreactive host LESS.
3. Lattice-substituted impurities consume no acid at all, because they are never
   contacted (see :mod:`ae.physics.diffusion`). :func:`reagent_balance` uses the
   LEACHABLE inventory when the lattice split is measured, and raises otherwise,
   so an uncharacterized ore cannot produce a reagent budget that silently
   assumes all impurities are available.
4. No activity coefficients, no speciation equilibria, no complexation. At the
   ionic strengths of a spent leach liquor, activity coefficients depart
   substantially from unity, and fluoride forms a series of complexes
   (SiF6(2-), AlF(x) species) that change both the free-fluoride concentration
   and the lime demand. A real permit calculation requires a speciation model
   (PHREEQC or equivalent), not this stoichiometry.
5. Fluoride effluent is computed as TOTAL fluoride, not as free fluoride ion.
   Discharge limits are usually written on total fluoride, so this is the right
   basis for a compliance screen, but CaF2 solubility sets a floor on what
   precipitation can achieve (CaF2 is sparingly soluble, not insoluble) and that
   floor is not computed here because no solubility product could be sourced in
   this build.
6. Sludge mass is the stoichiometric dry precipitate only. Real neutralization
   sludge carries water at 40 to 70 percent by mass after filtration and
   entrains gypsum and metal hydroxides, so the disposal tonnage is several
   times the number computed here. The multiplier is a design input.
7. No reagent prices are hardcoded. :func:`reagent_cost` reads them from the
   SITE and raises if the site does not carry a price for a reagent the recipe
   uses, or if the reagent is marked not locally available. HF availability is a
   real constraint in many jurisdictions and is enforced by
   :meth:`ae.core.site.ReagentPrices.price`, not assumed away.
8. Nitric and sulphuric acid are handled only by their proton count (HNO3
   monoprotic, H2SO4 diprotic). Their distinctive chemistry, oxidation of
   sulphides by HNO3 and calcium sulphate scaling with H2SO4, is not modelled,
   and the latter is a real flowsheet risk when a feedstock carries calcium.

References
----------
Reaction stoichiometries above are balanced from first principles (mass and
charge), not taken from a source, and :func:`check_reaction_balance` verifies
each one numerically in the test suite. Atomic weights come from
:data:`ae.core.feedstock.MOLAR_MASS` (IUPAC 2021 standard atomic weights).

Xia, M., Yang, X. and Hou, Z. (2024) Preparation of High-Purity Quartz Sand by
Vein Quartz Purification and Characteristics: A Case Study of Pakistan Vein
Quartz, Minerals 14(7), 727, doi 10.3390/min14070727. Reports the impurity
endpoints (128.86 to 24.23 ug/g) this module's impurity-load path is benchmarked
against, but reports NO reagent consumption, so no reagent figure in this module
is validated against it.

Yang, C. and Li, S. (2020) Kinetics of iron removal from quartz under
ultrasound-assisted leaching, High Temperature Materials and Processes 39(1),
395-404, doi 10.1515/htmp-2020-0081. Reports Fe2O3 0.0857 to 0.0223 percent,
used as the iron-load benchmark. Reagent consumption is not reported in the
retrievable abstract.
"""

from __future__ import annotations

import enum
import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import MOLAR_MASS, Feedstock
from ae.core.provenance import Tag, Value
from ae.core.site import Site
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = [
    "Acid",
    "Base",
    "ACID_PROTONS",
    "ACID_MOLAR_MASS",
    "BASE_EQUIVALENTS",
    "CATION_CHARGE",
    "SILICA_HF_STOICH",
    "M_SIO2",
    "Reaction",
    "LEACH_REACTIONS",
    "check_reaction_balance",
    "acid_demand_per_cation",
    "impurity_moles",
    "impurity_acid_demand",
    "hf_silica_demand",
    "neutralization_demand",
    "charge_balance",
    "fluoride_effluent",
    "reagent_balance",
    "reagent_cost",
]


class Acid(str, enum.Enum):
    HF = "HF"
    HCl = "HCl"
    HNO3 = "HNO3"
    H2SO4 = "H2SO4"


class Base(str, enum.Enum):
    LIME = "Ca(OH)2"
    CAUSTIC = "NaOH"


#: Protons per molecule. Sets acid equivalents and neutralization demand.
ACID_PROTONS: Final[dict[Acid, int]] = {
    Acid.HF: 1, Acid.HCl: 1, Acid.HNO3: 1, Acid.H2SO4: 2,
}

#: g/mol, computed from IUPAC 2021 atomic weights.
ACID_MOLAR_MASS: Final[dict[Acid, float]] = {
    Acid.HF: 1.008 + 18.998,            # 20.006
    Acid.HCl: 1.008 + 35.45,            # 36.458
    Acid.HNO3: 1.008 + 14.007 + 3 * 15.999,   # 63.012
    Acid.H2SO4: 2 * 1.008 + 32.06 + 4 * 15.999,  # 98.072
}

#: Hydroxide equivalents per molecule of base, and its molar mass in g/mol.
BASE_EQUIVALENTS: Final[dict[Base, tuple[int, float]]] = {
    Base.LIME: (2, 40.078 + 2 * (15.999 + 1.008)),   # Ca(OH)2, 74.092
    Base.CAUSTIC: (1, 22.9898 + 15.999 + 1.008),     # NaOH, 39.997
}

#: Formal cation charge in the oxide, which sets monoprotic acid demand.
CATION_CHARGE: Final[dict[str, int]] = {
    "Al": 3, "Ti": 4, "Fe": 3, "Li": 1, "Na": 1, "K": 1, "Ca": 2, "B": 3,
    "Mg": 2, "Mn": 2, "Cr": 3, "Cu": 2, "Zr": 4, "P": 5, "Ge": 4, "U": 6, "Th": 4,
}

#: Moles of HF per mole of SiO2, by product route. The choice changes HF demand
#: by 50 percent and is therefore reported, never silently defaulted.
SILICA_HF_STOICH: Final[dict[str, int]] = {
    "H2SiF6_solution": 6,   # SiO2 + 6 HF -> H2SiF6 + 2 H2O
    "SiF4_gas": 4,          # SiO2 + 4 HF -> SiF4 + 2 H2O
}

#: kg/mol for SiO2, from IUPAC 2021 atomic weights: 28.085 + 2(15.999).
M_SIO2: Final[float] = (MOLAR_MASS["Si"] + 2 * MOLAR_MASS["O"]) / 1000.0


class Reaction(BaseModel):
    """A balanced dissolution reaction, stored so balance can be verified."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    #: Species to (stoichiometric coefficient, charge, {element: atoms per formula unit}).
    reactants: dict[str, tuple[float, int, dict[str, int]]]
    products: dict[str, tuple[float, int, dict[str, int]]]
    acid: Acid
    cation: str
    acid_per_cation: float = Field(gt=0.0)


def _sp(coeff: float, charge: int, atoms: dict[str, int]) -> tuple[float, int, dict[str, int]]:
    return coeff, charge, atoms


#: Balanced reactions used for the tabulated cases. Every one is checked by
#: check_reaction_balance in the test suite, for both element counts and charge.
LEACH_REACTIONS: Final[dict[str, Reaction]] = {
    "hematite_HCl": Reaction(
        name="Fe2O3 + 6 HCl -> 2 FeCl3 + 3 H2O",
        reactants={"Fe2O3": _sp(1, 0, {"Fe": 2, "O": 3}),
                   "HCl": _sp(6, 0, {"H": 1, "Cl": 1})},
        products={"FeCl3": _sp(2, 0, {"Fe": 1, "Cl": 3}),
                  "H2O": _sp(3, 0, {"H": 2, "O": 1})},
        acid=Acid.HCl, cation="Fe", acid_per_cation=3.0),
    "alumina_HCl": Reaction(
        name="Al2O3 + 6 HCl -> 2 AlCl3 + 3 H2O",
        reactants={"Al2O3": _sp(1, 0, {"Al": 2, "O": 3}),
                   "HCl": _sp(6, 0, {"H": 1, "Cl": 1})},
        products={"AlCl3": _sp(2, 0, {"Al": 1, "Cl": 3}),
                  "H2O": _sp(3, 0, {"H": 2, "O": 1})},
        acid=Acid.HCl, cation="Al", acid_per_cation=3.0),
    "lime_HCl": Reaction(
        name="CaO + 2 HCl -> CaCl2 + H2O",
        reactants={"CaO": _sp(1, 0, {"Ca": 1, "O": 1}),
                   "HCl": _sp(2, 0, {"H": 1, "Cl": 1})},
        products={"CaCl2": _sp(1, 0, {"Ca": 1, "Cl": 2}),
                  "H2O": _sp(1, 0, {"H": 2, "O": 1})},
        acid=Acid.HCl, cation="Ca", acid_per_cation=2.0),
    "potash_HCl": Reaction(
        name="K2O + 2 HCl -> 2 KCl + H2O",
        reactants={"K2O": _sp(1, 0, {"K": 2, "O": 1}),
                   "HCl": _sp(2, 0, {"H": 1, "Cl": 1})},
        products={"KCl": _sp(2, 0, {"K": 1, "Cl": 1}),
                  "H2O": _sp(1, 0, {"H": 2, "O": 1})},
        acid=Acid.HCl, cation="K", acid_per_cation=1.0),
    "silica_HF": Reaction(
        name="SiO2 + 6 HF -> H2SiF6 + 2 H2O",
        reactants={"SiO2": _sp(1, 0, {"Si": 1, "O": 2}),
                   "HF": _sp(6, 0, {"H": 1, "F": 1})},
        products={"H2SiF6": _sp(1, 0, {"H": 2, "Si": 1, "F": 6}),
                  "H2O": _sp(2, 0, {"H": 2, "O": 1})},
        acid=Acid.HF, cation="Si", acid_per_cation=6.0),
}


def check_reaction_balance(reaction: Reaction) -> dict[str, float]:
    """Return per-element and charge residuals for a reaction. All must be zero.

    A non-zero residual means the reaction as written creates or destroys atoms
    or charge, so any stoichiometric factor derived from it is wrong.
    """
    resid: dict[str, float] = {}
    elements = set()
    for side in (reaction.reactants, reaction.products):
        for _, (_, _, atoms) in side.items():
            elements.update(atoms)
    for el in sorted(elements):
        left = sum(c * atoms.get(el, 0) for _, (c, _, atoms) in reaction.reactants.items())
        right = sum(c * atoms.get(el, 0) for _, (c, _, atoms) in reaction.products.items())
        resid[el] = float(left - right)
    q_left = sum(c * z for _, (c, z, _) in reaction.reactants.items())
    q_right = sum(c * z for _, (c, z, _) in reaction.products.items())
    resid["charge"] = float(q_left - q_right)
    return resid


def acid_demand_per_cation(element: str, acid: Acid) -> float:
    r"""Moles of acid consumed per mole of ``element`` dissolved from its oxide.

    Applies the charge rule: a cation of charge :math:`z` needs :math:`z`
    equivalents of acid, and an acid supplying :math:`p` protons per molecule
    therefore contributes :math:`p` equivalents, giving :math:`\nu = z/p`.
    For HCl (p = 1) on Fe3+ this returns 3, matching the balanced hematite
    reaction above; for H2SO4 (p = 2) it returns 1.5.
    """
    if element not in CATION_CHARGE:
        raise KeyError(
            f"no formal charge tabulated for {element!r}; tabulated: "
            f"{sorted(CATION_CHARGE)}. Acid demand cannot be inferred without it."
        )
    z = CATION_CHARGE[element]
    p = ACID_PROTONS[acid]
    nu = z / p
    assert nu > 0.0, "acid demand per cation must be positive"
    return float(nu)


def impurity_moles(
    feedstock: Feedstock, ore_mass: Quantity, elements: tuple[str, ...] | None = None,
    use_leachable: bool = True,
) -> dict[str, Quantity]:
    """Moles of each impurity element in a charge of ``ore_mass``.

    With ``use_leachable=True`` the lattice-bound fraction is excluded, because
    it is never contacted by reagent, and the call raises
    :class:`ae.core.provenance.MissingValueError` when the lattice split has not
    been measured. That is the intended behaviour: a reagent budget that assumes
    the whole inventory is available overstates consumption and understates the
    achievable product grade at the same time.
    """
    require_dimensionality(ore_mass, "mass", "ore_mass")
    m = ore_mass.to("kg")
    if float(m.magnitude) <= 0.0:
        raise ValueError(f"ore mass must be positive, got {m}")
    els = elements or tuple(e for e in feedstock.impurities.total if e != "OH")
    out: dict[str, Quantity] = {}
    for el in els:
        if use_leachable:
            from ae.physics.leaching import leachable_ppm
            ppm = leachable_ppm(feedstock, el)
        else:
            ppm = feedstock.impurities.total_ppm(el)
        if el not in MOLAR_MASS:
            raise KeyError(f"no molar mass for {el!r} in ae.core.feedstock.MOLAR_MASS")
        moles = m * (ppm * 1e-6) / Q_(MOLAR_MASS[el], "g/mol")
        out[el] = moles.to("mol")
        assert float(out[el].magnitude) >= 0.0, f"{el}: negative moles is unphysical"
    return out


def impurity_acid_demand(
    feedstock: Feedstock, ore_mass: Quantity, acid: Acid,
    elements: tuple[str, ...] | None = None, use_leachable: bool = True,
) -> tuple[Quantity, dict[str, Quantity]]:
    """Total and per-element acid demand for dissolving the impurity load."""
    moles = impurity_moles(feedstock, ore_mass, elements, use_leachable)
    per: dict[str, Quantity] = {}
    total = Q_(0.0, "mol")
    for el, n in moles.items():
        d = n * acid_demand_per_cation(el, acid)
        per[el] = d.to("mol")
        total = total + d
    return total.to("mol"), per


def hf_silica_demand(
    ore_mass: Quantity, silica_dissolved_fraction: float,
    route: str = "H2SiF6_solution",
) -> tuple[Quantity, Quantity]:
    r"""HF consumed by attack on the silica matrix, and the mass of SiO2 lost.

    Returns ``(moles_HF, mass_SiO2_dissolved)``. This is usually the dominant HF
    consumer and the dominant yield loss, and it scales with the CHARGE mass,
    not with the impurity level.
    """
    require_dimensionality(ore_mass, "mass", "ore_mass")
    f = require_fraction(silica_dissolved_fraction, "silica_dissolved_fraction")
    if route not in SILICA_HF_STOICH:
        raise KeyError(
            f"unknown silica-HF route {route!r}; known: {sorted(SILICA_HF_STOICH)}. The "
            f"choice changes HF demand by 50 percent and must be stated."
        )
    nu = SILICA_HF_STOICH[route]
    m_diss = (ore_mass.to("kg") * f).to("kg")
    n_hf = (nu * m_diss / Q_(M_SIO2, "kg/mol")).to("mol")
    assert float(n_hf.magnitude) >= 0.0, "HF demand cannot be negative"
    return n_hf, m_diss


def neutralization_demand(
    acid_moles: dict[Acid, Quantity], base: Base,
) -> dict[str, object]:
    r"""Base required to neutralize residual acid, and the sludge produced.

    Demand is computed on PROTON equivalents, so a diprotic acid counts twice,
    and divided by the hydroxide equivalents of the base.

    Fluoride precipitation. Lime precipitates fluoride as CaF2, one mole of
    CaF2 per two moles of F. Caustic does NOT: NaF is soluble, so the caustic
    route returns zero fluoride precipitated and sets
    ``fluoride_removed_by_base=False``. Treating caustic as fluoride treatment
    is a discharge-permit error, so it is flagged rather than left implicit.
    """
    eq_per_base, m_base = BASE_EQUIVALENTS[base]
    proton_eq = Q_(0.0, "mol")
    for acid, n in acid_moles.items():
        if float(n.to("mol").magnitude) < 0.0:
            raise ValueError(f"{acid.value}: negative moles is unphysical")
        proton_eq = proton_eq + n.to("mol") * ACID_PROTONS[acid]
    base_moles = (proton_eq / eq_per_base).to("mol")
    base_mass = (base_moles * Q_(m_base, "g/mol")).to("kg")

    f_moles = acid_moles.get(Acid.HF, Q_(0.0, "mol")).to("mol")
    if base is Base.LIME:
        caf2_moles = (f_moles / 2.0).to("mol")
        m_caf2 = MOLAR_MASS["Ca"] + 2 * 18.998
        caf2_mass = (caf2_moles * Q_(m_caf2, "g/mol")).to("kg")
        f_removed = True
        note = (
            "Lime precipitates fluoride as CaF2 (Ca(OH)2 + 2 HF -> CaF2 + 2 H2O). CaF2 is "
            "sparingly soluble, not insoluble, so a residual dissolved fluoride remains; "
            "its floor is set by the CaF2 solubility product, which could not be sourced "
            "in this build and is therefore NOT applied here."
        )
    else:
        caf2_moles = Q_(0.0, "mol")
        caf2_mass = Q_(0.0, "kg")
        f_removed = False
        note = (
            "Caustic neutralizes acidity but does NOT remove fluoride: NaF is soluble, so "
            "all fluoride charged remains in the discharge stream. A fluoride limit cannot "
            "be met on the caustic route without a separate precipitation step."
        )
    assert float(base_moles.magnitude) >= 0.0, "base demand cannot be negative"
    return {
        "base": base.value,
        "proton_equivalents_mol": float(proton_eq.magnitude),
        "base_moles_mol": float(base_moles.magnitude),
        "base_mass_kg": float(base_mass.magnitude),
        "fluoride_removed_by_base": f_removed,
        "CaF2_moles_mol": float(caf2_moles.magnitude),
        "CaF2_sludge_dry_kg": float(caf2_mass.magnitude),
        "note": note,
    }


def charge_balance(
    cations: dict[str, Quantity], anions: dict[str, Quantity],
    h_plus: Quantity | None = None, oh_minus: Quantity | None = None,
) -> dict[str, float]:
    r"""Evaluate electroneutrality of a liquor.

    ``cations`` maps element symbol to moles (charge taken from
    :data:`CATION_CHARGE`); ``anions`` maps anion name to moles, with the charge
    magnitude read from :data:`_ANION_CHARGE`. Returns both sides and the
    residual; a non-zero residual means a species was omitted or double counted.
    """
    pos = 0.0
    for el, n in cations.items():
        if el not in CATION_CHARGE:
            raise KeyError(f"no charge for cation {el!r}")
        pos += CATION_CHARGE[el] * float(n.to("mol").magnitude)
    if h_plus is not None:
        pos += float(h_plus.to("mol").magnitude)
    neg = 0.0
    for an, n in anions.items():
        if an not in _ANION_CHARGE:
            raise KeyError(
                f"no charge for anion {an!r}; tabulated: {sorted(_ANION_CHARGE)}"
            )
        neg += abs(_ANION_CHARGE[an]) * float(n.to("mol").magnitude)
    if oh_minus is not None:
        neg += float(oh_minus.to("mol").magnitude)
    scale = max(abs(pos), abs(neg), 1e-30)
    return {
        "cation_equivalents": pos,
        "anion_equivalents": neg,
        "residual": pos - neg,
        "relative_residual": (pos - neg) / scale,
    }


#: Anion charges for the species this module's liquors contain.
_ANION_CHARGE: Final[dict[str, int]] = {
    "Cl": -1, "F": -1, "NO3": -1, "SO4": -2, "SiF6": -2, "OH": -1,
}


def fluoride_effluent(
    site: Site, fluoride_moles: Quantity, effluent_volume: Quantity,
    fluoride_precipitated_moles: Quantity | None = None,
) -> dict[str, object]:
    """Fluoride discharge concentration against the site's permitted limit.

    Raises
    ------
    KeyError
        If the site's permitting regime carries no fluoride limit. A fluoride
        discharge number with no limit to compare against is not a compliance
        result, and inventing a limit would misrepresent the jurisdiction.
    """
    require_dimensionality(effluent_volume, "volume", "effluent_volume")
    if float(effluent_volume.to("m**3").magnitude) <= 0.0:
        raise ValueError("effluent volume must be positive")
    n_total = fluoride_moles.to("mol")
    n_ppt = (fluoride_precipitated_moles or Q_(0.0, "mol")).to("mol")
    n_free = n_total - n_ppt
    if float(n_free.magnitude) < 0.0:
        raise ValueError(
            f"precipitated fluoride ({n_ppt}) exceeds total fluoride ({n_total}); more "
            f"cannot be removed than was charged"
        )
    m_f = (n_free * Q_(18.998, "g/mol")).to("mg")
    conc = (m_f / effluent_volume.to("L")).to("mg/L")

    if site.permitting is None or "F" not in site.permitting.effluent_limits:
        raise KeyError(
            f"site {site.site_id} carries no fluoride effluent limit "
            f"(permitting={'absent' if site.permitting is None else 'present'}); a "
            f"discharge concentration cannot be screened against an unstated limit, and "
            f"the limit must come from the jurisdiction, not from this module"
        )
    limit = site.permitting.effluent_limits["F"].quantity.to("mg/L")
    ratio = float((conc / limit).to("dimensionless").magnitude)
    return {
        "site_id": site.site_id,
        "fluoride_total_mol": float(n_total.magnitude),
        "fluoride_precipitated_mol": float(n_ppt.magnitude),
        "fluoride_discharged_mol": float(n_free.magnitude),
        "concentration_mg_per_L": float(conc.magnitude),
        "limit_mg_per_L": float(limit.magnitude),
        "ratio_to_limit": ratio,
        "compliant": ratio <= 1.0,
        "limit_source": (
            site.permitting.effluent_limits["F"].source.citation
            if site.permitting.effluent_limits["F"].source else
            f"tag {site.permitting.effluent_limits['F'].tag.value}: "
            f"{site.permitting.effluent_limits['F'].basis}"
        ),
    }


def reagent_balance(
    feedstock: Feedstock,
    site: Site,
    ore_mass: Quantity,
    acid: Acid,
    base: Base = Base.LIME,
    silica_dissolved_fraction: float = 0.0,
    excess_factor: float = 1.0,
    water_mass: Quantity | None = None,
    silica_hf_route: str = "H2SiF6_solution",
    elements: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Full reagent, neutralization and mass balance for one leach charge.

    Closes the overall mass balance to a relative tolerance of 1e-9 and raises
    if it does not. ``excess_factor`` multiplies the stoichiometric acid demand
    and defaults to 1.0, the stoichiometric floor, which is explicitly not a
    plant number (LIMITATIONS item 1).
    """
    require_dimensionality(ore_mass, "mass", "ore_mass")
    if excess_factor < 1.0:
        raise ValueError(
            f"excess_factor must be at least 1.0 (stoichiometric); {excess_factor} would "
            f"charge less acid than the reaction consumes"
        )
    imp_demand, imp_per_el = impurity_acid_demand(
        feedstock, ore_mass, acid, elements, use_leachable=True
    )
    if acid is Acid.HF:
        si_demand, m_si_diss = hf_silica_demand(
            ore_mass, silica_dissolved_fraction, silica_hf_route
        )
    else:
        si_demand, m_si_diss = Q_(0.0, "mol"), Q_(0.0, "kg")
        if silica_dissolved_fraction > 0.0:
            raise ValueError(
                f"silica_dissolved_fraction={silica_dissolved_fraction} was given with "
                f"{acid.value}, but only HF attacks the silica matrix appreciably; set it "
                f"to zero or use HF"
            )
    stoich_total = (imp_demand + si_demand).to("mol")
    charged = (stoich_total * excess_factor).to("mol")
    unreacted_acid = (charged - stoich_total).to("mol")
    acid_mass = (charged * Q_(ACID_MOLAR_MASS[acid], "g/mol")).to("kg")

    # Neutralization is charged on the WHOLE acid inventory, not on the unreacted
    # excess alone. Acid consumed by the reaction is not destroyed: it leaves as
    # a dissolved metal salt (FeCl3, AlCl3) or as fluorosilicic acid, and its
    # anion still has to be neutralized or precipitated before discharge. Basing
    # the lime demand on the excess alone would report zero reagent and zero
    # sludge at the stoichiometric floor (excess_factor = 1.0) for a liquor that
    # still carries every mole of fluoride charged.
    neut = neutralization_demand({acid: charged}, base)

    # Dissolved impurity mass leaving with the liquor.
    imp_moles = impurity_moles(feedstock, ore_mass, elements, use_leachable=True)
    m_imp_diss = Q_(0.0, "kg")
    for el, n in imp_moles.items():
        m_imp_diss = m_imp_diss + (n * Q_(MOLAR_MASS[el], "g/mol")).to("kg")

    m_water = (water_mass or Q_(0.0, "kg")).to("kg")
    m_base = Q_(neut["base_mass_kg"], "kg")
    m_sludge = Q_(neut["CaF2_sludge_dry_kg"], "kg")

    m_in = (ore_mass.to("kg") + acid_mass + m_base + m_water).to("kg")
    m_product = (ore_mass.to("kg") - m_si_diss - m_imp_diss).to("kg")
    # Everything not in the solid product or the sludge reports to the liquor.
    m_liquor = (m_in - m_product - m_sludge).to("kg")
    m_out = (m_product + m_sludge + m_liquor).to("kg")

    rel = abs(float((m_out - m_in).magnitude)) / max(float(m_in.magnitude), 1e-30)
    if rel > 1e-9:
        raise AssertionError(
            f"mass balance does not close: in {m_in}, out {m_out}, relative residual "
            f"{rel:.3e}. An unbalanced flowsheet must not be reported."
        )
    if float(m_product.magnitude) < 0.0:
        raise ValueError(
            f"computed product mass is negative ({m_product}): the specified silica "
            f"dissolution consumes more than the charge"
        )

    ratio = (
        float((si_demand / imp_demand).to("dimensionless").magnitude)
        if float(imp_demand.magnitude) > 0.0 else float("inf")
    )
    return {
        "sample_id": feedstock.sample_id,
        "site_id": site.site_id,
        "characterized": feedstock.characterized,
        "acid": acid.value,
        "ore_mass_kg": float(ore_mass.to("kg").magnitude),
        "impurity_acid_demand_mol": float(imp_demand.magnitude),
        "impurity_acid_demand_per_element_mol": {
            k: float(v.magnitude) for k, v in imp_per_el.items()},
        "silica_acid_demand_mol": float(si_demand.magnitude),
        "silica_hf_route": silica_hf_route if acid is Acid.HF else None,
        "silica_to_impurity_demand_ratio": ratio,
        "stoichiometric_acid_mol": float(stoich_total.magnitude),
        "excess_factor": float(excess_factor),
        "acid_charged_mol": float(charged.magnitude),
        "acid_charged_kg": float(acid_mass.magnitude),
        "unreacted_acid_mol": float(unreacted_acid.magnitude),
        "acid_neutralized_mol": float(charged.magnitude),
        "neutralization": neut,
        "silica_dissolved_kg": float(m_si_diss.magnitude),
        "impurity_dissolved_kg": float(m_imp_diss.magnitude),
        "product_mass_kg": float(m_product.magnitude),
        "liquor_mass_kg": float(m_liquor.magnitude),
        "sludge_dry_kg": float(m_sludge.magnitude),
        "mass_in_kg": float(m_in.magnitude),
        "mass_out_kg": float(m_out.magnitude),
        "mass_balance_relative_residual": rel,
    }


def reagent_cost(
    site: Site, acid: Acid, acid_mass: Quantity, base: Base | None = None,
    base_mass: Quantity | None = None,
) -> dict[str, object]:
    """Delivered reagent cost from the SITE's own price list.

    No price is hardcoded. Raises through
    :meth:`ae.core.site.ReagentPrices.price` if the site does not carry a price
    for the reagent, or if the reagent is priced but marked not locally
    available (HF is the usual case, and its availability is a real constraint
    rather than a modelling nuisance).
    """
    require_dimensionality(acid_mass, "mass", "acid_mass")
    p_acid = site.reagents.price(acid.value)
    cost = (p_acid * acid_mass.to("kg")).to(str(site.currency.value))
    out: dict[str, object] = {
        "site_id": site.site_id,
        "currency": site.currency.value,
        "acid": acid.value,
        "acid_mass_kg": float(acid_mass.to("kg").magnitude),
        "acid_unit_price": f"{p_acid:~P}",
        "acid_cost": float(cost.magnitude),
        "base_cost": 0.0,
        "total_cost": float(cost.magnitude),
    }
    if base is not None:
        if base_mass is None:
            raise ValueError("base_mass is required when a base is specified")
        require_dimensionality(base_mass, "mass", "base_mass")
        p_base = site.reagents.price(base.value)
        c_base = (p_base * base_mass.to("kg")).to(str(site.currency.value))
        out["base"] = base.value
        out["base_mass_kg"] = float(base_mass.to("kg").magnitude)
        out["base_unit_price"] = f"{p_base:~P}"
        out["base_cost"] = float(c_base.magnitude)
        out["total_cost"] = float(cost.magnitude) + float(c_base.magnitude)
    return out
