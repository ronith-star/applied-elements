r"""Shrinking-core leaching kinetics for impurity removal from a silica feedstock.

Scope: hot acid leaching of quartz concentrate, where a reagent (HF, HCl, HNO3,
H2SO4, or a mixture) dissolves discrete impurity phases (mineral inclusions,
grain-boundary films, fracture-filling oxides) from a particle of quartz. The
model answers one question: for a given particle size, reagent concentration,
temperature and residence time, what fraction of the LEACHABLE impurity
inventory is removed. It does not and cannot address lattice-substituted Al, Ti,
Li or B, which are unreachable by any aqueous reagent (see
:mod:`ae.physics.diffusion` for the quantitative reason).

EQUATIONS
=========

Geometry and definitions
------------------------
A spherical particle of initial radius :math:`R` contains a reactive solid B at
molar density :math:`\rho_B`. Fluid reagent A at bulk molar concentration
:math:`C_{A}` reacts by

.. math:: A(\mathrm{fluid}) + b\,B(\mathrm{solid}) \rightarrow \mathrm{products}

where :math:`b` is moles of B consumed per mole of A. Conversion of B is

.. math:: X = 1 - \left(\frac{r_c}{R}\right)^3, \qquad u \equiv (1-X)^{1/3} = \frac{r_c}{R}

with :math:`r_c` the unreacted-core radius. Symbols, units and valid ranges:

===================  =============  ==================================
Symbol               Unit           Valid range
===================  =============  ==================================
:math:`R`            m              1e-7 to 1e-1 (0.1 um to 100 mm)
:math:`\rho_B`       mol m^-3       > 0
:math:`C_A`          mol m^-3       > 0
:math:`b`            dimensionless  > 0
:math:`k_g`          m s^-1         > 0 (film coefficient)
:math:`k_s`          m s^-1         > 0 (surface rate constant)
:math:`D_e`          m^2 s^-1       > 0 (effective pore diffusivity)
:math:`X`            dimensionless  [0, 1]
:math:`t`            s              >= 0
:math:`\tau`         s              > 0 (time for complete conversion)
:math:`E_a`          J mol^-1       > 0
:math:`T`            K              > 0
:math:`\mathcal{R}`  J mol^-1 K^-1  8.314462618 (CODATA 2018)
===================  =============  ==================================

Regime 1: fluid-film (external mass transfer) control
-----------------------------------------------------
Derivation from first principles. The molar flux of A to the particle surface is
:math:`N_A = k_g C_A` (mol m^-2 s^-1), the external area :math:`4\pi R^2` is
constant for a particle of unchanging size, and every mole arriving is consumed,
so the consumption of B is

.. math:: -\frac{dN_B}{dt} = 4\pi R^2 b k_g C_A = \mathrm{constant}.

With :math:`N_B = \tfrac{4}{3}\pi R^3 \rho_B (1-X)` this integrates to

.. math:: X(t) = \frac{t}{\tau_f}, \qquad
          \boxed{\tau_f = \frac{\rho_B R}{3 b k_g C_A}}

Conversion is LINEAR in time. Dimensional check:
(mol m^-3)(m) / [(m s^-1)(mol m^-3)] = s.

Regime 2: product-layer (ash-layer) diffusion control
-----------------------------------------------------
Derivation. A is transported through a porous residue of thickness
:math:`R - r_c` by quasi-steady diffusion. Spherical-shell Fick's first law with
constant flux of moles per unit time :math:`\dot{n}` gives
:math:`\dot{n} = 4\pi D_e C_A \left(1/r_c - 1/R\right)^{-1}`. Equating to the
rate of shrinkage of the core, :math:`\dot{n} b = -4\pi r_c^2 \rho_B \, dr_c/dt`,
and integrating from :math:`r_c = R` gives Levenspiel's ash-diffusion relation

.. math:: g_{pl}(X) \equiv 1 - 3(1-X)^{2/3} + 2(1-X) = \frac{t}{\tau_{pl}},
          \qquad
          \boxed{\tau_{pl} = \frac{\rho_B R^2}{6 b D_e C_A}}

Dimensional check: (mol m^-3)(m^2) / [(m^2 s^-1)(mol m^-3)] = s.

Closed form for X(t). Writing :math:`u = (1-X)^{1/3}`,
:math:`g_{pl} = 2u^3 - 3u^2 + 1 = (u-1)^2 (2u+1)`, so
:math:`2u^3 - 3u^2 + (1-\theta) = 0` with :math:`\theta = t/\tau_{pl}`. The
substitution :math:`u = v + 1/2` depresses it to :math:`v^3 - \tfrac{3}{4} v +
\left(\tfrac{1}{4} - \tfrac{\theta}{2}\right) = 0`, whose three roots are real
for :math:`\theta \in [0,1]`. The trigonometric solution is

.. math:: u(\theta) = \frac{1}{2} + \cos\!\left(\frac{\arccos(2\theta-1)}{3}
          + \frac{4\pi}{3}\right), \qquad X = 1 - u^3

This branch (:math:`k=2`) is the one satisfying :math:`u(0)=1` and
:math:`u(1)=0`; the other two leave [0, 1]. :func:`conversion` uses it.
``test_closed_form_inversion_matches_bracketed_root_solve`` in
``tests/test_leaching.py`` checks the closed form against an independent
bisection root of :math:`g(X) - \theta = 0` at nine values of
:math:`\theta` spanning [0, 1], and
``test_golden_conversion_inversions_are_exact`` pins the hand-traceable
arithmetic at :math:`\theta = 0.25`. This is an exact algebraic inversion, not
a fit.

Regime 3: surface-reaction control
----------------------------------
Derivation. A first-order heterogeneous reaction at the core surface consumes A
at :math:`k_s C_A` per unit core area, so
:math:`-4\pi r_c^2 \rho_B \, dr_c/dt = 4\pi r_c^2 b k_s C_A`, i.e. the core
radius recedes at constant velocity :math:`dr_c/dt = -b k_s C_A/\rho_B`.
Integrating,

.. math:: g_{sr}(X) \equiv 1 - (1-X)^{1/3} = \frac{t}{\tau_{sr}},
          \qquad
          \boxed{\tau_{sr} = \frac{\rho_B R}{b k_s C_A}}, \qquad
          X(t) = 1 - \left(1 - \frac{t}{\tau_{sr}}\right)^3

Note :math:`\tau_{sr} = 3 \tau_f` for :math:`k_s = k_g`: the same surface rate
constant gives a threefold longer completion time when the reaction front
recedes into the particle than when it sits at the fixed outer surface.

Temperature dependence
----------------------
Every rate constant and diffusivity is Arrhenius,

.. math:: k(T) = A \exp\!\left(-\frac{E_a}{\mathcal{R} T}\right)

with :math:`A` in the same unit as :math:`k`. The apparent activation energy is
the standard regime discriminator: film control is weakly temperature dependent
(:math:`E_a` of order 5 to 20 kJ mol^-1, set by liquid viscosity), product-layer
diffusion is intermediate (roughly 20 to 40 kJ mol^-1), and surface reaction
control is strong (typically above 40 kJ mol^-1). Yang and Li (2020) report
27.72 kJ mol^-1 for ultrasound-assisted Fe removal from quartz and assign
product-layer diffusion as rate determining, which is consistent with that
banding. The banding is a heuristic, not a law: it is stated as ASSUMED in
:data:`EA_REGIME_BANDS` with its basis, and :func:`identify_regime` reports it
only as corroboration of a fit, never as the primary evidence.

Particle-size discrimination
----------------------------
The three regimes scale differently in :math:`R` (:math:`\tau_f \propto R`,
:math:`\tau_{pl} \propto R^2`, :math:`\tau_{sr} \propto R`), so a size series
separates product-layer diffusion from the other two but cannot separate film
from surface-reaction control. That separation needs a stirring-speed series:
:math:`k_g` depends on agitation, :math:`k_s` does not.
:func:`size_exponent_from_series` implements the former.

VALIDATION STATUS
=================
The conversion-time FORM of this model is NOT validated against experimental
quartz data in this build. Retrieval attempts logged 2026-09-16, per source:

- Yang and Li (2020), doi 10.1515/htmp-2020-0081: full-text retrieval attempted
  via Unpaywall (200, no OA location), Semantic Scholar (DOI redirect only), PMC
  (no PMCID), CrossRef TDM (no accessible content) and DOI resolution (HTTP
  202). A direct request to the publisher (degruyter.com) was refused by the
  sandbox network allowlist. The ABSTRACT was retrieved, giving the activation
  energies (27.72 kJ/mol ultrasound-assisted, 20.44 kJ/mol regular), the
  endpoint (74 percent Fe removal in 40 min) and the assay pair (Fe2O3 0.0857 to
  0.0223 percent). No per-point conversion-time table.
- Arslan and Bayat (2009), doi 10.1007/bf03403416: full-text retrieval attempted
  via Unpaywall (200, no OA location), Semantic Scholar (200, no openAccessPdf),
  PMC (no PMCID) and CrossRef TDM (no accessible content); the DOI landing page
  resolved but served no machine-readable text. No abstract and no data table
  could be extracted.
- Europe PMC full-text searches for open-access shrinking-core leaching datasets
  ("shrinking core" with "quartz", "product layer diffusion" with "leaching
  kinetics") returned zero hits.

Consequently:

- :func:`identify_regime` is exercised by a SYNTHETIC self-consistency test
  (``test_synthetic_regime_recovery`` in ``tests/test_leaching.py``), which
  confirms the fitter recovers the regime and tau it was given. That is a test
  of the code, NOT a validation of the physics. It is marked as such in the test
  name and docstring and is not a ``@pytest.mark.benchmark``.
- The two benchmark tests in this module validate the IMPURITY ACCOUNTING
  (removal fraction from feed and product assay) against published endpoints,
  with the error reported. They do not validate the kinetic form.

LIMITATIONS
===========
1. Lattice impurities are outside this model entirely. :func:`leachable_ppm`
   raises :class:`ae.core.provenance.MissingValueError` when the lattice split
   has not been measured, because the leachable inventory is
   ``total - lattice`` and guessing the split fabricates the purity ceiling.
   Do not add a default lattice fraction to make this function return.
2. The kinetic FORM is unvalidated for quartz (see VALIDATION STATUS). Treat
   any tau or activation energy this module produces from a fit as a
   description of the supplied data, not as transferable to another ore.
3. Single-size, single-shape, isothermal, constant-bulk-concentration. Real
   leach circuits have a size distribution (Rosin-Rammler or similar), and
   integrating conversion over that distribution can shift the time to 90
   percent conversion by tens of percent. :func:`conversion_over_size_distribution`
   does that integration; the single-size functions do not.
4. Constant :math:`C_A` assumes reagent in large excess. For HF leaching of
   quartz this can fail badly, because HF attacks the silica matrix itself
   (see :mod:`ae.physics.reagents`), so reagent depletion may be dominated by
   the matrix rather than the impurity. If the computed reagent consumption from
   :mod:`ae.physics.reagents` approaches the reagent charged, the constant-:math:`C_A`
   assumption is void and this model over-predicts conversion.
5. No shrinking-particle case. If the solid being dissolved is the bulk of the
   particle (a carbonate sand, say) rather than a dispersed minor phase, the
   particle shrinks and the film-control relation changes to Levenspiel's
   shrinking-sphere form with Stokes or Ranz-Marshall mass transfer. Quartz
   leaching is a minor-phase problem, so the unchanging-size treatment applies,
   but do not reuse these functions for a matrix-dissolution problem.
6. Mixed control is not modelled. When two resistances are comparable, the
   additive-resistance sum (Levenspiel section 25.3) applies and none of the
   three single-regime linearisations fits well. :func:`identify_regime`
   reports ``mixed`` when no single regime clears the R-squared margin, rather
   than forcing a choice.
7. Ultrasound, pressure (autoclave), and chelating additives are not modelled.
   Yang and Li (2020) report that ultrasound RAISES the apparent activation
   energy from 20.44 to 27.72 kJ mol^-1 while shortening the time to 74 percent
   Fe removal to 40 minutes, so ultrasound is not a simple multiplier on the
   rate constant and cannot be represented in this framework without its own
   calibration.
8. Temperature range. The Arrhenius form is extrapolated freely by
   :meth:`Arrhenius.at`, which is dangerous above the reagent boiling point at
   the operating pressure. The calibration domain of any fitted pair is recorded
   on the :class:`Arrhenius` object and :meth:`Arrhenius.at` warns outside it.

References
----------
Levenspiel, O. (1999) Chemical Reaction Engineering, 3rd edition, Wiley,
chapter 25 (shrinking-core model for particles of unchanging size). The three
conversion-time relations above are re-derived from first principles in this
docstring rather than quoted, so the derivation can be checked independently of
the book.

Yang, C. and Li, S. (2020) Kinetics of iron removal from quartz under
ultrasound-assisted leaching, High Temperature Materials and Processes 39(1),
395-404, doi 10.1515/htmp-2020-0081.

Xia, M., Yang, X. and Hou, Z. (2024) Preparation of High-Purity Quartz Sand by
Vein Quartz Purification and Characteristics: A Case Study of Pakistan Vein
Quartz, Minerals 14(7), 727, doi 10.3390/min14070727.

Tiesinga, E., Mohr, P. J., Newell, D. B. and Taylor, B. N. (2021) CODATA
Recommended Values of the Fundamental Physical Constants: 2018, Reviews of
Modern Physics 93(2), article 025010, doi 10.1103/RevModPhys.93.025010 (also
published as Journal of Physical and Chemical Reference Data 50(3), article
033105, doi 10.1063/5.0064853). The source of the in-text "CODATA 2018" label on
the molar gas constant. The 2021 publication year is the journal record for the
2018 adjustment, which is the name the adjustment is known by. The value
8.314462618 J mol^-1 K^-1 is EXACT by the 2019 SI redefinition, being the
product of the two defining constants k = 1.380649e-23 J K^-1 and
N_A = 6.02214076e23 mol^-1, and it is reproduced by that multiplication in this
module's tests rather than taken on the authority of the citation.
"""

from __future__ import annotations

import datetime as _dt
import enum
import math
import warnings
from collections.abc import Sequence
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import Feedstock
from ae.core.provenance import MISSING, Source, Tag, Tier, Value, _Missing
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = [
    "EA_REGIME_BANDS",
    "GAS_CONSTANT",
    "SOURCE_XIA_2024",
    "SOURCE_YANG_2020",
    "Arrhenius",
    "LeachSystem",
    "Regime",
    "RegimeFit",
    "arrhenius_fit",
    "conversion",
    "conversion_over_size_distribution",
    "conversion_profile",
    "g_of_conversion",
    "identify_regime",
    "leachable_ppm",
    "removal_fraction_from_assay",
    "size_exponent_from_series",
    "tau_film",
    "tau_for",
    "tau_from_single_point",
    "tau_product_layer",
    "tau_surface_reaction",
]

#: Molar gas constant, CODATA 2018 exact value. J mol^-1 K^-1.
GAS_CONSTANT: Final[Quantity] = Q_(8.314462618, "J/(mol*K)")

SOURCE_YANG_2020: Final[Source] = Source(
    citation="Yang, C. and Li, S. (2020) Kinetics of iron removal from quartz under "
             "ultrasound-assisted leaching, High Temperature Materials and Processes "
             "39(1), 395-404",
    tier=Tier.T1,
    doi="10.1515/htmp-2020-0081",
    accessed=_dt.date(2026, 9, 16),
    extraction="manual",
    note="Abstract only. Publisher full text (degruyter.com) is unreachable from this "
         "sandbox, so the per-point conversion-time table could not be extracted.",
)

SOURCE_XIA_2024: Final[Source] = Source(
    citation="Xia, M., Yang, X. and Hou, Z. (2024) Preparation of High-Purity Quartz Sand "
             "by Vein Quartz Purification and Characteristics: A Case Study of Pakistan "
             "Vein Quartz, Minerals 14(7), 727",
    tier=Tier.T1,
    doi="10.3390/min14070727",
    accessed=_dt.date(2026, 9, 16),
    extraction="manual",
)


class Regime(str, enum.Enum):
    """Rate-controlling step of the shrinking-core model."""

    FILM = "film"                        # external fluid-film mass transfer
    PRODUCT_LAYER = "product_layer"      # diffusion through the reacted residue
    SURFACE_REACTION = "surface_reaction"  # chemical reaction at the core surface
    MIXED = "mixed"                      # no single regime dominates the fit


#: Apparent activation energy bands used only as CORROBORATION in regime
#: identification. Values in kJ mol^-1 as (low, high).
#:
#: This banding is an engineering heuristic, not a measured constant, and it is
#: recorded as ASSUMED with its basis. Film control tracks the temperature
#: dependence of liquid viscosity and diffusivity in the boundary layer (low
#: single-digit to ~20 kJ mol^-1); pore diffusion adds tortuosity and the
#: temperature dependence of the bulk diffusivity (~20 to 40); a surface
#: chemical step carries a true bond-breaking barrier (>40). The one quartz
#: datapoint available here, 27.72 kJ mol^-1 assigned to product-layer diffusion
#: by Yang and Li (2020), falls in the middle band, which is weak corroboration
#: from a single system and is not treated as validation.
EA_REGIME_BANDS: Final[dict[Regime, Value]] = {
    Regime.FILM: Value(
        quantity=Q_(np.array([0.0, 20.0]), "kJ/mol"),
        tag=Tag.ASSUMED,
        basis="Engineering heuristic, not a measurement: film control is limited by "
              "boundary-layer transport, whose temperature dependence follows liquid "
              "viscosity and molecular diffusivity, giving apparent Ea below about "
              "20 kJ/mol. Estimate; used only to corroborate a regression, never as "
              "primary evidence.",
    ),
    Regime.PRODUCT_LAYER: Value(
        quantity=Q_(np.array([20.0, 40.0]), "kJ/mol"),
        tag=Tag.ASSUMED,
        basis="Engineering heuristic. Bracket chosen to contain the single quartz "
              "datapoint retrievable here (27.72 kJ/mol, Yang and Li 2020, assigned by "
              "those authors to product-layer diffusion). One system is not a "
              "calibration; this is an estimate.",
    ),
    Regime.SURFACE_REACTION: Value(
        quantity=Q_(np.array([40.0, 300.0]), "kJ/mol"),
        tag=Tag.ASSUMED,
        basis="Engineering heuristic: a rate-controlling chemical step carries a "
              "bond-breaking barrier, conventionally taken as above 40 kJ/mol. Upper "
              "bound 300 kJ/mol is a physical sanity ceiling for an aqueous leach, not "
              "a measurement. Estimate.",
    ),
}


class Arrhenius(BaseModel):
    """Arrhenius temperature dependence of a rate constant or a diffusivity.

    ``k(T) = A exp(-Ea / (R T))``. ``prefactor`` carries the unit of the target
    quantity (m/s for a mass-transfer or surface rate constant, m^2/s for a
    diffusivity), so dimensional correctness of the tau formulae is enforced by
    the unit system rather than by convention.

    ``calibration_T`` records the temperature interval over which the pair was
    measured or fitted. :meth:`at` warns outside it, because an Arrhenius
    extrapolation is the single most common way a leach model produces a
    physically impossible answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    prefactor: Value
    activation_energy: Value
    calibration_T: tuple[float, float] | None = Field(
        default=None, description="Calibration interval in K as (low, high)")

    @model_validator(mode="after")
    def _physically_possible(self) -> Arrhenius:
        a = self.prefactor.quantity
        ea = self.activation_energy.quantity.to("J/mol")
        if float(a.magnitude) <= 0.0:
            raise ValueError(
                f"Arrhenius prefactor must be positive, got {a}. A non-positive "
                f"prefactor gives a non-positive rate, which violates the second law "
                f"for a spontaneous dissolution."
            )
        if float(ea.magnitude) < 0.0:
            raise ValueError(
                f"activation energy must be non-negative, got {ea}. A negative Ea "
                f"means the rate falls with temperature, which for an elementary step "
                f"is unphysical; an apparent negative Ea indicates a change of "
                f"mechanism and must be modelled as two regimes, not one."
            )
        if self.calibration_T is not None:
            lo, hi = self.calibration_T
            if not (0.0 < lo <= hi):
                raise ValueError(
                    f"calibration_T must satisfy 0 < low <= high, got {self.calibration_T}")
        return self

    def at(self, temperature: Quantity) -> Quantity:
        """Evaluate ``k(T)``. Warns if ``T`` is outside the calibration interval."""
        require_dimensionality(temperature, "temperature", "temperature")
        t_k = float(temperature.to("K").magnitude)
        if t_k <= 0.0:
            raise ValueError(f"absolute temperature must be positive, got {t_k} K")
        if self.calibration_T is not None:
            lo, hi = self.calibration_T
            if not (lo <= t_k <= hi):
                warnings.warn(
                    f"evaluating Arrhenius at {t_k:.1f} K, outside its calibration "
                    f"interval [{lo:.1f}, {hi:.1f}] K; the extrapolation is not "
                    f"supported by the data the pair was fitted to",
                    RuntimeWarning, stacklevel=2,
                )
        ea = self.activation_energy.quantity.to("J/mol")
        expo = float((-ea / (GAS_CONSTANT * temperature.to("K"))).to("dimensionless").magnitude)
        k = self.prefactor.quantity * math.exp(expo)
        assert float(k.magnitude) > 0.0, "Arrhenius rate must be strictly positive"
        return k


class LeachSystem(BaseModel):
    """One (feedstock, element, reagent, particle size) leach configuration.

    Every kinetic quantity in this module is computed from a ``LeachSystem``, so
    the ore enters through :attr:`feedstock` and nothing is hardcoded to a
    deposit. ``solid_molar_density`` is the molar density of the REACTIVE SOLID
    PHASE being dissolved (for example hematite in a fracture film), not of the
    quartz matrix, because the shrinking core is the impurity phase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    feedstock: Feedstock
    element: str
    reagent: str
    particle_radius: Value
    reagent_concentration: Value
    solid_molar_density: Value
    stoich_b: float = Field(gt=0.0, description="moles of solid B per mole of reagent A")
    temperature: Value
    film_coefficient: Arrhenius | _Missing = MISSING
    product_layer_diffusivity: Arrhenius | _Missing = MISSING
    surface_rate_constant: Arrhenius | _Missing = MISSING

    @model_validator(mode="after")
    def _dimensions_and_ranges(self) -> LeachSystem:
        require_dimensionality(self.particle_radius.quantity, "length", "particle_radius")
        require_dimensionality(self.temperature.quantity, "temperature", "temperature")
        r = float(self.particle_radius.quantity.to("m").magnitude)
        if not (1e-7 <= r <= 1e-1):
            raise ValueError(
                f"particle_radius {r} m is outside the model's stated valid range "
                f"1e-7 to 1e-1 m; below 0.1 um the continuum shrinking-core picture "
                f"fails and above 100 mm the isothermal-particle assumption fails"
            )
        for field, unit in (("reagent_concentration", "mol/m**3"),
                            ("solid_molar_density", "mol/m**3")):
            q = getattr(self, field).quantity.to(unit)
            if float(q.magnitude) <= 0.0:
                raise ValueError(f"{field} must be positive, got {q}")
        if self.element not in self.feedstock.impurities.total and self.element != "Si":
            raise ValueError(
                f"{self.element} has no measured total concentration in feedstock "
                f"{self.feedstock.sample_id}; a leach model cannot be run against an "
                f"element that was never assayed"
            )
        return self

    def rate_constant(self, regime: Regime) -> Quantity:
        """Temperature-evaluated transport or rate coefficient for ``regime``."""
        attr = {
            Regime.FILM: self.film_coefficient,
            Regime.PRODUCT_LAYER: self.product_layer_diffusivity,
            Regime.SURFACE_REACTION: self.surface_rate_constant,
        }.get(regime)
        if attr is None:
            raise ValueError(f"{regime} has no associated coefficient; MIXED is not a rate law")
        if isinstance(attr, _Missing):
            raise KeyError(
                f"{regime.value} coefficient was not supplied for element "
                f"{self.element} on feedstock {self.feedstock.sample_id}; supply an "
                f"Arrhenius pair or choose a regime whose coefficient is known"
            )
        return attr.at(self.temperature.quantity)


class RegimeFit(BaseModel):
    """Outcome of fitting all three shrinking-core linearisations to one dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    best: Regime
    r_squared: dict[str, float]
    tau_s: dict[str, float]
    margin: float = Field(description="R-squared of the best minus that of the runner-up")
    n_points: int
    apparent_Ea_kJ_per_mol: float | None = None
    Ea_band_consistent: bool | None = None
    note: str = ""


def leachable_ppm(feedstock: Feedstock, element: str) -> float:
    """Concentration of ``element`` that an aqueous leach could in principle reach.

    Returns ``total_ppm - lattice_ppm``. Lattice-substituted Al, Ti, Li and B sit
    on silicon or interstitial sites inside the quartz structure and no reagent
    reaches them at leaching temperatures, so they are subtracted before any
    kinetic calculation.

    Raises
    ------
    ae.core.provenance.MissingValueError
        If the lattice split was never measured, propagated unchanged from
        :meth:`ae.core.feedstock.ImpurityProfile.lattice_ppm`. This is correct
        behaviour: the leachable inventory is the difference of two numbers and
        substituting a guess for one of them fabricates the purity ceiling of the
        deposit. Do not add a default.
    """
    total = feedstock.impurities.total_ppm(element)
    lattice = feedstock.impurities.lattice_ppm(element)
    leachable = total - lattice
    assert leachable >= -1e-9, (
        f"lattice_ppm ({lattice}) exceeds total_ppm ({total}) for {element}: a "
        f"sub-population cannot exceed the bulk, so the profile is inconsistent"
    )
    return max(0.0, leachable)


def tau_film(system: LeachSystem) -> Quantity:
    r"""Time for complete conversion under fluid-film control.

    :math:`\tau_f = \rho_B R / (3 b k_g C_A)`.
    """
    rho = system.solid_molar_density.quantity.to("mol/m**3")
    radius = system.particle_radius.quantity.to("m")
    conc = system.reagent_concentration.quantity.to("mol/m**3")
    kg = system.rate_constant(Regime.FILM).to("m/s")
    tau = rho * radius / (3.0 * system.stoich_b * kg * conc)
    return _checked_tau(tau, "tau_film")


def tau_product_layer(system: LeachSystem) -> Quantity:
    r"""Time for complete conversion under product-layer diffusion control.

    :math:`\tau_{pl} = \rho_B R^2 / (6 b D_e C_A)`.
    """
    rho = system.solid_molar_density.quantity.to("mol/m**3")
    radius = system.particle_radius.quantity.to("m")
    conc = system.reagent_concentration.quantity.to("mol/m**3")
    de = system.rate_constant(Regime.PRODUCT_LAYER).to("m**2/s")
    tau = rho * radius**2 / (6.0 * system.stoich_b * de * conc)
    return _checked_tau(tau, "tau_product_layer")


def tau_surface_reaction(system: LeachSystem) -> Quantity:
    r"""Time for complete conversion under surface-reaction control.

    :math:`\tau_{sr} = \rho_B R / (b k_s C_A)`.
    """
    rho = system.solid_molar_density.quantity.to("mol/m**3")
    radius = system.particle_radius.quantity.to("m")
    conc = system.reagent_concentration.quantity.to("mol/m**3")
    ks = system.rate_constant(Regime.SURFACE_REACTION).to("m/s")
    tau = rho * radius / (system.stoich_b * ks * conc)
    return _checked_tau(tau, "tau_surface_reaction")


def _checked_tau(tau: Quantity, name: str) -> Quantity:
    """Enforce that a completion time is a positive, finite duration in seconds."""
    tau_s = tau.to("s")
    val = float(tau_s.magnitude)
    assert math.isfinite(val) and val > 0.0, (
        f"{name} must be a positive finite time, got {val} s; a non-positive tau "
        f"implies instantaneous or reversed conversion"
    )
    require_dimensionality(tau_s, "time", name)
    return tau_s


def tau_for(system: LeachSystem, regime: Regime) -> Quantity:
    """Dispatch to the tau expression for ``regime``."""
    if regime is Regime.FILM:
        return tau_film(system)
    if regime is Regime.PRODUCT_LAYER:
        return tau_product_layer(system)
    if regime is Regime.SURFACE_REACTION:
        return tau_surface_reaction(system)
    raise ValueError(f"{regime} is not a single rate law and has no tau")


def g_of_conversion(regime: Regime, conversion_x: float | np.ndarray) -> np.ndarray:
    r"""Linearising function :math:`g(X) = t/\tau` for each regime.

    ``FILM``: :math:`X`. ``PRODUCT_LAYER``: :math:`1 - 3(1-X)^{2/3} + 2(1-X)`.
    ``SURFACE_REACTION``: :math:`1 - (1-X)^{1/3}`. All three map [0, 1] onto
    [0, 1] monotonically, which is what makes a regression through the origin the
    right diagnostic.
    """
    x = np.atleast_1d(np.asarray(conversion_x, dtype=float))
    if np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
        raise ValueError(f"conversion must lie in [0, 1], got range [{x.min()}, {x.max()}]")
    x = np.clip(x, 0.0, 1.0)
    rest = 1.0 - x
    if regime is Regime.FILM:
        return x
    if regime is Regime.PRODUCT_LAYER:
        return 1.0 - 3.0 * rest ** (2.0 / 3.0) + 2.0 * rest
    if regime is Regime.SURFACE_REACTION:
        return 1.0 - rest ** (1.0 / 3.0)
    raise ValueError(f"{regime} has no linearising function")


def conversion(regime: Regime, t: Quantity, tau: Quantity) -> float:
    r"""Closed-form conversion :math:`X(t)` for one regime.

    Uses the exact trigonometric inversion of the product-layer cubic derived in
    the module docstring, so no iteration is involved.
    """
    require_dimensionality(t, "time", "t")
    require_dimensionality(tau, "time", "tau")
    theta = float((t.to("s") / tau.to("s")).to("dimensionless").magnitude)
    if theta < 0.0:
        raise ValueError(f"time must be non-negative, got theta = {theta}")
    if theta >= 1.0:
        return 1.0
    if regime is Regime.FILM:
        x = theta
    elif regime is Regime.SURFACE_REACTION:
        x = 1.0 - (1.0 - theta) ** 3
    elif regime is Regime.PRODUCT_LAYER:
        phi = math.acos(max(-1.0, min(1.0, 2.0 * theta - 1.0)))
        u = 0.5 + math.cos(phi / 3.0 + 4.0 * math.pi / 3.0)
        u = max(0.0, min(1.0, u))
        x = 1.0 - u**3
    else:
        raise ValueError(f"{regime} is not a single rate law")
    return require_fraction(x, f"conversion under {regime.value}")


def conversion_profile(
    system: LeachSystem, regime: Regime, times: Quantity
) -> np.ndarray:
    """Conversion at each time in ``times`` for ``system`` under ``regime``."""
    tau = tau_for(system, regime)
    out = np.array([conversion(regime, Q_(float(t), str(times.units)), tau)
                    for t in np.atleast_1d(times.magnitude)], dtype=float)
    assert np.all(np.diff(out) >= -1e-12) or np.any(np.diff(np.atleast_1d(times.magnitude)) < 0), (
        "conversion must be non-decreasing in time for monotonically increasing times"
    )
    return out


def tau_from_single_point(regime: Regime, conversion_x: float, t: Quantity) -> Quantity:
    r"""Back out :math:`\tau` from one measured (X, t) pair: :math:`\tau = t / g(X)`.

    One point cannot distinguish regimes; it only scales an assumed regime. Used
    for scenario work when a paper reports a single endpoint (for example
    74 percent Fe removal in 40 minutes) and nothing else.
    """
    require_fraction(conversion_x, "conversion_x")
    if conversion_x <= 0.0:
        raise ValueError("cannot infer tau from zero conversion")
    if conversion_x >= 1.0:
        raise ValueError(
            "cannot infer tau from complete conversion: g(1) = 1 places only a lower "
            "bound on the rate, since the endpoint may have been reached earlier"
        )
    g = float(g_of_conversion(regime, conversion_x)[0])
    return _checked_tau(t.to("s") / g, "tau_from_single_point")


def identify_regime(
    times: Quantity,
    conversions: Sequence[float] | np.ndarray,
    apparent_Ea: Quantity | None = None,
    margin: float = 0.02,
) -> RegimeFit:
    r"""Fit all three linearisations to (t, X) data and report which dominates.

    Each regime predicts :math:`g(X) = t/\tau`, a straight line through the
    origin with slope :math:`1/\tau`. The fit is therefore a one-parameter
    least-squares regression of :math:`g(X)` on :math:`t` with zero intercept,
    :math:`1/\tau = \sum g t / \sum t^2`, and the goodness of fit is

    .. math:: R^2 = 1 - \frac{\sum (g_i - t_i/\tau)^2}{\sum (g_i - \bar{g})^2}

    Forcing the intercept to zero is deliberate: a fitted intercept absorbs the
    induction period and makes all three regimes look equally good, which is the
    classic way this diagnostic is abused.

    Returns ``Regime.MIXED`` when the best R-squared does not exceed the
    runner-up by ``margin``, because three monotone functions on [0, 1] are
    similar enough that a small margin is not evidence.

    Parameters
    ----------
    times
        Sampling times, a pint Quantity array with time dimensionality.
    conversions
        Measured conversion at each time, each in [0, 1].
    apparent_Ea
        Optional independently measured activation energy. If given, the fit
        reports whether it falls in :data:`EA_REGIME_BANDS` for the winning
        regime. That is corroboration only: the bands are ASSUMED heuristics.
    margin
        Minimum R-squared advantage required to name a single regime.
    """
    require_dimensionality(times, "time", "times")
    t = np.asarray(np.atleast_1d(times.to("s").magnitude), dtype=float)
    x = np.asarray(conversions, dtype=float)
    if t.shape != x.shape:
        raise ValueError(
            f"times and conversions must have the same shape, got {t.shape} and {x.shape}")
    if t.size < 4:
        raise ValueError(
            f"regime identification needs at least 4 points to be meaningful, got {t.size}; "
            f"with fewer, any of the three regimes fits within noise"
        )
    if np.any(t < 0.0):
        raise ValueError("times must be non-negative")
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError(f"conversions must lie in [0, 1], got [{x.min()}, {x.max()}]")

    r2: dict[str, float] = {}
    taus: dict[str, float] = {}
    for regime in (Regime.FILM, Regime.PRODUCT_LAYER, Regime.SURFACE_REACTION):
        g = np.asarray(g_of_conversion(regime, x), dtype=float)
        denom = float(np.sum(t * t))
        if denom <= 0.0:
            raise ValueError("all times are zero; cannot regress")
        slope = float(np.sum(g * t) / denom)      # slope = 1/tau
        resid = g - slope * t
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((g - g.mean()) ** 2))
        r2[regime.value] = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        taus[regime.value] = 1.0 / slope if slope > 0.0 else float("inf")

    order = sorted(r2, key=lambda k: r2[k], reverse=True)
    gap = r2[order[0]] - r2[order[1]]
    best = Regime(order[0]) if gap >= margin else Regime.MIXED

    band_ok: bool | None = None
    ea_val: float | None = None
    if apparent_Ea is not None:
        require_dimensionality(apparent_Ea, "molar_energy", "apparent_Ea")
        ea_val = float(apparent_Ea.to("kJ/mol").magnitude)
        if best is not Regime.MIXED:
            lo, hi = (float(v) for v in EA_REGIME_BANDS[best].quantity.to("kJ/mol").magnitude)
            band_ok = lo <= ea_val <= hi

    note = (
        f"best fit {best.value} with R2 margin {gap:.4f} over {order[1]}. "
        f"R-squared alone cannot separate these three monotone functions when the "
        f"data span a narrow conversion range; a particle-size series or a "
        f"stirring-speed series is required for a defensible assignment."
    )
    if best is Regime.MIXED:
        note = (
            f"no single regime clears the {margin} R-squared margin (best {order[0]} at "
            f"{r2[order[0]]:.4f}, runner-up {order[1]} at {r2[order[1]]:.4f}); "
            f"mixed control or an induction period is likely."
        )
    return RegimeFit(
        best=best, r_squared=r2, tau_s=taus, margin=gap, n_points=int(t.size),
        apparent_Ea_kJ_per_mol=ea_val, Ea_band_consistent=band_ok, note=note,
    )


def arrhenius_fit(
    temperatures: Quantity, rate_constants: Quantity
) -> tuple[Quantity, Quantity, float]:
    r"""Least-squares Arrhenius fit: returns ``(Ea, prefactor, r_squared)``.

    Regresses :math:`\ln k` on :math:`1/T`, slope :math:`-E_a/\mathcal{R}`,
    intercept :math:`\ln A`. Requires at least three temperatures: two points
    define a line exactly and report R-squared of 1 regardless of the data
    quality, which is misleading.
    """
    require_dimensionality(temperatures, "temperature", "temperatures")
    t_k = np.asarray(np.atleast_1d(temperatures.to("K").magnitude), dtype=float)
    k = np.asarray(np.atleast_1d(rate_constants.magnitude), dtype=float)
    if t_k.shape != k.shape:
        raise ValueError(f"shape mismatch: {t_k.shape} vs {k.shape}")
    if t_k.size < 3:
        raise ValueError(
            f"an Arrhenius fit needs at least 3 temperatures, got {t_k.size}; two "
            f"points give R-squared of exactly 1 whatever the scatter"
        )
    if np.any(t_k <= 0.0):
        raise ValueError("absolute temperatures must be positive")
    if np.any(k <= 0.0):
        raise ValueError(
            f"rate constants must be positive (got min {k.min()}); a non-positive rate "
            f"constant cannot be fitted on a log scale and is unphysical"
        )
    inv_t = 1.0 / t_k
    ln_k = np.log(k)
    slope, intercept = np.polyfit(inv_t, ln_k, 1)
    pred = slope * inv_t + intercept
    ss_res = float(np.sum((ln_k - pred) ** 2))
    ss_tot = float(np.sum((ln_k - ln_k.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    # slope has units of K (regression of ln k on 1/T), so Ea = -slope * R * K.
    ea = -float(slope) * GAS_CONSTANT * Q_(1.0, "K")
    if float(ea.to("J/mol").magnitude) < 0.0:
        warnings.warn(
            f"fitted activation energy is negative ({ea.to('kJ/mol'):~P}), so the rate "
            f"falls with temperature; this signals a mechanism change or a systematic "
            f"error, not a valid single-step Arrhenius pair",
            RuntimeWarning, stacklevel=2,
        )
    prefactor = Q_(float(np.exp(intercept)), str(rate_constants.units))
    return ea.to("kJ/mol"), prefactor, r_squared


def size_exponent_from_series(
    radii: Quantity, taus: Quantity
) -> tuple[float, float]:
    r"""Fit :math:`\tau \propto R^n` and return ``(n, r_squared)``.

    The exponent discriminates regimes independently of any R-squared comparison
    on a single conversion curve: :math:`n = 1` for film and surface-reaction
    control, :math:`n = 2` for product-layer diffusion. It cannot separate film
    from surface reaction, which needs a stirring-speed series instead.
    """
    require_dimensionality(radii, "length", "radii")
    require_dimensionality(taus, "time", "taus")
    r = np.asarray(np.atleast_1d(radii.to("m").magnitude), dtype=float)
    tau = np.asarray(np.atleast_1d(taus.to("s").magnitude), dtype=float)
    if r.shape != tau.shape:
        raise ValueError(f"shape mismatch: {r.shape} vs {tau.shape}")
    if r.size < 3:
        raise ValueError("need at least 3 sizes to fit an exponent with a goodness of fit")
    if np.any(r <= 0.0) or np.any(tau <= 0.0):
        raise ValueError("radii and taus must be positive")
    slope, intercept = np.polyfit(np.log(r), np.log(tau), 1)
    pred = slope * np.log(r) + intercept
    ss_res = float(np.sum((np.log(tau) - pred) ** 2))
    ss_tot = float(np.sum((np.log(tau) - np.log(tau).mean()) ** 2))
    return float(slope), (1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan"))


def conversion_over_size_distribution(
    system: LeachSystem,
    regime: Regime,
    t: Quantity,
    radii: Quantity,
    mass_fractions: Sequence[float] | np.ndarray,
) -> float:
    r"""Mass-weighted conversion across a particle size distribution.

    .. math:: \bar{X}(t) = \sum_i w_i X\!\left(t; \tau(R_i)\right)

    with :math:`\sum w_i = 1`. Because :math:`\tau` grows with :math:`R`, the
    coarse tail controls the time to high overall conversion, and a single-size
    calculation at the mean radius systematically over-predicts.
    """
    require_dimensionality(radii, "length", "radii")
    w = np.asarray(mass_fractions, dtype=float)
    r = np.atleast_1d(radii.to("m").magnitude)
    if w.shape != r.shape:
        raise ValueError(f"mass_fractions and radii must align, got {w.shape} and {r.shape}")
    if np.any(w < 0.0):
        raise ValueError("mass fractions must be non-negative")
    total = float(w.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            f"mass fractions must sum to 1 for the mass balance to close, got {total}"
        )
    x_bar = 0.0
    for wi, ri in zip(w, r):
        sub = system.model_copy(update={
            "particle_radius": Value(
                quantity=Q_(float(ri), "m"), tag=Tag.DERIVED,
                basis="one bin of a supplied particle size distribution",
            )
        })
        x_bar += float(wi) * conversion(regime, t, tau_for(sub, regime))
    return require_fraction(x_bar, "mass-weighted conversion")


def removal_fraction_from_assay(feed_ppm: float, product_ppm: float) -> float:
    r"""Impurity removal fraction from a feed and a product assay.

    .. math:: \eta = \frac{C_{feed} - C_{product}}{C_{feed}}

    This is the accounting identity that published purification results report,
    and it is separated from the kinetics deliberately: it is the only part of
    this module that can be validated against the literature available here.
    Assumes negligible silica dissolution, so that concentrations are on a
    common mass basis. If matrix loss is significant, the mass-corrected form in
    :mod:`ae.physics.reagents` must be used instead.
    """
    if feed_ppm <= 0.0:
        raise ValueError(f"feed concentration must be positive, got {feed_ppm}")
    if product_ppm < 0.0:
        raise ValueError(
            f"product concentration cannot be negative, got {product_ppm}; a negative "
            f"concentration is unphysical"
        )
    if product_ppm > feed_ppm:
        raise ValueError(
            f"product ({product_ppm}) exceeds feed ({feed_ppm}); leaching cannot add "
            f"impurity, so either contamination occurred or the assays are on "
            f"different mass bases"
        )
    return require_fraction((feed_ppm - product_ppm) / feed_ppm, "removal fraction")
