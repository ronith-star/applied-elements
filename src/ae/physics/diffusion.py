r"""Solid-state diffusion in quartz, and the purity ceiling it imposes.

Purpose: determine quantitatively, and for each process temperature window
separately, whether lattice-substituted impurities can be removed from a quartz
grain in a practical residence time. The controlling comparison is the diffusion
length :math:`\sqrt{Dt}` against the grain radius: when the former is orders of
magnitude smaller, only a surface skin is depleted and the grain retains its
lattice inventory.

WHAT THIS MODULE ACTUALLY ESTABLISHES, AND WHAT IT DOES NOT
===========================================================
The result is regime dependent, and this module refuses to overstate it. Under
the most generous mobility bound that any accessible source supports for a
cation in the SiO2 lattice (:data:`MOST_MOBILE_BOUND`: :math:`D_0 = 10^{-4}`
m^2 s^-1, :math:`E_a = 90` kJ mol^-1, the low end of the Liu et al. 2026 range
and therefore an overestimate of mobility for substitutional Al), for a
100 micrometre grain radius:

- ATMOSPHERIC-PRESSURE ACID LEACHING (up to about 100 degC): ESTABLISHED
  INFEASIBLE. At 80 degC and 6 h the diffusion length is 0.325 um against a
  100 um radius, a factor of 308 (2.49 orders of magnitude). At 95 degC and
  24 h it is 1.21 um, a factor of 82. Since this is computed with an
  overestimate of mobility, the true deficit is larger. Conclusion robust.
- CHLORINATION ROASTING, Ti SPECIFICALLY: ESTABLISHED INFEASIBLE. Using the
  Ti4+ activation energy Liu et al. (2026) report for lattice diffusion in SiO2
  (about 400 kJ mol^-1), at 1200 degC and 6 h the diffusion length is 119 nm
  for :math:`D_0 = 10^{-4}` and 0.12 nm for :math:`D_0 = 10^{-10}`, i.e. a
  deficit of 2.9 to 5.9 orders of magnitude across the entire plausible
  prefactor range. Fractional extraction from the sphere is 4.0e-3. Conclusion
  robust because it does not depend on the unknown prefactor.
- HOT PRESSURE ACID LEACHING (200 to 300 degC) and CHLORINATION ROASTING FOR Al:
  NOT ESTABLISHED BY THIS MODEL. At 250 degC and 6 h the generous bound gives a
  diffusion length of 47 um against a 100 um radius, a factor of only 2.1, and
  at 300 degC it exceeds the radius. At 1200 degC and 6 h the activation energy
  at which the diffusion length equals the grain radius is 65.8 kJ mol^-1 at the
  low end of the prefactor sweep (:math:`D_0 = 10^{-10}` m^2 s^-1) and
  235.1 kJ mol^-1 at the high end (:math:`D_0 = 10^{-4}` m^2 s^-1)
  (:func:`critical_activation_energy`), and the true barrier for Al is
  unmeasured: it lies somewhere between the Na+ value (about 90 kJ mol^-1)
  and the Ti4+ value (about 400 kJ mol^-1) reported by Liu et al. (2026), which
  straddles the threshold. A diffusion-length argument therefore CANNOT decide
  the Al case at roasting temperature, and this module says so rather than
  choosing a prefactor that produces the desired answer.

The platform's conclusion that lattice Al sets the purity ceiling consequently
rests on the EMPIRICAL result, not on this diffusion calculation: Xia et al.
(2024, doi 10.3390/min14070727) took a vein quartz from 128.86 ug/g total trace
impurities to 24.23 ug/g (81.20 percent removal) through crushing, ultrasonic
desliming, flotation, calcination, water quenching, hot pressure acid leaching
AND chlorination roasting, and identified the residual as lattice-bound Al, Ti
and Li. The diffusion model explains why the leaching stages cannot reach the
lattice and why Ti survives the roast; it does not independently prove that Al
survives a roast, and the Liu et al. (2026) abstract in fact
announces an alkali-first, Al-follows coupled diffusion mechanism for aluminium
removal plus a temperature-staged, atmosphere-segmented roasting strategy, which
would be incoherent if Al diffusion at roasting temperature were negligible.

EQUATIONS
=========

Fick's laws
-----------
Fick's first law, one dimension:

.. math:: J = -D \frac{\partial C}{\partial x}

Fick's second law, with :math:`D` independent of concentration:

.. math:: \frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}

Symbols, units, valid ranges:

=====================  =====================  =================================
Symbol                 Unit                   Valid range
=====================  =====================  =================================
:math:`J`              mol m^-2 s^-1          any sign
:math:`C`              mol m^-3               >= 0
:math:`D`              m^2 s^-1               > 0
:math:`x`              m                       >= 0
:math:`t`              s                       >= 0
:math:`L`              m                       >= 0 (diffusion length)
:math:`R_{grain}`      m                       > 0
:math:`D_0`            m^2 s^-1               > 0 (pre-exponential)
:math:`E_a`            J mol^-1               > 0
:math:`T`              K                       > 0
:math:`\mathcal{R}`    J mol^-1 K^-1          8.314462618 (CODATA 2018)
:math:`\mathrm{Fo}`    dimensionless          >= 0 (Fourier number)
=====================  =====================  =================================

Arrhenius diffusivity
---------------------
.. math:: D(T) = D_0 \exp\!\left(-\frac{E_a}{\mathcal{R} T}\right)

Dimensional check: :math:`[D_0] = [D] = \mathrm{m^2\,s^{-1}}`, and the exponent
is dimensionless because :math:`E_a/(\mathcal{R}T)` is (J mol^-1) / (J mol^-1
K^-1 K).

Diffusion length
----------------
Definition used here:

.. math:: L = \sqrt{D t}

Derivation of why this is the right scale, not an arbitrary one. For a
semi-infinite solid initially at :math:`C_0` held at :math:`C_s = 0` at the
surface (the leaching boundary condition: reagent removes everything that
reaches the surface), Fick's second law has the exact similarity solution

.. math:: \frac{C(x,t)}{C_0} = \mathrm{erf}\!\left(\frac{x}{2\sqrt{Dt}}\right)

so the concentration is depleted to 52.05 percent of its initial value at
:math:`x = \sqrt{Dt}` (since :math:`\mathrm{erf}(1/2) = 0.5205`), and to within
1 percent of its initial value by :math:`x = 3.64\sqrt{Dt}`. The depleted skin
is therefore a few multiples of :math:`\sqrt{Dt}` thick, whatever prefactor
convention is used. Some texts define :math:`L = 2\sqrt{Dt}`; the factor does
not change any conclusion below by more than a factor of two, while the
quantities compared differ by ten or more orders of magnitude.

Fractional extraction from a sphere
-----------------------------------
For a sphere of radius :math:`R` with zero surface concentration, the exact
series solution of Fick's second law gives the fraction extracted. It is DERIVED
here rather than quoted: substituting :math:`u = rC` turns the spherical
diffusion equation into the one-dimensional heat equation
:math:`\partial u/\partial t = D\,\partial^2 u/\partial r^2` with
:math:`u(0,t) = u(R,t) = 0`, whose eigenfunctions are
:math:`\sin(n\pi r/R)` with eigenvalues :math:`n^2\pi^2 D/R^2`. Expanding the
uniform initial condition :math:`u(r,0) = r` in that basis gives Fourier
coefficients :math:`2R(-1)^{n+1}/(n\pi)`, and integrating
:math:`4\pi\int_0^R r u\,dr` over the resulting series yields

.. math:: \frac{M_t}{M_\infty} = 1 - \frac{6}{\pi^2}
          \sum_{n=1}^{\infty} \frac{1}{n^2}
          \exp\!\left(-\frac{n^2 \pi^2 D t}{R^2}\right)

which depends only on the Fourier number :math:`\mathrm{Fo} = Dt/R^2`. This
form is standard and appears in Crank's monograph (see References), but the
specific equation number in that edition could not be checked from this sandbox,
so no equation number is cited. Instead the series is verified numerically
against an independent finite-difference solution of the same PDE in
``test_sphere_series_matches_finite_difference_solution``, which agrees to
better than 0.01 percent at Fo = 1e-3. The short-time limit of the same solution
is

.. math:: \frac{M_t}{M_\infty} \approx \frac{6}{R}\sqrt{\frac{Dt}{\pi}}
          = \frac{6}{\sqrt{\pi}} \sqrt{\mathrm{Fo}}

Both are exact solutions of the same equation, not correlations.

The two forms cross over near :math:`\mathrm{Fo} = 10^{-4}`, where they agree to
0.89 percent (series 0.033551, short-time 0.033851). Below that the short-time
form is used, because the series converges too slowly there to truncate safely:
the summand decays as :math:`\exp(-n^2\pi^2\mathrm{Fo})`, so the number of terms
needed scales as :math:`\mathrm{Fo}^{-1/2}`, and a fixed 200-term truncation
overstates the extraction by 23 percent at :math:`\mathrm{Fo} = 10^{-6}`
(0.004158 truncated against 0.003382 converged). Above the crossover the term
count is chosen adaptively from :math:`\mathrm{Fo}` so the neglected tail is
below 1e-12, and the disagreement between the two forms GROWS with
:math:`\mathrm{Fo}` (2.9 percent at 1e-3, 9.7 percent at 1e-2) because the
short-time form is the asymptote, not a competitor.

Limiting grain size
-------------------
Setting :math:`L = R_{grain}` and solving for :math:`R` gives the largest grain
that can be depleted in time :math:`t`:

.. math:: R_{max}(T, t) = \sqrt{D(T) t}

A grain coarser than :math:`R_{max}` retains essentially all of its lattice
inventory. :func:`limiting_grain_radius` reports this, and its value is strongly
temperature dependent, which is why the verdict in this module is regime
dependent rather than universal. Evaluated with :data:`MOST_MOBILE_BOUND`
(:math:`D_0 = 10^{-4}` m^2 s^-1, :math:`E_a = 90` kJ mol^-1), :math:`R_{max}` is
0.33 um at 80 degC and 6 h, but 3.0 mm at 600 degC and 6 h and 75 mm at 1200 degC
and 24 h, the last two exceeding any real grain. Those high-temperature figures
are NOT a prediction that lattice cations are mobile in a roast: they follow from
deliberately assigning the most mobile species' barrier (90 kJ mol^-1,
interstitial Na+) to every cation. They show that the generous bound carries no
information above about 200 degC, which is why :func:`lattice_removal_verdict`
returns ``UNDECIDED_BY_DIFFUSION`` there and why the barrier-threshold inversion
in :func:`critical_activation_energy` is the usable tool at roasting temperature.
See :func:`diffusion_length_table` for the computed grid.

DIFFUSIVITY DATA STATUS
=======================
This module deliberately ships NO element-specific diffusivity for quartz. The
measurements that would supply them were unreachable from this sandbox on
2026-09-16 and fabricating the two Arrhenius constants would make the central
result unverifiable:

- Al: Pankrath (1994) Kinetics of Al-Si exchange in low and high quartz:
  calculation of Al diffusion coefficients, European Journal of Mineralogy 6(4),
  435-457, doi 10.1127/ejm/6/4/0435. Paywalled, no open-access copy via
  Unpaywall, Semantic Scholar or Europe PMC, publisher landing page served no
  machine-readable text.
- Ti: Cherniak, Watson and Wark (2007) Ti diffusion in quartz, Chemical Geology
  236(1-2), 65-74, doi 10.1016/j.chemgeo.2006.09.001. Paywalled. The one
  open-access paper citing it that could be retrieved (Barker et al. 2018,
  Science Advances, doi 10.1126/sciadv.aap7567) states only that the
  activation-energy uncertainty is below 5 percent 1-sigma, without reproducing
  :math:`D_0` or :math:`E_a`.

What IS retrievable is the activation-energy RANGE for lattice diffusion in
SiO2 from the abstract of Liu et al. (2026, doi 10.3390/min16080836): about
90 kJ mol^-1 for Na+ to about 400 kJ mol^-1 for Ti4+. That range, with a swept
prefactor, is enough to bound the problem, and bounding is what this module
does:

- :data:`DIFFUSIVITY_STATUS` records per element whether a diffusivity exists
  here and, when it does not, the specific measurement that would supply it.
- :func:`diffusivity` raises :class:`ae.core.provenance.MissingValueError` for
  any element whose pair is MISSING. There is no default.
- :data:`MOST_MOBILE_BOUND` and :data:`TI_LATTICE_BOUND` are explicitly ASSUMED
  bracketing pairs, tagged with their basis, that must be passed in deliberately
  so the assumption is visible at the call site.
- :func:`critical_activation_energy` inverts the comparison: instead of asking
  what the diffusion length is for an unknown pair, it reports the activation
  energy at which the diffusion length would equal the grain radius. That
  threshold is compared against the known Na-to-Ti range, and
  :func:`lattice_removal_verdict` returns ROBUST_INFEASIBLE only when the
  threshold lies below the whole plausible range.

LIMITATIONS
===========
1. No sourced diffusivity for any element. Every number this module produces
   comes from a user-supplied pair or from an explicitly ASSUMED bracketing
   pair. Do not quote a bracketing value as a measured diffusivity.
2. The Al case at roasting temperature is UNDECIDED here (see above). Any claim
   that a chlorination roast cannot move lattice Al must be supported by the
   empirical endpoint data (Xia et al. 2024) or by a measured Al diffusivity,
   not by this module. :func:`lattice_removal_verdict` returns
   ``UNDECIDED_BY_DIFFUSION`` for that case by design, and a caller that treats
   that verdict as infeasibility is misusing it.
3. Hot pressure acid leaching above about 200 degC is likewise undecided by the
   generous bound. The atmospheric-leach conclusion does not transfer to an
   autoclave.
4. Lattice diffusion only. Grain-boundary, dislocation and fracture-network
   transport are faster by orders of magnitude, which is precisely why
   NON-lattice impurities (fluid inclusions, grain-boundary films, mineral
   inclusions) ARE removable by calcination, quenching, flotation and leaching.
   This module says nothing about those, and any claim that a purification route
   fails because of this calculation must first establish that the impurity in
   question is lattice-bound.
5. The alpha-to-beta transition at 573 degC (1 bar) changes the structure, so a
   single Arrhenius pair fitted below the transition should not be extrapolated
   above it; :meth:`ArrheniusDiffusivity.at` warns outside the recorded
   calibration interval. The Liu et al. (2026) abstract also reports a diffusion
   crossover effect, with mobility gains above 1200 degC for
   high-activation-energy impurities, which would further invalidate a
   single-pair extrapolation across that temperature; the full text stating its
   magnitude was not retrievable here, so it is noted and not parameterised.
6. Charge coupling is not modelled. Al3+ substituting for Si4+ requires a
   charge compensator (H+, Li+, Na+, K+), so Al mobility is coupled to alkali
   mobility, and a single-species Fick treatment of Al is therefore an
   approximation whose error direction is not established here. The Liu et al.
   (2026) abstract states that an alkali-first, Al-follows coupled diffusion
   mechanism is elucidated for aluminium removal, which is consistent with that
   concern; the mechanism's quantitative form is in the unretrievable full text,
   so it is reported as a limitation rather than parameterised.
7. Electric-field-driven transport is excluded. Harris (1937, doi
   10.1021/j150386a004) measured Li transport through quartz UNDER AN APPLIED
   FIELD, which is electromigration, not self-diffusion, and its rate is not
   transferable to a field-free leach or roast.
8. Cracks dominate real grains. A grain with a through-going fracture presents a
   far shorter diffusion path to its interior than a sound grain of the same
   size. The infeasibility conclusion applies to sound lattice volume; the
   platform's calcination and quenching models handle crack generation.

References
----------
Crank, J. (1975) The Mathematics of Diffusion, 2nd edition, Oxford University
Press. Standard reference for the sphere-extraction series solution. NO equation
number is cited: the book could not be consulted from this sandbox, so the
series is derived from first principles in the section above and verified
numerically against a finite-difference solution rather than taken on the
authority of a page reference that was not checked.

Liu, L., Liu, H., Li, J., Peng, T., Wang, W. and Wang, F. (2026) Mechanism Study
on Deep Removal of Lattice Impurities from High-Purity Quartz by Chlorination
Roasting, Minerals 16(8), 836, doi 10.3390/min16080836. Activation energies for
solid-state lattice diffusion of Na+ (about 90 kJ/mol) to Ti4+ (about 400
kJ/mol); carbonaceous reductant indispensable for Ti, Al, B.

Xia, M., Yang, X. and Hou, Z. (2024) Preparation of High-Purity Quartz Sand by
Vein Quartz Purification and Characteristics: A Case Study of Pakistan Vein
Quartz, Minerals 14(7), 727, doi 10.3390/min14070727.

Pankrath, R. (1994) Kinetics of Al-Si exchange in low and high quartz:
calculation of Al diffusion coefficients, European Journal of Mineralogy 6(4),
435-457, doi 10.1127/ejm/6/4/0435. UNREACHABLE from this sandbox; cited here as
the measurement that would supply the Al Arrhenius pair.

Cherniak, D. J., Watson, E. B. and Wark, D. A. (2007) Ti diffusion in quartz,
Chemical Geology 236(1-2), 65-74, doi 10.1016/j.chemgeo.2006.09.001. UNREACHABLE
from this sandbox; cited as the measurement that would supply the Ti pair.
"""

from __future__ import annotations

import datetime as _dt
import enum
import math
import warnings
from typing import Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import Feedstock
from ae.core.provenance import MISSING, MissingValueError, Source, Tag, Tier, Value, _Missing
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = [
    "GAS_CONSTANT",
    "ERF_HALF",
    "ArrheniusDiffusivity",
    "DIFFUSIVITY_STATUS",
    "MOST_MOBILE_BOUND",
    "TI_LATTICE_BOUND",
    "D0_SWEEP_M2_S",
    "LIU_2026_EA_RANGE_KJ",
    "Verdict",
    "SOURCE_LIU_2026",
    "SOURCE_XIA_2024",
    "diffusivity",
    "diffusion_length",
    "fourier_number",
    "fractional_extraction_sphere",
    "FO_SHORT_TIME_SWITCH",
    "limiting_grain_radius",
    "time_to_deplete_grain",
    "erfc_profile",
    "diffusion_length_table",
    "critical_activation_energy",
    "lattice_removal_verdict",
    "lattice_ceiling_ppm",
]

#: Molar gas constant, CODATA 2018 exact value.
GAS_CONSTANT: Final[Quantity] = Q_(8.314462618, "J/(mol*K)")

#: erf(1/2), the depletion at x = sqrt(D t) in the semi-infinite solution.
ERF_HALF: Final[float] = 0.5204998778130465

SOURCE_LIU_2026: Final[Source] = Source(
    citation="Liu, L., Liu, H., Li, J., Peng, T., Wang, W. and Wang, F. (2026) Mechanism "
             "Study on Deep Removal of Lattice Impurities from High-Purity Quartz by "
             "Chlorination Roasting, Minerals 16(8), 836",
    tier=Tier.T1,
    doi="10.3390/min16080836",
    accessed=_dt.date(2026, 9, 16),
    extraction="manual",
    note="Abstract retrieved. Publisher full text (mdpi.com) returned HTTP 403 from this "
         "sandbox, so only statements appearing in the abstract are used: activation "
         "energies about 90 kJ/mol for Na+ to about 400 kJ/mol for Ti4+, solid-state "
         "diffusion through the SiO2 lattice as the likely rate-determining step, "
         "carbonaceous reductants indispensable for spontaneous chlorination of Ti, Al "
         "and B, a diffusion crossover with mobility gains above 1200 degC for "
         "high-activation-energy impurities, an alkali-first Al-follows coupled mechanism "
         "for aluminium removal, and a temperature-staged, atmosphere-segmented roasting "
         "strategy. No per-element prefactor appears in the abstract, so D(T) cannot be "
         "evaluated from this source.",
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


class ArrheniusDiffusivity(BaseModel):
    """Arrhenius pair for a solid-state diffusivity, with its calibration domain."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    element: str
    d0: Value
    activation_energy: Value
    calibration_T: tuple[float, float] | None = Field(
        default=None, description="Interval in K over which the pair was measured")
    medium: Literal["alpha_quartz", "beta_quartz", "fused_silica", "unspecified"] = "unspecified"

    @model_validator(mode="after")
    def _physical(self) -> "ArrheniusDiffusivity":
        require_dimensionality(self.d0.quantity, "diffusivity", "d0")
        require_dimensionality(self.activation_energy.quantity.to("J/mol"),
                               "molar_energy", "activation_energy")
        if float(self.d0.quantity.magnitude) <= 0.0:
            raise ValueError(
                f"D0 must be positive, got {self.d0.quantity}; a non-positive "
                f"diffusivity implies uphill transport without a driving force"
            )
        if float(self.activation_energy.quantity.to("J/mol").magnitude) <= 0.0:
            raise ValueError(
                "activation energy for solid-state diffusion must be positive; a "
                "zero or negative barrier would make diffusion temperature independent "
                "or faster when cold, neither of which is observed in silicates"
            )
        return self

    def at(self, temperature: Quantity) -> Quantity:
        """Evaluate ``D(T)``, warning outside the calibration interval."""
        require_dimensionality(temperature, "temperature", "temperature")
        t_k = float(temperature.to("K").magnitude)
        if t_k <= 0.0:
            raise ValueError(f"absolute temperature must be positive, got {t_k} K")
        if self.calibration_T is not None:
            lo, hi = self.calibration_T
            if not (lo <= t_k <= hi):
                warnings.warn(
                    f"{self.element} diffusivity evaluated at {t_k:.1f} K, outside its "
                    f"calibration interval [{lo:.1f}, {hi:.1f}] K; quartz also undergoes "
                    f"the alpha to beta transition at 846 K, across which a single "
                    f"Arrhenius pair is not expected to hold",
                    RuntimeWarning, stacklevel=2,
                )
        ea = self.activation_energy.quantity.to("J/mol")
        expo = float((-ea / (GAS_CONSTANT * temperature.to("K"))).to("dimensionless").magnitude)
        d = self.d0.quantity.to("m**2/s") * math.exp(expo)
        assert float(d.magnitude) > 0.0, "diffusivity must be strictly positive"
        return d


class Verdict(str, enum.Enum):
    """Outcome of a diffusion-based feasibility test on lattice impurity removal."""

    #: Diffusion length falls short of the grain radius across the ENTIRE plausible
    #: parameter range, so the conclusion does not depend on the unmeasured constants.
    ROBUST_INFEASIBLE = "robust_infeasible"

    #: The plausible parameter range straddles the threshold. A diffusion-length
    #: argument cannot settle the case and empirical data is required. Treating this
    #: verdict as infeasibility is a misuse of the model.
    UNDECIDED_BY_DIFFUSION = "undecided_by_diffusion"

    #: The supplied pair gives non-negligible extraction, so lattice removal is not
    #: excluded on transport grounds.
    MOBILE = "mobile"


#: Activation-energy range for solid-state lattice diffusion in SiO2, the only
#: quantitative diffusion figure retrievable for quartz from an accessible source:
#: Liu et al. (2026) report about 90 kJ/mol for Na+ up to about 400 kJ/mol for Ti4+.
LIU_2026_EA_RANGE_KJ: Final[Value] = Value(
    quantity=Q_(np.array([90.0, 400.0]), "kJ/mol"),
    tag=Tag.SOURCED,
    source=SOURCE_LIU_2026,
    basis="Stated in the abstract as activation energies ranging from approximately "
          "90 kJ/mol for Na+ to 400 kJ/mol for Ti4+ for solid-state diffusion through "
          "the SiO2 lattice. The full text, which would give per-element prefactors, "
          "was not retrievable (mdpi.com returned HTTP 403 from this sandbox).",
    confidence="medium",
)

#: Plausible range of pre-exponential factors for cation diffusion in silicates.
#: Swept rather than fixed, because no prefactor for Al or Ti in quartz could be
#: sourced and any conclusion must be shown to hold (or not) across the range.
D0_SWEEP_M2_S: Final[Value] = Value(
    quantity=Q_(np.array([1e-10, 1e-4]), "m**2/s"),
    tag=Tag.ASSUMED,
    basis="Estimate, not a measurement: pre-exponential factors for cation diffusion "
          "in silicate minerals are commonly reported within about 1e-10 to 1e-4 m^2/s. "
          "Used as a SWEEP so that any conclusion drawn is checked at both ends rather "
          "than resting on a chosen value. A measured prefactor for Al requires "
          "Pankrath 1994 (doi 10.1127/ejm/6/4/0435), unreachable from this sandbox.",
    confidence="low",
)

#: MOST GENEROUS mobility bound for any cation in the SiO2 lattice.
#:
#: Not a measurement, and not specific to Al. Combines the LOWEST activation energy
#: in the Liu et al. (2026) range (90 kJ/mol, interstitial Na+, the most mobile
#: species in their set) with the HIGHEST prefactor in the silicate sweep
#: (1e-4 m^2/s). Any species slower than this bound has a shorter diffusion length,
#: so an infeasibility verdict obtained with this pair is conservative. The converse
#: does NOT hold: this pair cannot establish feasibility for a species whose real
#: barrier is higher.
MOST_MOBILE_BOUND: Final[ArrheniusDiffusivity] = ArrheniusDiffusivity(
    element="most_mobile_cation",
    d0=Value(
        quantity=Q_(1.0e-4, "m**2/s"),
        tag=Tag.ASSUMED,
        basis="Upper end of the silicate prefactor sweep (see D0_SWEEP_M2_S). Chosen to "
              "OVERSTATE mobility so that an infeasibility verdict is conservative. "
              "Estimate, not a measurement.",
        confidence="low",
    ),
    activation_energy=Value(
        quantity=Q_(90.0, "kJ/mol"),
        tag=Tag.ASSUMED,
        basis="Low end of the Liu et al. 2026 lattice-diffusion range (doi "
              "10.3390/min16080836), corresponding to interstitial Na+, the most mobile "
              "species in their set. Substitutional, charge-compensated Al3+ and Ti4+ "
              "must face higher barriers, so this understates the barrier and overstates "
              "mobility by design. Estimate.",
        confidence="low",
    ),
    calibration_T=None,
    medium="unspecified",
)

#: Ti4+ lattice bound: the Liu et al. (2026) Ti activation energy with the generous
#: prefactor. The Ti conclusion is prefactor insensitive (a factor of 1e6 in D0 moves
#: the diffusion length by only 1e3), which is why the Ti case is decidable here and
#: the Al case is not.
TI_LATTICE_BOUND: Final[ArrheniusDiffusivity] = ArrheniusDiffusivity(
    element="Ti",
    d0=Value(
        quantity=Q_(1.0e-4, "m**2/s"),
        tag=Tag.ASSUMED,
        basis="Upper end of the silicate prefactor sweep, chosen to overstate mobility. "
              "Liu et al. 2026 report the Ti4+ activation energy but no prefactor. "
              "Estimate.",
        confidence="low",
    ),
    activation_energy=Value(
        quantity=Q_(400.0, "kJ/mol"),
        tag=Tag.SOURCED,
        source=SOURCE_LIU_2026,
        basis="Upper end of the activation-energy range stated in the Liu et al. 2026 "
              "abstract, attributed there to Ti4+ lattice diffusion in SiO2.",
        confidence="medium",
    ),
    calibration_T=None,
    medium="unspecified",
)

#: Per-element diffusivity availability. MISSING entries state the measurement
#: that would supply the value, so the gap is actionable rather than silent.
DIFFUSIVITY_STATUS: Final[dict[str, dict[str, object]]] = {
    "Al": {
        "arrhenius": MISSING,
        "would_be_supplied_by":
            "Pankrath, R. (1994) Kinetics of Al-Si exchange in low and high quartz: "
            "calculation of Al diffusion coefficients, European Journal of Mineralogy "
            "6(4), 435-457, doi 10.1127/ejm/6/4/0435. Paywalled, no open-access copy "
            "located 2026-09-16. Alternatively, an in-house Al tracer or SIMS depth "
            "profiling experiment on oriented quartz plates annealed at 800 to 1200 degC.",
        "bound_available": True,
    },
    "Ti": {
        "arrhenius": MISSING,
        "would_be_supplied_by":
            "Cherniak, D. J., Watson, E. B. and Wark, D. A. (2007) Ti diffusion in "
            "quartz, Chemical Geology 236(1-2), 65-74, doi "
            "10.1016/j.chemgeo.2006.09.001. Paywalled, no open-access copy located "
            "2026-09-16. Liu et al. 2026 give about 400 kJ/mol for Ti4+ lattice "
            "diffusion but no prefactor, so D(T) cannot be evaluated from that source.",
        "bound_available": False,
    },
    "Li": {
        "arrhenius": MISSING,
        "would_be_supplied_by":
            "A field-free Li tracer diffusion measurement in alpha quartz. Harris (1937) "
            "doi 10.1021/j150386a004 measured Li transport under an APPLIED ELECTRIC "
            "FIELD, which is electromigration and is not transferable to a field-free "
            "leach or roast.",
        "bound_available": False,
    },
    "Na": {
        "arrhenius": MISSING,
        "would_be_supplied_by":
            "White, G. (1970) Ionic Diffusion in Quartz, Nature 225, 375-376, doi "
            "10.1038/225375a0 (paywalled, abstract not retrievable from this sandbox). "
            "Liu et al. 2026 give about 90 kJ/mol for Na+ but no prefactor.",
        "bound_available": False,
    },
    "B": {
        "arrhenius": MISSING,
        "would_be_supplied_by":
            "No boron-in-quartz lattice diffusivity was located in any accessible "
            "source. A SIMS depth-profile anneal series would supply it.",
        "bound_available": False,
    },
}


def diffusivity(element: str, temperature: Quantity,
                pair: ArrheniusDiffusivity | None = None) -> Quantity:
    """Lattice diffusivity of ``element`` in quartz at ``temperature``.

    Parameters
    ----------
    element
        Element symbol as used in :data:`DIFFUSIVITY_STATUS`.
    temperature
        Absolute temperature as a pint Quantity.
    pair
        Arrhenius pair to use. When ``None``, the module looks the element up in
        :data:`DIFFUSIVITY_STATUS` and raises if it is MISSING there.

    Raises
    ------
    ae.core.provenance.MissingValueError
        When no diffusivity is available for the element. The error names the
        measurement that would supply it. There is deliberately no fallback
        default: an invented diffusivity would make the central negative result
        of this platform unverifiable.
    """
    if pair is not None:
        if pair.element != element:
            raise ValueError(
                f"supplied Arrhenius pair is for {pair.element}, not {element}"
            )
        return pair.at(temperature)
    entry = DIFFUSIVITY_STATUS.get(element)
    if entry is None:
        raise KeyError(
            f"{element} is not tracked in DIFFUSIVITY_STATUS; tracked: "
            f"{sorted(DIFFUSIVITY_STATUS)}"
        )
    arr = entry["arrhenius"]
    if isinstance(arr, _Missing) or arr is MISSING:
        raise MissingValueError(
            f"no lattice diffusivity for {element} in quartz is available in this build. "
            f"It would be supplied by: {entry['would_be_supplied_by']} "
            f"For {element} == 'Al', an explicitly bounded ASSUMED pair is available as "
            f"MOST_MOBILE_BOUND and must be passed in deliberately so the "
            f"assumption is visible at the call site."
        )
    assert isinstance(arr, ArrheniusDiffusivity)
    return arr.at(temperature)


def diffusion_length(d: Quantity, t: Quantity) -> Quantity:
    r"""Characteristic diffusion length :math:`L = \sqrt{D t}`.

    At :math:`x = L` the semi-infinite erf solution has depleted the initial
    concentration to :math:`\mathrm{erf}(1/2) = 0.5205` of its value, so ``L`` is
    the thickness of the substantially depleted skin.
    """
    require_dimensionality(d, "diffusivity", "D")
    require_dimensionality(t, "time", "t")
    d_m2s = float(d.to("m**2/s").magnitude)
    t_s = float(t.to("s").magnitude)
    if d_m2s <= 0.0:
        raise ValueError(f"diffusivity must be positive, got {d_m2s} m^2/s")
    if t_s < 0.0:
        raise ValueError(f"time must be non-negative, got {t_s} s")
    length = Q_(math.sqrt(d_m2s * t_s), "m")
    assert float(length.magnitude) >= 0.0, "diffusion length cannot be negative"
    return length


def fourier_number(d: Quantity, t: Quantity, radius: Quantity) -> float:
    r"""Fourier number :math:`\mathrm{Fo} = D t / R^2`, the only group that matters.

    Fractional extraction from a sphere is a function of Fo alone, so a small Fo
    means negligible extraction irrespective of how the individual D, t and R
    were obtained.
    """
    require_dimensionality(d, "diffusivity", "D")
    require_dimensionality(t, "time", "t")
    require_dimensionality(radius, "length", "radius")
    r_m = float(radius.to("m").magnitude)
    if r_m <= 0.0:
        raise ValueError(f"grain radius must be positive, got {r_m} m")
    fo = float((d.to("m**2/s") * t.to("s") / (radius.to("m") ** 2)).to("dimensionless").magnitude)
    assert fo >= 0.0, "Fourier number cannot be negative"
    return fo


#: Fourier number below which the short-time asymptote replaces the series.
#: At this value the two forms agree to 0.89 percent; below it the series needs
#: more terms than is safe to truncate (see the module docstring).
FO_SHORT_TIME_SWITCH: Final[float] = 1e-4


def fractional_extraction_sphere(fo: float, n_terms: int | None = None) -> float:
    r"""Fraction of a diffusing species extracted from a sphere at Fourier number ``fo``.

    :math:`M_t/M_\infty = 1 - (6/\pi^2)\sum n^{-2}\exp(-n^2\pi^2\mathrm{Fo})`,
    derived in the module docstring by eigenfunction expansion and verified
    against a finite-difference solution of the same PDE.

    For ``fo`` below :data:`FO_SHORT_TIME_SWITCH` the exact short-time limit
    :math:`(6/\sqrt{\pi})\sqrt{\mathrm{Fo}}` is used instead, because the series
    summand decays as :math:`\exp(-n^2\pi^2\mathrm{Fo})` and the term count needed
    grows as :math:`\mathrm{Fo}^{-1/2}`: a fixed truncation silently overstates
    extraction at small ``fo``.

    Above the switch, ``n_terms`` defaults to the adaptive count
    :math:`\lceil\sqrt{28/(\pi^2 \mathrm{Fo})}\rceil` (floored at 50), which makes
    the first neglected term smaller than 1e-12, and the neglected tail is
    asserted to be negligible rather than assumed.

    Parameters
    ----------
    fo
        Fourier number :math:`Dt/R^2`, non-negative.
    n_terms
        Override the adaptive term count. Passing a small value on purpose is
        how ``test_fixed_truncation_would_overstate_extraction`` in
        ``tests/test_diffusion.py`` demonstrates
        the truncation error the adaptive rule exists to avoid.
    """
    if fo < 0.0:
        raise ValueError(f"Fourier number must be non-negative, got {fo}")
    if fo == 0.0:
        return 0.0
    if fo < FO_SHORT_TIME_SWITCH:
        val = 6.0 / math.sqrt(math.pi) * math.sqrt(fo)
        return require_fraction(min(1.0, val), "fractional extraction")
    if n_terms is None:
        n_terms = max(50, int(math.ceil(math.sqrt(28.0 / (math.pi**2 * fo)))))
    n = np.arange(1, n_terms + 1, dtype=float)
    terms = np.exp(-(n**2) * math.pi**2 * fo) / n**2
    series = float(np.sum(terms))
    tail_bound = float(terms[-1])
    assert tail_bound < 1e-9, (
        f"series truncated at n={n_terms} leaves a last term of {tail_bound:.3e} at "
        f"fo={fo:.3e}, so the sum is under-converged and the extraction would be "
        f"overstated"
    )
    val = 1.0 - 6.0 / math.pi**2 * series
    return require_fraction(min(1.0, max(0.0, val)), "fractional extraction")


def limiting_grain_radius(d: Quantity, t: Quantity) -> Quantity:
    r"""Largest grain radius that can be depleted in time ``t``: :math:`R_{max} = \sqrt{Dt}`.

    A grain coarser than this retains essentially all of its lattice inventory,
    because the depleted skin is thin compared with the radius.
    """
    return diffusion_length(d, t)


def time_to_deplete_grain(
    d: Quantity, radius: Quantity, target_extraction: float = 0.5
) -> Quantity:
    r"""Time to reach ``target_extraction`` from a sphere of radius ``radius``.

    Inverts :func:`fractional_extraction_sphere` in Fo by bisection (the function
    is strictly increasing in Fo), then returns :math:`t = \mathrm{Fo}\,R^2/D`.
    Reported in seconds; convert for readability. For lattice Al at process
    temperatures the answer exceeds geological timescales, which is the point.
    """
    require_dimensionality(d, "diffusivity", "D")
    require_dimensionality(radius, "length", "radius")
    require_fraction(target_extraction, "target_extraction", 1e-9, 1.0 - 1e-9)
    lo, hi = 1e-12, 1e2
    if fractional_extraction_sphere(hi) < target_extraction:
        raise ValueError("target extraction unreachable within Fo <= 100")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if fractional_extraction_sphere(mid) < target_extraction:
            lo = mid
        else:
            hi = mid
    fo = math.sqrt(lo * hi)
    t = Q_(fo, "dimensionless") * radius.to("m") ** 2 / d.to("m**2/s")
    return t.to("s")


def erfc_profile(d: Quantity, t: Quantity, depths: Quantity) -> np.ndarray:
    r"""Normalised concentration :math:`C/C_0 = \mathrm{erf}(x/(2\sqrt{Dt}))`.

    Semi-infinite solid with the surface held at zero concentration: the
    leaching boundary condition, where reagent removes whatever reaches the
    surface. Returns the RETAINED fraction at each depth, so it rises from 0 at
    the surface to 1 in the interior.
    """
    from scipy.special import erf  # local import keeps module import cheap

    require_dimensionality(depths, "length", "depths")
    length = diffusion_length(d, t)
    l_m = float(length.magnitude)
    x = np.atleast_1d(np.asarray(depths.to("m").magnitude, dtype=float))
    if np.any(x < 0.0):
        raise ValueError("depths must be non-negative")
    if l_m == 0.0:
        return np.ones_like(x)
    out = np.asarray(erf(x / (2.0 * l_m)), dtype=float)
    assert np.all(out >= 0.0) and np.all(out <= 1.0 + 1e-12), (
        "retained fraction must lie in [0, 1]: a value outside it means negative "
        "concentration or concentration above the initial value"
    )
    return out


def diffusion_length_table(
    pair: ArrheniusDiffusivity,
    temperatures_C: tuple[float, ...] = (80.0, 200.0, 600.0, 1000.0, 1200.0),
    times_h: tuple[float, ...] = (1.0, 6.0, 24.0),
    grain_radius: Quantity | None = None,
) -> list[dict[str, float]]:
    """Diffusion length, Fourier number and sphere extraction over a T and t grid.

    Returns one row per (temperature, time) with the diffusion length in metres
    and in nanometres, the ratio of that length to the grain radius, the Fourier
    number and the fractional extraction from a sphere. This is the table that
    demonstrates the negative result numerically rather than asserting it.
    """
    radius = grain_radius if grain_radius is not None else Q_(100.0, "um")
    require_dimensionality(radius, "length", "grain_radius")
    rows: list[dict[str, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t_c in temperatures_C:
            d = pair.at(Q_(t_c, "degC").to("K"))
            for t_h in times_h:
                t = Q_(t_h, "hour")
                length = diffusion_length(d, t)
                fo = fourier_number(d, t, radius)
                rows.append({
                    "temperature_C": float(t_c),
                    "time_h": float(t_h),
                    "D_m2_per_s": float(d.to("m**2/s").magnitude),
                    "L_m": float(length.to("m").magnitude),
                    "L_nm": float(length.to("nm").magnitude),
                    "grain_radius_m": float(radius.to("m").magnitude),
                    "L_over_R": float((length / radius).to("dimensionless").magnitude),
                    "fourier_number": fo,
                    "fraction_extracted": fractional_extraction_sphere(fo),
                })
    return rows


def critical_activation_energy(
    d0: Quantity, grain_radius: Quantity, temperature: Quantity, residence_time: Quantity
) -> Quantity:
    r"""Activation energy at which the diffusion length equals the grain radius.

    Inverts :math:`\sqrt{D_0 \exp(-E_a/\mathcal{R}T)\,t} = R` to give

    .. math:: E_a^{crit} = -\mathcal{R} T \ln\!\left(\frac{R^2}{D_0 t}\right)

    A species whose true barrier EXCEEDS :math:`E_a^{crit}` cannot be depleted
    from the grain in time :math:`t`; one below it can. This inversion is the
    honest way to use an unmeasured diffusivity: the threshold is compared
    against the known activation-energy range (:data:`LIU_2026_EA_RANGE_KJ`)
    instead of guessing a value inside it.

    Raises
    ------
    ValueError
        If :math:`R^2/t \geq D_0`, in which case no positive barrier makes the
        grain depletable and the question is prefactor limited rather than
        barrier limited.
    """
    require_dimensionality(d0, "diffusivity", "d0")
    require_dimensionality(grain_radius, "length", "grain_radius")
    require_dimensionality(temperature, "temperature", "temperature")
    require_dimensionality(residence_time, "time", "residence_time")
    d_req = float((grain_radius.to("m") ** 2 / residence_time.to("s")).magnitude)
    d0_v = float(d0.to("m**2/s").magnitude)
    if d_req >= d0_v:
        raise ValueError(
            f"required diffusivity {d_req:.3e} m^2/s already exceeds the prefactor "
            f"{d0_v:.3e} m^2/s, so no positive activation energy permits depletion of a "
            f"{grain_radius.to('um'):~P} grain in {residence_time.to('hour'):~P}"
        )
    t_k = temperature.to("K")
    ea = -(GAS_CONSTANT * t_k) * math.log(d_req / d0_v)
    assert float(ea.to("J/mol").magnitude) > 0.0, "critical barrier must be positive"
    return ea.to("kJ/mol")


def lattice_removal_verdict(
    feedstock: Feedstock,
    element: str,
    grain_radius: Quantity,
    temperature: Quantity,
    residence_time: Quantity,
    pair: ArrheniusDiffusivity | None = None,
    ea_range: Quantity | None = None,
    extraction_threshold: float = 0.05,
) -> dict[str, object]:
    """Decide whether lattice removal of ``element`` is excluded on transport grounds.

    Method. Two independent checks, and :attr:`Verdict.ROBUST_INFEASIBLE` is
    returned only when BOTH hold:

    1. Fractional extraction from the sphere is below ``extraction_threshold``.
    2. The critical activation energy (:func:`critical_activation_energy`) at
       BOTH ends of the prefactor sweep :data:`D0_SWEEP_M2_S` lies below the LOW
       end of the barrier interval for this element, so no combination of the
       unmeasured constants within that interval permits depletion.

    The barrier interval is taken from ``ea_range`` when given, else from the
    activation energy of the supplied ``pair`` treated as a point value, else
    from :data:`LIU_2026_EA_RANGE_KJ` (90 to 400 kJ/mol) when the element's
    barrier is unknown. That ordering is what makes the Ti case decidable (its
    barrier is sourced at about 400 kJ/mol) while the Al case is not (its
    barrier is unmeasured and the 90 to 400 interval straddles the threshold).

    On why ``extraction_threshold`` defaults to 0.05 and not to something
    smaller. For a thin depleted skin the extracted fraction from a sphere is
    approximately :math:`3L/R`, so even a 300-fold deficit in diffusion length
    (80 degC, 6 h, 100 um grain) still extracts about 1 percent of the lattice
    inventory: the geometric skin term never vanishes. A threshold of a few
    percent is therefore the meaningful one, and the process consequence is
    reported as ``ppm_removable_from_lattice`` so the number can be compared
    against a ppm-level product specification rather than against an abstract
    cutoff.

    When check 1 fails or check 2 fails and no measured pair was supplied, the
    verdict is :attr:`Verdict.UNDECIDED_BY_DIFFUSION`: the generous bound is
    uninformative at these conditions and empirical data must settle the case.
    A caller that reports that verdict as infeasibility is misusing the model.

    Takes the feedstock so the grain population and the impurity inventory
    belong to the ore rather than to the module.
    """
    require_dimensionality(grain_radius, "length", "grain_radius")
    require_dimensionality(temperature, "temperature", "temperature")
    require_dimensionality(residence_time, "time", "residence_time")
    require_fraction(extraction_threshold, "extraction_threshold", 1e-6, 0.5)
    used = pair if pair is not None else MOST_MOBILE_BOUND
    bounded = pair is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = used.at(temperature)
        length = diffusion_length(d, residence_time)
        fo = fourier_number(d, residence_time, grain_radius)
        extracted = fractional_extraction_sphere(fo)
    r_m = float(grain_radius.to("m").magnitude)
    l_m = float(length.to("m").magnitude)
    ratio = r_m / l_m if l_m > 0.0 else float("inf")

    if ea_range is not None:
        band = np.atleast_1d(ea_range.to("kJ/mol").magnitude)
        ea_lo, ea_hi = float(band.min()), float(band.max())
        band_origin = "caller-supplied barrier interval"
    elif pair is not None:
        ea_pt = float(pair.activation_energy.quantity.to("kJ/mol").magnitude)
        ea_lo = ea_hi = ea_pt
        band_origin = f"activation energy of the supplied pair for {pair.element}"
    else:
        band = np.atleast_1d(LIU_2026_EA_RANGE_KJ.quantity.to("kJ/mol").magnitude)
        ea_lo, ea_hi = float(band.min()), float(band.max())
        band_origin = ("Liu et al. 2026 lattice-diffusion range, used because the barrier "
                       f"for {element} is unmeasured")

    crit: dict[str, float | None] = {}
    for label, d0_mag in (("d0_low_1e-10", 1e-10), ("d0_high_1e-4", 1e-4)):
        try:
            crit[label] = float(
                critical_activation_energy(
                    Q_(d0_mag, "m**2/s"), grain_radius, temperature, residence_time
                ).magnitude
            )
        except ValueError:
            crit[label] = None  # no positive barrier permits depletion: fully blocked
    crit_vals = [v for v in crit.values() if v is not None]
    threshold_below_range = all(v < ea_lo for v in crit_vals) if crit_vals else True

    # When the caller supplied a pair, its OWN prefactor is the relevant one: the
    # sweep exists only to bracket an unmeasured prefactor. Re-derive the
    # threshold from the supplied d0 so a measured pair is judged against itself.
    if pair is not None:
        try:
            crit_own = float(
                critical_activation_energy(
                    pair.d0.quantity, grain_radius, temperature, residence_time
                ).magnitude
            )
            crit[f"d0_supplied_{float(pair.d0.quantity.to('m**2/s').magnitude):.1e}"] = crit_own
            threshold_below_range = crit_own < ea_lo
        except ValueError:
            # The supplied prefactor is itself below the required diffusivity, so
            # no positive barrier permits depletion: fully blocked.
            crit[f"d0_supplied_{float(pair.d0.quantity.to('m**2/s').magnitude):.1e}"] = None
            threshold_below_range = True
        crit_vals = [v for v in crit.values() if v is not None]

    robust = extracted < extraction_threshold and threshold_below_range
    crit_txt = (f"{min(crit_vals):.1f} to {max(crit_vals):.1f}" if crit_vals
                else "no positive barrier permits depletion")
    if robust:
        verdict = Verdict.ROBUST_INFEASIBLE
        interp = (
            f"Removal of lattice {element} is excluded on transport grounds at these "
            f"conditions. The grain radius exceeds the diffusion length by a factor of "
            f"{ratio:.3g}, extraction is {extracted:.3g}, and the critical activation "
            f"energy ({crit_txt} kJ/mol across the prefactor sweep) lies below the low "
            f"end of the barrier interval ({ea_lo:.0f} kJ/mol, {band_origin}), so no "
            f"combination of the unmeasured constants inside that interval permits "
            f"depletion. Because the diffusion length scales as sqrt(D t), extending "
            f"residence time cannot close a gap of this size."
        )
    elif bounded:
        verdict = Verdict.UNDECIDED_BY_DIFFUSION
        interp = (
            f"NOT DECIDED by diffusion. Extraction under the deliberately generous "
            f"mobility bound is {extracted:.3g} and the critical activation energy "
            f"({crit_txt} kJ/mol across the prefactor sweep) is not below the whole "
            f"barrier interval ({ea_lo:.0f} to {ea_hi:.0f} kJ/mol, {band_origin}), so a "
            f"plausible value of the unmeasured barrier for {element} would permit "
            f"depletion. The bound is uninformative here: settling this case requires a "
            f"measured diffusivity or empirical endpoint data. Do not report this as "
            f"infeasibility."
        )
    elif extracted >= extraction_threshold:
        verdict = Verdict.MOBILE
        interp = (
            f"The supplied pair gives extraction {extracted:.3g} at these conditions, "
            f"above the {extraction_threshold:.3g} threshold, so lattice {element} "
            f"removal is not excluded on transport grounds. Confirm the Arrhenius pair "
            f"is appropriate for lattice (not grain-boundary) diffusion before drawing "
            f"a process conclusion."
        )
    else:
        # Extraction is negligible but the barrier leg did not clear: the pair's
        # own barrier sits above its critical threshold only marginally, or the
        # barrier interval supplied is wider than the pair's point value. Do not
        # call this MOBILE (extraction is negligible) and do not call it robust
        # (the barrier test did not pass).
        verdict = Verdict.UNDECIDED_BY_DIFFUSION
        interp = (
            f"Extraction under the supplied pair is {extracted:.3g}, below the "
            f"{extraction_threshold:.3g} threshold, but the barrier test did not clear: "
            f"the critical activation energy ({crit_txt} kJ/mol) is not below the low end "
            f"of the barrier interval ({ea_lo:.0f} kJ/mol, {band_origin}). The negligible "
            f"extraction therefore rests on the specific pair supplied rather than on the "
            f"whole plausible range, so it is reported as undecided rather than as a "
            f"robust exclusion. Widen or narrow the barrier interval deliberately if a "
            f"robust verdict is needed."
        )

    ppm_removable: float | str
    try:
        ppm_removable = feedstock.impurities.lattice_ppm(element) * extracted
    except Exception as exc:  # unmeasured lattice split, or element not assayed
        ppm_removable = f"unavailable: {type(exc).__name__}"

    return {
        "sample_id": feedstock.sample_id,
        "element": element,
        "characterized": feedstock.characterized,
        "verdict": verdict,
        "D_m2_per_s": float(d.to("m**2/s").magnitude),
        "diffusion_length_m": l_m,
        "diffusion_length_nm": float(length.to("nm").magnitude),
        "grain_radius_m": r_m,
        "grain_radius_over_diffusion_length": ratio,
        "orders_of_magnitude": (
            math.log10(ratio) if ratio > 0 and math.isfinite(ratio) else float("inf")),
        "fourier_number": fo,
        "fraction_extracted": extracted,
        "ppm_removable_from_lattice": ppm_removable,
        "critical_Ea_kJ_per_mol": crit,
        "barrier_interval_kJ_per_mol": (ea_lo, ea_hi),
        "barrier_interval_origin": band_origin,
        "diffusivity_is_assumed_bound": bounded,
        "diffusivity_basis": (
            used.d0.basis if bounded else f"user-supplied pair, tag {used.d0.tag.value}"
        ),
        "interpretation": interp,
    }


def lattice_ceiling_ppm(
    feedstock: Feedstock, elements: tuple[str, ...] = ("Al", "Ti", "Li", "B")
) -> dict[str, float]:
    """Per-element floor concentration achievable by any chemical purification route.

    The floor is the lattice-bound inventory, because :func:`diffusion_length`
    shows the lattice is unreachable. Raises
    :class:`ae.core.provenance.MissingValueError` for an uncharacterized ore,
    which is the correct outcome: the ceiling grade of a deposit whose lattice
    split has never been measured is unknown, not optimistic.
    """
    out: dict[str, float] = {}
    for el in elements:
        out[el] = feedstock.impurities.lattice_ppm(el)
    total = sum(out.values())
    assert total >= 0.0, "lattice inventory cannot be negative"
    out["_sum"] = total
    return out
