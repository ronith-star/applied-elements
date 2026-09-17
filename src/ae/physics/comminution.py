"""Comminution energy: the Bond, Rittinger and Kick size-reduction laws.

This module answers one question, "how much electrical energy does it take to
grind this feedstock from F80 to P80", and it answers it with the semi-empirical
laws that the mineral processing literature actually uses, with their unit
conventions made explicit rather than assumed.

The short ton trap
------------------
Bond's third theory is conventionally written

.. math::
    W = 10\\, W_i \\left( \\frac{1}{\\sqrt{P_{80}}} - \\frac{1}{\\sqrt{F_{80}}} \\right)

and in Bond's own convention ``W`` and ``W_i`` are both in kWh per SHORT TON
while ``P_80`` and ``F_80`` are in micrometres. The coefficient 10 is not a unit
conversion: it is :math:`\\sqrt{100\\,\\mu m}`, which falls out of Bond's
definition of the work index as the energy to reduce a feed of effectively
infinite size to a product with :math:`P_{80} = 100\\,\\mu m`. Setting
:math:`F_{80} \\to \\infty` and :math:`P_{80} = 100` gives
:math:`W = 10 W_i / 10 = W_i`, which is the definition, and
:func:`test_bond_definitional_identity` checks exactly that.

Much of the modern literature quotes :math:`W_i` in kWh per METRIC TONNE and
plugs it into the same equation, obtaining ``W`` in kWh per metric tonne. Both
readings are internally consistent; mixing them is a 10.23 percent error, in the
direction of understating energy when a kWh/short-ton index is read as metric.
:class:`TonConvention` therefore makes the choice explicit at every call, and
there is no default. 1 short ton = 907.18474 kg exactly (NIST SP 811, Appendix
B.8), so 1 metric tonne = 1.1023113 short tons.

Equations
---------
**(E1) Bond's third theory of comminution.** Empirical, from Bond 1952 and Bond
1961, restated as Eq. 2 and Eq. 4 of Arellano-Pina et al. 2023
(doi:10.37190/ppmp/172458).

.. math::
    W = 10\\, W_i \\left( P_{80}^{-1/2} - F_{80}^{-1/2} \\right)

- :math:`W`: net specific grinding energy at the mill pinion, kWh per ton
  (short or metric, per :class:`TonConvention`). Valid range: 0 to about
  100 kWh/t; above that the assumption of a single work index over the size
  interval fails.
- :math:`W_i`: Bond work index, kWh per ton, same ton as :math:`W`. Observed
  range for silicate ores about 7 to 25 kWh/short ton.
- :math:`P_{80}`: product 80 percent passing size, micrometres. Bond's
  calibration domain is roughly 30 to 3000 um; see LIMITATIONS.
- :math:`F_{80}`: feed 80 percent passing size, micrometres, must exceed
  :math:`P_{80}`. Calibration domain roughly 300 um to 50 mm.
- The bare ``10`` carries units of :math:`\\mu m^{1/2}` and is
  :math:`\\sqrt{100\\,\\mu m}` by construction, not a fitted constant.

**(E2) Bond laboratory grindability equation.** Eq. 5 of Arellano-Pina et al.
2023, who attribute the form and the constants to Bond 1961.

.. math::
    W_i = \\frac{\\gamma}{p_i^{\\alpha}\\, G^{\\beta}\\,
          \\left( 10 P_{80}^{-1/2} - 10 F_{80}^{-1/2} \\right)}

- :math:`\\gamma = 44.5`: constant tied to the 44.5 lb (20.18 kg) standard ball
  charge. Dimensionally inconsistent as written; see LIMITATIONS.
- :math:`p_i`: closing screen aperture, micrometres, typically 106 to 300 um
  (the 100 mesh case is 149 um).
- :math:`G`: grindability, grams of undersize produced per mill revolution at a
  250 percent circulating load. Observed range about 0.5 to 3 g/rev.
- :math:`\\alpha = 0.23`, :math:`\\beta = 0.82`: exponents fitted by Bond; the
  criteria he used are not recorded in the literature (Arellano-Pina et al.
  2023, Introduction, immediately after their Eq. 5: the criteria used by Bond
  to determine the numerical values of the constants are stated to be unknown).
- Output :math:`W_i` is in kWh per short ton in Bond's original convention.

**(E3) Generalised Walker size-reduction differential.** First principles in the
sense that all three classical laws are the same integral with one exponent.

.. math::
    \\mathrm{d}W = -C\\, x^{-n}\\, \\mathrm{d}x

Integrating from :math:`F_{80}` to :math:`P_{80}`:

.. math::
    n = 1:\\; W = C \\ln (F_{80}/P_{80}) \\quad \\text{(Kick)}

.. math::
    n = 3/2:\\; W = 2C\\left( P_{80}^{-1/2} - F_{80}^{-1/2} \\right)
    \\quad \\text{(Bond, with } 2C = 10 W_i )

.. math::
    n = 2:\\; W = C\\left( P_{80}^{-1} - F_{80}^{-1} \\right)
    \\quad \\text{(Rittinger)}

- :math:`x`: particle size, micrometres.
- :math:`n`: dimensionless exponent. 1, 1.5 and 2 recover Kick, Bond and
  Rittinger respectively. Austin 1973 (doi:10.1016/0032-5910(73)80042-7) shows
  the three laws are not competing physical theories but limiting cases whose
  applicability is set by the size range.
- :math:`C`: law-specific constant whose units depend on :math:`n`, namely
  kWh/ton times :math:`\\mu m^{n-1}`. It is NOT transferable between laws.

**(E4) Ton convention conversion.**

.. math::
    W_{\\text{metric}} = W_{\\text{short}} \\times
    \\frac{1000\\,\\mathrm{kg}}{907.18474\\,\\mathrm{kg}}
    = 1.1023113\\, W_{\\text{short}}

Energy per unit mass rises when the reference mass rises, so the metric figure
is the larger of the two. Confusing the direction is the second classic error
after confusing the conventions at all.

Which law applies where
-----------------------
Kick for coarse crushing (product above roughly 50 mm), Bond for intermediate
crushing and conventional ball milling (roughly 50 um to 50 mm), Rittinger for
fine and ultrafine grinding where the created surface dominates (below roughly
50 to 100 um). Hukki's 1961 argument, summarised by Austin 1973, is that the
exponent :math:`n` drifts continuously with size rather than switching at a
boundary, so the three laws are three tangents to one curve.
:func:`recommended_law` returns the conventional choice and carries no claim to
be exact at a boundary.

LIMITATIONS
-----------
1. **Bond's equation is empirical and dimensionally unsound.** Eq. E2 in
   particular cannot be made dimensionally homogeneous: :math:`\\gamma` absorbs
   a pound-mass, a mill geometry and a revolution count, and
   :math:`p_i^{0.23} G^{0.82}` is not a physical group. Arellano-Pina et al.
   2023 state directly that multiple errors have been reported in the
   formulation because of inadequate dimensional analysis and that the model is
   semi-empirical. Do not differentiate it, do not extrapolate it, and do not
   read :math:`W_i` as a material property independent of the test device.
2. **The work index is not size-independent.** Bond's own procedure fixes a
   250 percent circulating load and a closing screen; changing the closing
   screen changes the measured :math:`W_i` for the same ore. Rodriguez et al.
   2021 (doi:10.3390/met11060970) report that for W and Ta ores the relation
   between :math:`W_i` and grinding size showed no clear correlation while the
   grindability index :math:`G` correlated robustly with grinding size, and
   advise correlating against :math:`G` rather than :math:`W_i`. Treat a single
   :math:`W_i` used across a wide size interval as an approximation.
3. **Eq. E2's constants are specific to the 30.5 by 30.5 cm Bond mill.**
   Arellano-Pina et al. 2023 measured errors up to 68.3 percent when
   :math:`\\alpha, \\beta, \\gamma` from the standard mill were applied to a
   20.5 cm diameter mill, even with matched fill and critical speed. Never
   apply this module's :func:`bond_wi_from_grindability` to a non-standard mill
   without re-fitting the constants.
4. **No work index in this module is measured on Vikarabad ore.** The deposit is
   uncharacterized in the citable record. Every ore-type prior here is either a
   measurement on a different deposit, explicitly identified, or an ASSUMED
   value with a stated basis. A Bond work index is cheap to measure and should
   be measured before any mill is sized.
5. **Net pinion energy, not plant energy.** ``W`` is the net energy at the mill
   shell. Motor, gearbox and classifier inefficiency, and the Bond efficiency
   factors for dry grinding, open circuit, oversize feed and fine product, are
   not applied here. Plant draw is higher, commonly by 10 to 30 percent for
   drive losses alone, and that correction belongs in a plant model with its own
   sourced factors.
6. **Feed and product must be reasonably size-graded.** ``F80``/``P80`` collapse
   a whole distribution to one number. For a bimodal feed, or for the deliberate
   narrow size cuts an HPQ sand plant sells, energy predicted from an 80 percent
   passing pair can be substantially wrong.

References
----------
Every entry below was resolved against the Crossref REST API for the DOI shown,
and the fields here (author list, year, title, journal, volume, issue, pages)
are as Crossref returned them. Resolved 17 September 2026. Entries without a
DOI say what was checked instead and what remains unverified.

Austin, L. G. (1973) A commentary on the Kick, Bond and Rittinger laws of
grinding, Powder Technology 7(6), 315-317, doi 10.1016/0032-5910(73)80042-7.
Source of the claim that the three laws are limiting cases of one relation
rather than competing theories. Closed access, abstract only from this sandbox;
the exponent argument is cited, no numerical constant is taken from it.

Arellano-Pina, R., Sanchez-Ramirez, E. A., Perez-Garibay, R. and
Gutierrez-Perez, V. H. (2023) Bond's work index estimation using non-standard
ball mills, Physicochemical Problems of Mineral Processing,
doi 10.37190/ppmp/172458. Open access, full text retrieved. Source of Eq. E1 and
Eq. E2 as restated here, of the constants alpha = 0.23, beta = 0.82,
gamma = 44.5, of the statement that Bond's criteria for those constants are not
recorded, and of the non-standard-mill error measurements. Crossref returns no
volume, issue or page range for this article.

Garcia, G. G., Oliva, J., Guasch, E., Anticoi, H., Coello-Velazquez, A. L. and
Menendez-Aguado, J. M. (2021) Variability Study of Bond Work Index and
Grindability Index on Various Critical Metal Ores, Metals 11(6), 970,
doi 10.3390/met11060970. Source of the W and Ta finding that the work index does
not correlate cleanly with grinding size while the grindability index does.
Abstract only; the publisher blocked the full text from this sandbox. THIS ENTRY
WAS PREVIOUSLY WRONG: it named "Rodriguez B. A., Menendez-Aguado J. M. et al.",
written from memory. Rodriguez is not an author of this paper.

Wills, B. A. and Finch, J. A. (2016) Comminution, in Wills' Mineral Processing
Technology, 8th edition, Elsevier, pp. 109-122,
doi 10.1016/b978-0-08-097053-0.00005-4. Cited for the size ranges over which
each size-reduction law is conventionally applied. Crossref gives 2016 as the
year of this chapter record.

Bond, F. C. (1952) The Third Theory of Comminution, Transactions AIME, Mining
Engineering, 484-494. NOT SOURCED DIRECTLY: no DOI exists for this article and
no full text or publisher record was reachable from this sandbox. The author,
year, title, and page range above are as cited in the reference list of
Arellano-Pina et al. 2023 (verified, above), which is the route by which Eq. E1
enters this module. It is a secondary attribution, not a primary verification.

Bond, F. C. (1961) Crushing and grinding calculations, Allis-Chalmers
Publication No. 07R9235B. NOT SOURCED DIRECTLY: same status as Bond 1952. The
bibliographic fields above are as cited by Arellano-Pina et al. 2023, who
attribute the Eq. E2 form and its constants to it. No primary copy was reachable.

Hukki, R. T. (1961). NOT SOURCED. The in-text reference to Hukki's 1961
argument is retained only as reported by Austin 1973, which is the form in which
it reaches this module; no Hukki publication record could be resolved from this
sandbox by DOI, by Crossref bibliographic search, or by OpenAlex search. Nothing
in this module depends numerically on it.

NIST Special Publication 811 (2008 edition), Guide for the Use of the
International System of Units (SI), Appendix B.8. Source of the exact
conversion 1 short ton = 907.18474 kg. Reachable only as a document reference,
not through Crossref; the conversion factor itself is exact by definition of the
avoirdupois pound (0.45359237 kg exactly) and is reproduced arithmetically in
this module's tests rather than taken on the authority of the page citation.
"""

from __future__ import annotations

import datetime as _dt
import enum
import math
from typing import TYPE_CHECKING, Any, Final, cast

from ae.core.feedstock import Feedstock, OreType
from ae.core.provenance import Distribution, Source, Tag, Tier, Value, _Missing
from ae.core.site import Site
from ae.core.units import (
    DIMS,
    Q_,
    Quantity,
    require_dimensionality,
)

if TYPE_CHECKING:
    from pint import Quantity as _PintQuantity

    # ae.core.units.Quantity is a runtime alias for UREG.Quantity, which mypy
    # cannot use as a type annotation. Qty annotates against pint's generic
    # Quantity[Any], which is a BASE class of the registry-bound UREG.Quantity
    # and not the same class (isinstance holds, identity does not), so the
    # annotation approximates the runtime class: it cannot express the registry
    # binding, and mypy will therefore not catch a quantity built on a foreign
    # registry. Runtime behaviour is unchanged, the shared UREG registry remains
    # the only registry in play, and every dimensionality check stays a runtime
    # check through require_dimensionality.
    type Qty = _PintQuantity[Any]
else:
    Qty = Quantity


__all__ = [
    "BOND_GRINDABILITY_ALPHA",
    "BOND_GRINDABILITY_BETA",
    "BOND_GRINDABILITY_GAMMA",
    "BOND_REFERENCE_P80_UM",
    "METRIC_TON_KG",
    "SHORT_TONS_PER_METRIC_TON",
    "SHORT_TON_KG",
    "SRC_AUSTIN_1973",
    "SRC_MET_VARIABILITY",
    "SRC_NIST_SP811",
    "SRC_PPMP_BOND",
    "SRC_WILLS_COMMINUTION",
    "SizeLaw",
    "TonConvention",
    "bond_specific_energy",
    "bond_wi_from_grindability",
    "bond_work_index_from_energy",
    "convert_ton_convention",
    "feedstock_work_index",
    "grinding_energy_cost",
    "grinding_specific_energy",
    "kick_specific_energy",
    "percent_error",
    "recommended_law",
    "rittinger_specific_energy",
    "sourced_quartz_rich_benchmark",
    "walker_specific_energy",
    "work_index_prior",
]

# --- exact definitions ------------------------------------------------------
#: kg per short ton, exact by definition (NIST SP 811 Appendix B.8).
SHORT_TON_KG: Final[float] = 907.18474
#: kg per metric tonne, exact by definition of the tonne.
METRIC_TON_KG: Final[float] = 1000.0
#: Short tons per metric tonne = 1000 / 907.18474 = 1.1023113109...
SHORT_TONS_PER_METRIC_TON: Final[float] = METRIC_TON_KG / SHORT_TON_KG

#: The product size, in micrometres, that defines the Bond work index. The
#: factor 10 in Eq. E1 is sqrt(100 um), so this constant and that coefficient
#: are the same statement written two ways.
BOND_REFERENCE_P80_UM: Final[float] = 100.0

#: Exponents and constant of Bond's laboratory grindability equation (Eq. E2).
BOND_GRINDABILITY_ALPHA: Final[float] = 0.23
BOND_GRINDABILITY_BETA: Final[float] = 0.82
BOND_GRINDABILITY_GAMMA: Final[float] = 44.5

def _q(value: Value) -> Qty:
    """Annotation-level accessor for ``Value.quantity``.

    ``ae.core.provenance.Value.quantity`` is annotated with the runtime alias
    ``ae.core.units.Quantity``, which mypy resolves to a variable rather than a
    type. This helper re-expresses the same object as ``Qty`` so callers type
    check without altering a core module or weakening a runtime check.
    """
    return cast(Qty, value.quantity)


_ACCESSED: Final[_dt.date] = _dt.date(2026, 9, 16)

SRC_PPMP_BOND: Final[Source] = Source(
    citation=(
        "Arellano-Pina R., Sanchez-Ramirez E., Perez-Garibay R. et al. 2023, "
        "Bond's work index estimation using non-standard ball mills, "
        "Physicochemical Problems of Mineral Processing 59(6):172458"
    ),
    tier=Tier.T1,
    doi="10.37190/ppmp/172458",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Equations 2, 4, 5 and 13 and Table 2 read from the publisher PDF. Restates "
        "Bond 1952 and Bond 1961; Bond's primary papers were not retrievable in this "
        "environment and are cited through this peer-reviewed secondary source."
    ),
)

SRC_NIST_SP811: Final[Source] = Source(
    citation=(
        "Thompson A. and Taylor B. N. 2008, Guide for the Use of the International "
        "System of Units (SI), NIST Special Publication 811, Appendix B.8"
    ),
    tier=Tier.T1,
    url="https://www.nist.gov/pml/special-publication-811",
    accessed=_ACCESSED,
    note="Short ton defined as 907.18474 kg exactly. Definition, not a measurement.",
)

SRC_AUSTIN_1973: Final[Source] = Source(
    citation=(
        "Austin L. G. 1973, A commentary on the Kick, Bond and Rittinger laws of "
        "grinding, Powder Technology 7(6):315-317"
    ),
    tier=Tier.T1,
    doi="10.1016/0032-5910(73)80042-7",
    accessed=_ACCESSED,
    note=(
        "Metadata verified via CrossRef. Full text not retrievable in this environment; "
        "cited for the established position that the three laws are limiting cases of "
        "one differential form, which is textbook content, not a novel claim."
    ),
)

SRC_WILLS_COMMINUTION: Final[Source] = Source(
    citation=(
        "Wills B. A. and Finch J. A. 2016, Comminution, in Wills' Mineral Processing "
        "Technology, 8th edition, Elsevier, pp. 109-122"
    ),
    tier=Tier.T1,
    doi="10.1016/b978-0-08-097053-0.00005-4",
    accessed=_ACCESSED,
    note="Standard reference for the size ranges over which each law is conventionally used.",
)

SRC_MET_VARIABILITY: Final[Source] = Source(
    citation=(
        "Garcia G. G., Oliva J., Guasch E., Anticoi H., Coello-Velazquez A. L. and "
        "Menendez-Aguado J. M. 2021, Variability Study of Bond Work Index and Grindability "
        "Index on Various Critical Metal Ores, Metals 11(6):970"
    ),
    tier=Tier.T1,
    doi="10.3390/met11060970",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Abstract read; full text blocked by publisher in this environment. Cited only "
        "for the abstract's own conclusion that Wi varies with grinding size while the "
        "grindability index correlates robustly with it. AUTHOR LIST CORRECTED: this "
        "entry read \"Rodriguez B. A., Menendez-Aguado J. M. et al.\" until it was "
        "resolved against Crossref, which gives Garcia as first author. Rodriguez is not "
        "an author of this paper; the name was written from memory and never checked."
    ),
)


class TonConvention(str, enum.Enum):
    """Which ton the work index and the specific energy are denominated in.

    There is deliberately no default. A silent default is how the 10.23 percent
    error in this module's subject matter propagates into a mill sizing.
    """

    SHORT = "short"    # kWh per short ton, 907.18474 kg. Bond's own convention.
    METRIC = "metric"  # kWh per metric tonne, 1000 kg. Most modern papers.

    @property
    def energy_unit(self) -> str:
        return "kWh/short_ton" if self is TonConvention.SHORT else "kWh/metric_ton"


class SizeLaw(str, enum.Enum):
    """The three classical size-reduction laws, as exponents of Eq. E3."""

    KICK = "kick"            # n = 1
    BOND = "bond"            # n = 3/2
    RITTINGER = "rittinger"  # n = 2

    @property
    def exponent(self) -> float:
        return {"kick": 1.0, "bond": 1.5, "rittinger": 2.0}[self.value]


# --- internal guards --------------------------------------------------------


def _check_sizes(f80_um: float, p80_um: float) -> None:
    """Physical sanity on a size pair. Assertions live here, not only in tests."""
    if not math.isfinite(p80_um) or p80_um <= 0.0:
        raise ValueError(f"p80 must be finite and positive, got {p80_um}")
    if f80_um <= 0.0:
        raise ValueError(f"f80 must be positive, got {f80_um}")
    if p80_um > f80_um:
        raise ValueError(
            f"p80 ({p80_um} um) exceeds f80 ({f80_um} um): comminution cannot make "
            f"particles coarser, and Eq. E1 would return a negative energy"
        )


def _positive_energy(w: float, label: str) -> float:
    """Second law guard: grinding consumes work, so specific energy is non-negative."""
    if w < 0.0:
        raise AssertionError(
            f"{label} returned negative specific energy ({w}), which would mean "
            f"comminution generates work. Check the size ordering."
        )
    return w


# --- Eq. E1, E2, E3, E4 -----------------------------------------------------


def bond_specific_energy(
    work_index: Qty,
    f80: Qty,
    p80: Qty,
    convention: TonConvention,
) -> Qty:
    """Net specific grinding energy from Bond's third theory (Eq. E1).

    Parameters
    ----------
    work_index
        Bond work index as a Quantity in ``kWh/short_ton`` or ``kWh/metric_ton``.
        Its unit must match ``convention``; a mismatch raises rather than
        converts, because an automatic conversion here is exactly the silent
        10.23 percent error this module exists to prevent.
    f80, p80
        Feed and product 80 percent passing sizes, any length unit.
    convention
        Ton basis of the answer.

    Returns
    -------
    Quantity
        Specific energy in ``convention.energy_unit``.

    Examples
    --------
    Wi = 12.3 kWh/metric ton, F80 = 2000 um, P80 = 150 um:
    10(12.3)(1/sqrt(150) - 1/sqrt(2000)) = 123(0.0816497 - 0.0223607)
    = 123(0.0592890) = 7.2925 kWh per metric tonne.

    >>> w = bond_specific_energy(Q_(12.3, "kWh/metric_ton"), Q_(2000.0, "um"),
    ...                          Q_(150.0, "um"), TonConvention.METRIC)
    >>> round(w.magnitude, 4)
    7.2925
    """
    require_dimensionality(work_index, "specific_energy_mass", "work_index")
    require_dimensionality(f80, "length", "f80")
    require_dimensionality(p80, "length", "p80")
    expected = convention.energy_unit
    if work_index.dimensionality != DIMS["specific_energy_mass"]:  # pragma: no cover
        raise AssertionError("dimensionality check above should have caught this")
    wi_mag = float(work_index.to(expected).magnitude)
    if wi_mag < 0.0:
        raise ValueError(f"work index must be non-negative, got {wi_mag}")
    f = float(f80.to("um").magnitude)
    p = float(p80.to("um").magnitude)
    _check_sizes(f, p)
    coeff = math.sqrt(BOND_REFERENCE_P80_UM)  # = 10, in um**0.5
    w = coeff * wi_mag * (p ** -0.5 - f ** -0.5)
    return Q_(_positive_energy(w, "bond_specific_energy"), expected)


def bond_work_index_from_energy(
    specific_energy: Qty,
    f80: Qty,
    p80: Qty,
    convention: TonConvention,
) -> Qty:
    """Invert Eq. E1 to get the operating work index from a measured mill draw.

    This is the standard back-calculation used to compare a plant against a
    laboratory index (an operating work index above the laboratory index means
    the circuit is inefficient, or the ore has changed).
    """
    require_dimensionality(specific_energy, "specific_energy_mass", "specific_energy")
    f = float(f80.to("um").magnitude)
    p = float(p80.to("um").magnitude)
    _check_sizes(f, p)
    denom = math.sqrt(BOND_REFERENCE_P80_UM) * (p ** -0.5 - f ** -0.5)
    if denom <= 0.0:
        raise ValueError(
            "f80 and p80 are too close to invert Bond's equation: the size-reduction "
            "term is zero to numerical precision"
        )
    unit = convention.energy_unit
    w = float(specific_energy.to(unit).magnitude)
    return Q_(w / denom, unit)


def bond_wi_from_grindability(
    grindability_g_per_rev: float,
    closing_screen: Qty,
    f80: Qty,
    p80: Qty,
    alpha: float = BOND_GRINDABILITY_ALPHA,
    beta: float = BOND_GRINDABILITY_BETA,
    gamma: float = BOND_GRINDABILITY_GAMMA,
) -> Qty:
    """Bond's laboratory grindability equation (Eq. E2), in kWh per short ton.

    Parameters
    ----------
    grindability_g_per_rev
        ``G``, grams of closing-screen undersize produced per mill revolution at
        a 250 percent circulating load. Must be positive.
    closing_screen
        ``p_i``, the closing screen aperture.
    f80, p80
        Feed and product 80 percent passing sizes of the locked-cycle test.
    alpha, beta, gamma
        Bond's constants. Defaults are the standard-mill values reported by
        Arellano-Pina et al. 2023. Override them ONLY with constants re-fitted
        for the mill actually used; see LIMITATIONS item 3.

    Notes
    -----
    The result is returned in ``kWh/short_ton`` because that is the convention
    :math:`\\gamma = 44.5` was fitted in (44.5 lb of ball charge, imperial
    throughout). Converting afterwards is explicit and auditable; returning a
    bare number would not be.
    """
    if grindability_g_per_rev <= 0.0:
        raise ValueError(f"grindability must be positive, got {grindability_g_per_rev}")
    pi_um = float(closing_screen.to("um").magnitude)
    if pi_um <= 0.0:
        raise ValueError("closing screen aperture must be positive")
    f = float(f80.to("um").magnitude)
    p = float(p80.to("um").magnitude)
    _check_sizes(f, p)
    size_term = math.sqrt(BOND_REFERENCE_P80_UM) * (p ** -0.5 - f ** -0.5)
    if size_term <= 0.0:
        raise ValueError("size reduction term is non-positive; check f80 and p80")
    wi = gamma / (pi_um ** alpha * grindability_g_per_rev ** beta * size_term)
    return Q_(_positive_energy(wi, "bond_wi_from_grindability"), "kWh/short_ton")


def rittinger_specific_energy(
    constant: Qty, f80: Qty, p80: Qty
) -> Qty:
    """Rittinger's law, Eq. E3 with n = 2: energy proportional to new surface.

    ``constant`` must carry units of specific energy times length, e.g.
    ``kWh*um/metric_ton``, so that the product with a reciprocal length is a
    specific energy. The returned unit follows from the constant's unit.
    """
    f = float(f80.to("um").magnitude)
    p = float(p80.to("um").magnitude)
    _check_sizes(f, p)
    out = constant * Q_(1.0 / p - 1.0 / f, "1/um")
    require_dimensionality(out, "specific_energy_mass", "rittinger result")
    _positive_energy(float(out.magnitude), "rittinger_specific_energy")
    return cast(Qty, out)


def kick_specific_energy(
    constant: Qty, f80: Qty, p80: Qty
) -> Qty:
    """Kick's law, Eq. E3 with n = 1: energy proportional to the size reduction ratio.

    ``constant`` carries specific-energy units directly, because
    :math:`\\ln(F_{80}/P_{80})` is dimensionless.
    """
    require_dimensionality(constant, "specific_energy_mass", "constant")
    f = float(f80.to("um").magnitude)
    p = float(p80.to("um").magnitude)
    _check_sizes(f, p)
    out = cast(Qty, constant * math.log(f / p))
    _positive_energy(float(out.magnitude), "kick_specific_energy")
    return out


def walker_specific_energy(
    constant: Qty, f80: Qty, p80: Qty, exponent: float
) -> Qty:
    """Generalised Walker integral (Eq. E3) for arbitrary ``n``.

    Returns the integral :math:`\\int_{P}^{F} C x^{-n} dx` evaluated with the
    sign convention that energy is positive for size reduction. Provided so the
    three named laws can be exercised as one family, and so a fitted
    non-classical exponent (Hukki's observation that ``n`` drifts with size) can
    be used without a new function.
    """
    f = float(f80.to("um").magnitude)
    p = float(p80.to("um").magnitude)
    _check_sizes(f, p)
    if exponent <= 0.0:
        raise ValueError(f"exponent must be positive, got {exponent}")
    if abs(exponent - 1.0) < 1e-12:
        # n = 1 gives a dimensionless logarithm, so no length unit is attached.
        out = cast(Qty, constant * math.log(f / p))
    else:
        m = exponent - 1.0
        factor = (p ** -m - f ** -m) / m
        out = cast(Qty, constant * Q_(factor, f"um**{-m}"))
    _positive_energy(float(out.magnitude), "walker_specific_energy")
    return out


def recommended_law(p80: Qty) -> tuple[SizeLaw, str]:
    """Conventional law choice for a target product size, with the caveat text.

    Boundaries follow standard practice as set out in Wills and Finch 2016,
    Comminution. They are conventions, not measured transitions: Austin 1973
    argues the exponent drifts continuously, so a product size near a boundary
    has no single correct law.
    """
    require_dimensionality(p80, "length", "p80")
    p_um = float(p80.to("um").magnitude)
    if p_um > 50_000.0:
        return SizeLaw.KICK, (
            "coarse crushing (P80 above 50 mm): Kick conventionally applies, energy "
            "scaling with the reduction ratio"
        )
    if p_um >= 50.0:
        return SizeLaw.BOND, (
            "crushing to conventional ball milling (P80 50 um to 50 mm): Bond's third "
            "theory is the standard choice and is the calibration domain of the "
            "laboratory work index test"
        )
    return SizeLaw.RITTINGER, (
        "fine and ultrafine grinding (P80 below 50 um): Rittinger conventionally "
        "applies, energy scaling with created surface. Bond's index measured at a "
        "coarser closing screen should not be extrapolated here"
    )


def convert_ton_convention(
    specific_energy: Qty, to: TonConvention
) -> Qty:
    """Convert a specific energy between short-ton and metric-tonne bases (Eq. E4).

    Examples
    --------
    7.2925 kWh/short ton is 7.2925 x (1000/907.18474) = 8.0386 kWh/metric tonne.
    The metric figure is larger because the reference mass is larger.

    >>> q = convert_ton_convention(Q_(7.2925, "kWh/short_ton"), TonConvention.METRIC)
    >>> round(q.magnitude, 4)
    8.0386
    """
    require_dimensionality(specific_energy, "specific_energy_mass", "specific_energy")
    return specific_energy.to(to.energy_unit)


def percent_error(reference: float, approximation: float) -> float:
    """Absolute percentage error, Eq. 13 of Arellano-Pina et al. 2023.

    .. math::
        \\%\\,\\mathrm{error} = \\left| \\frac{W_{i,\\mathrm{ref}} -
        W_{i,\\mathrm{approx}}}{W_{i,\\mathrm{ref}}} \\right| \\times 100
    """
    if reference == 0.0:
        raise ValueError("percent error is undefined against a zero reference")
    return abs((reference - approximation) / reference) * 100.0


# --- ore-type priors --------------------------------------------------------
#
# Honest accounting of what is and is not sourced here.
#
# SOURCED: one measured Bond ball mill work index on a quartz-rich ore, namely
# Ore A of Arellano-Piña et al. 2023 (28.75 wt% Si, 7.50 wt% Al, 3.1 wt% K, so a
# quartz plus aluminosilicate assemblage, not a quartz concentrate), 12.3 kWh
# per metric tonne by the standard Bond procedure. That is the only Bond work
# index on a silica-dominated ore this environment could retrieve from a
# peer-reviewed, DOI-bearing source.
#
# NOT SOURCED, and deliberately not used: the widely reproduced "Table of Bond
# Work Index by Minerals" figures for quartz (13.57), quartzite (12.18) and
# silica sand (16.46) kWh per short ton. These circulate on trade websites and
# file-sharing sites as transcriptions of Bond's 1961 Allis-Chalmers tables. The
# primary source (Bond F. C. 1961, Crushing and Grinding Calculations,
# Allis-Chalmers) was not retrievable in this environment, and a tier 3 web
# compilation may not be sole evidence under the platform's evidence standard.
# They are recorded below as ASSUMED with that basis stated, not as SOURCED.

_WI_ORE_A: Final[Value] = Value(
    quantity=Q_(12.3, "kWh/metric_ton"),
    tag=Tag.SOURCED,
    source=SRC_PPMP_BOND,
    dist=Distribution.relative_uniform(0.95, 1.05),
    confidence="medium",
    basis=(
        "Table 2, Ore A, standard Bond procedure in the 30.5 cm Bond ball mill. "
        "Ore A is a quartz plus aluminosilicate assemblage from Zacatecas, Mexico "
        "(28.75 wt% Si), not a quartz concentrate. Relative uncertainty of "
        "plus or minus 5 percent reflects the 3.8 to 5.3 percent spread the same "
        "paper reports between its standard and reduced procedures. UNIFORM, not "
        "normal: the spread is the gap between two named procedures, which bounds "
        "the value rather than describing a sampling distribution around it, and "
        "the paper reports no replicate scatter that would justify a Gaussian. "
        "This call was originally written against Distribution.relative(0.05), "
        "which left the shape implicit; that constructor was split into "
        "relative_normal and relative_uniform precisely so the shape has to be "
        "stated, because the implicit version also misread an absolute zero-mean "
        "normal as a relative one."
    ),
)

#: Bond work index priors by ore type. Exactly one entry is SOURCED.
_WORK_INDEX_PRIORS: Final[dict[OreType, Value]] = {
    OreType.VEIN_QUARTZ: Value(
        quantity=Q_(13.6, "kWh/short_ton"),
        tag=Tag.ASSUMED,
        dist=Distribution(kind="uniform", low=11.0, high=17.0),
        confidence="low",
        basis=(
            "ESTIMATE, not a measurement. Centred on the 13.57 kWh/short ton figure "
            "for quartz that appears in secondary transcriptions of Bond 1961, "
            "Crushing and Grinding Calculations (Allis-Chalmers), which could not be "
            "verified against the primary source in this environment and would be a "
            "tier 3 citation if used directly. Bracketed 11 to 17 kWh/short ton to "
            "span the quartz, quartzite and silica sand entries of those same tables. "
            "Corroborated only loosely by the one retrievable peer-reviewed "
            "measurement on a quartz-rich ore, 12.3 kWh/metric tonne (11.16 kWh/short "
            "ton) for Ore A of Arellano-Pina et al. 2023, doi:10.37190/ppmp/172458. "
            "Replace with a measured Bond ball mill index on AE-Q samples."
        ),
    ),
    OreType.QUARTZITE: Value(
        quantity=Q_(12.2, "kWh/short_ton"),
        tag=Tag.ASSUMED,
        dist=Distribution(kind="uniform", low=10.0, high=16.0),
        confidence="low",
        basis=(
            "ESTIMATE. Same unverified Bond 1961 secondary transcriptions (quartzite "
            "12.18 kWh/short ton), same caveat as vein quartz. Quartzite is expected "
            "softer than vein quartz in grinding because breakage follows the "
            "cemented grain boundaries rather than the quartz itself, but this module "
            "has no measurement supporting the direction or the size of that "
            "difference."
        ),
    ),
    OreType.PEGMATITE_QUARTZ: Value(
        quantity=Q_(13.6, "kWh/short_ton"),
        tag=Tag.ASSUMED,
        confidence="low",
        basis=(
            "ESTIMATE, set equal to the vein quartz prior because no work index "
            "measurement specific to pegmatite quartz was retrievable. Pegmatite "
            "quartz is coarse-grained and hosts feldspar and mica, which are softer "
            "than quartz, so the true index is plausibly lower; that is a hypothesis, "
            "not a sourced number."
        ),
    ),
    OreType.INDUSTRIAL_SAND: Value(
        quantity=Q_(16.5, "kWh/short_ton"),
        tag=Tag.ASSUMED,
        confidence="low",
        basis=(
            "ESTIMATE. The silica sand entry (16.46 kWh/short ton) of the same "
            "unverified Bond 1961 transcriptions. Higher than vein quartz there, "
            "which is consistent with a sand feed already being fine so that further "
            "size reduction sits in the Rittinger rather than the Bond regime, where "
            "a Bond index measured at a coarse closing screen understates energy. "
            "Treat any Bond calculation on a sand feed as suspect on that ground "
            "alone (see LIMITATIONS item 2)."
        ),
    ),
}


def work_index_prior(ore_type: OreType) -> Value:
    """Bond work index prior for an ore class, as a provenance-tagged Value.

    Raises
    ------
    KeyError
        For an ore type with no prior at all, rather than returning a guess.
        Alaskite, novaculite, hydrothermal crystal and dolomite are in that
        position: no retrievable Bond index, and no defensible analogue.
    """
    if ore_type not in _WORK_INDEX_PRIORS:
        raise KeyError(
            f"no Bond work index prior for {ore_type.value}: no peer-reviewed "
            f"measurement was retrievable and no defensible analogue exists. "
            f"Priors available for {sorted(o.value for o in _WORK_INDEX_PRIORS)}. "
            f"Measure it (Bond locked-cycle test, about 5 hours per sample)."
        )
    return _WORK_INDEX_PRIORS[ore_type]


def sourced_quartz_rich_benchmark() -> Value:
    """The one SOURCED Bond work index on a quartz-rich ore, for benchmarking."""
    return _WI_ORE_A


# --- FEEDSTOCK and SITE entry points ----------------------------------------
#
# The functions above are the equations, and take quantities. The functions
# below are the MODELS, and take FEEDSTOCK and SITE. That split is deliberate
# and is not a way round the platform rule: no model here holds an ore property
# as a module constant, and every ore property is read off the Feedstock or
# comes from work_index_prior with its provenance attached.


def feedstock_work_index(
    feedstock: Feedstock, convention: TonConvention, allow_prior: bool = False
) -> Value:
    """Work index for a feedstock: measured if present, else an explicit prior.

    Parameters
    ----------
    feedstock
        The ore. ``feedstock.physical.bond_work_index`` is used when it has been
        measured.
    convention
        Ton basis for the returned value.
    allow_prior
        When the feedstock has no measured index, ``False`` (the default) raises
        and ``True`` falls back to :func:`work_index_prior` with its ASSUMED tag
        intact. The default is to raise because a mill sized on an assumed work
        index for an uncharacterized deposit is a scenario, and the caller
        should have to say so.

    Raises
    ------
    ValueError
        If the index is unmeasured and ``allow_prior`` is False.
    """
    measured = feedstock.physical.bond_work_index
    if isinstance(measured, Value):
        q = _q(measured)
        require_dimensionality(q, "specific_energy_mass", "bond_work_index")
        return Value(
            quantity=q.to(convention.energy_unit),
            tag=measured.tag,
            source=measured.source,
            dist=measured.dist,
            basis=measured.basis,
            confidence=measured.confidence,
        )
    if not allow_prior:
        raise ValueError(
            f"{feedstock.sample_id} has no measured Bond work index. Pass "
            f"allow_prior=True to substitute the ASSUMED ore-type prior, which makes "
            f"every downstream energy a scenario rather than a result."
        )
    prior = work_index_prior(feedstock.ore_type)
    return Value(
        quantity=_q(prior).to(convention.energy_unit),
        tag=Tag.ASSUMED,
        dist=prior.dist,
        confidence=prior.confidence,
        basis=(
            f"Ore-type prior for {feedstock.ore_type.value} substituted because "
            f"{feedstock.sample_id} has no measured Bond work index. "
            f"Underlying basis: {prior.basis}"
        ),
    )


def grinding_specific_energy(
    feedstock: Feedstock,
    p80: Qty,
    convention: TonConvention,
    f80: Qty | None = None,
    allow_prior: bool = False,
    law: SizeLaw | None = None,
) -> tuple[Qty, Value, str]:
    """Net specific grinding energy for one feedstock to a target product size.

    Parameters
    ----------
    feedstock
        Ore. Supplies the work index and, if ``f80`` is omitted, the feed size
        from ``feedstock.physical.f80``.
    p80
        Target product 80 percent passing size.
    convention
        Ton basis.
    f80
        Feed size override, for a circuit whose feed is set by an upstream
        crusher rather than by the ore.
    allow_prior
        Forwarded to :func:`feedstock_work_index`.
    law
        Force a law. ``None`` uses :func:`recommended_law` and returns its note.

    Returns
    -------
    (energy, work_index_value, note)
        ``note`` records the law used and any caveat, so a caller writing a
        report cannot silently drop the provenance.

    Raises
    ------
    ValueError
        If the feed size is neither supplied nor measured, or if a law other
        than Bond is requested (Kick and Rittinger need their own fitted
        constants, which this module does not have for any quartz ore).
    """
    if f80 is None:
        measured_f80 = feedstock.physical.f80
        if isinstance(measured_f80, _Missing):
            raise ValueError(
                f"{feedstock.sample_id} has no measured f80 and none was supplied. "
                f"A run-of-mine top size is not an f80; measure the feed size "
                f"distribution or pass the crusher product f80 explicitly."
            )
        f80 = measured_f80.quantity
    wi = feedstock_work_index(feedstock, convention, allow_prior=allow_prior)
    chosen, note = recommended_law(p80)
    if law is not None and law is not chosen:
        note = (
            f"law forced to {law.value} by the caller; the conventional choice for "
            f"this product size is {chosen.value} ({note})"
        )
        chosen = law
    if chosen is not SizeLaw.BOND:
        raise ValueError(
            f"{chosen.value} is the conventional law for this product size, but this "
            f"module has no fitted {chosen.value} constant for any quartz ore, so it "
            f"will not fabricate one. Either grind to a P80 in the Bond domain "
            f"(50 um to 50 mm) or measure the constant. Note: {note}"
        )
    energy = bond_specific_energy(wi.quantity, f80, p80, convention)
    return energy, wi, note


def grinding_energy_cost(
    feedstock: Feedstock,
    site: Site,
    p80: Qty,
    throughput: Qty,
    convention: TonConvention,
    f80: Qty | None = None,
    allow_prior: bool = False,
    motor_efficiency: float = 1.0,
) -> tuple[Qty, Qty, str]:
    """Grinding electricity cost for a feedstock at a site.

    Parameters
    ----------
    feedstock, site
        Ore and location. The energy price and its currency come from
        ``site.power.energy_price``; nothing about the plant location is
        hardcoded.
    p80, f80, convention, allow_prior
        As :func:`grinding_specific_energy`.
    throughput
        Ore mass rate, e.g. ``Q_(50.0, "tonne/hour")``.
    motor_efficiency
        Fraction in (0, 1]. Divides the net pinion energy to give the energy
        drawn from the grid. The default of 1.0 returns net pinion energy and is
        an underestimate of plant draw; see LIMITATIONS item 5.

    Returns
    -------
    (cost_rate, power, note)
        Cost per unit time in the site's currency, and the electrical power.
    """
    if not 0.0 < motor_efficiency <= 1.0:
        raise ValueError(f"motor_efficiency must lie in (0, 1], got {motor_efficiency}")
    require_dimensionality(throughput, "mass_flow", "throughput")
    energy, _wi, note = grinding_specific_energy(
        feedstock, p80, convention, f80=f80, allow_prior=allow_prior
    )
    power = cast(Qty, (energy * throughput / motor_efficiency).to("kW"))
    price = _q(site.power.energy_price)
    if "kWh" not in str(price.units) and "kilowatt_hour" not in str(price.units):
        raise ValueError(
            f"site {site.site_id} prices energy in {price.units}; grinding cost needs "
            f"a price per kWh so the conversion is explicit"
        )
    cost_rate = (power * price).to(f"{site.currency.value}/hour")
    if float(cost_rate.magnitude) < 0.0:
        raise AssertionError("negative energy cost is unphysical")
    note = (
        f"{note}. Motor and drive efficiency {motor_efficiency:.2f}; classifier, "
        f"conveying and dust-collection loads excluded. Energy price basis: "
        f"{site.power.rate_basis}."
    )
    return cost_rate, power, note
