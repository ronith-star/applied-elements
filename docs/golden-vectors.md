# Golden test vectors

A golden vector in this suite is a hand-traceable worked example: stated inputs,
every intermediate step with its arithmetic, the expected output, and a tolerance
with a stated reason. Someone with a calculator must be able to reproduce it from
this document alone, without running the code. That is what makes it a check on
the model rather than a snapshot of current behaviour.

A regression vector is the other kind. Where an expected value can only come from
the code (a seeded Monte Carlo draw, a cross-validated ridge fit), the file says
so, calls itself a regression vector, and is counted separately. Three of the
thirty files here are regression vectors. Two of those three (RV-02, RV-03) derive
most of their expected values by hand and carry the closed-form answer alongside
the seeded one; they are still classified as regression, because the strictest
classification is the honest one.

## Counts

| kind | count | derivation |
| --- | --- | --- |
| golden | 27 | all independent |
| regression | 3 | 2 mixed (RV-02, RV-03), 1 from_code (RV-01) |
| total | 30 | 27 independently derived |

The thirty files carry 224 checked outputs and 224 tolerance bounds.

Independently derived means the expected value was computed by hand, or by a
script that imports nothing from `ae`, and then written into the file. It was not
obtained by calling the function and pasting the answer. Every intermediate step
of every such derivation is reproduced below.

## How to audit one vector

1. Read the `inputs` block of the YAML file in `data/golden/`.
2. Follow the arithmetic chain in this document with a calculator.
3. Compare your result against the `expected` block.
4. Read the tolerance reason and decide whether you accept it.

Step 4 is the one that matters. A tolerance with no stated reason is a fudge
factor, so every tolerance in this suite names the operation count that sets its
floating point bound, or the measured Monte Carlo scatter, or the solver stopping
criterion it inherits. Where a tolerance was set by measurement rather than by
analysis, the measurement is quoted.

Three tolerance kinds appear, and the distinction is load-bearing. A floating
point tolerance bounds the rounding of a stated number of operations and is
derived by counting them. A Monte Carlo tolerance bounds observed seed-to-seed
scatter and is derived by measuring it across seeds, which is quoted per seed. A
solver tolerance is inherited from the stopping criterion of the iteration that
produced the number, and is never tighter than that criterion.

## What the loader enforces

`tests/golden/test_golden_vectors.py` loads every file and checks five structural
properties, each with its own test and each verified by control (the defect was
injected, the guard confirmed to fire, the defect removed, the guard confirmed to
pass):

1. Every expected output key must be present in the dispatch result, so a vector
   naming an output the code does not produce fails on the missing key rather
   than passing vacuously.
2. Every expected output must carry a tolerance, every tolerance must state
   either an absolute or a relative bound but never both, and every reason must
   be substantive.
3. Every tolerance bound must parse as a number rather than as a string.
4. Every file must declare its kind, and a regression file must state why its
   values could only come from the code.
5. A golden file must declare `derivation: independent`. A file whose expected
   values came from running the code cannot be golden, whatever it is called.

`tests/golden/checks.py` holds the dispatch layer. Two rules govern it: no
function there may reproduce a formula the module under test also implements, and
where a second route to the same number is wanted it uses the module's own forward
function, so a wrong formula cannot be cancelled by the same wrong formula written
twice.

## Index

| id | title | kind | module | outputs |
| --- | --- | --- | --- | --- |
| GV-01 | Bond specific energy for a 12000 to 106 micron grind | golden | `ae.physics.comminution` | 2 |
| GV-02 | Operating work index inverted from measured mill power | golden | `ae.physics.comminution` | 2 |
| GV-03 | Rosin-Rammler quantiles, Sauter mean and geometric surface area | golden | `ae.physics.psd` | 8 |
| GV-04 | Log-normal PSD median conversions by weighting, Sauter mean and span | golden | `ae.physics.psd` | 7 |
| GV-05 | Furnas ternary maximum packing and optimal volume composition | golden | `ae.physics.packing` | 9 |
| GV-06 | Krieger-Dougherty relative viscosity and volume to mass fraction conversion | golden | `ae.physics.packing` | 2 |
| GV-07 | Inclusion exposure, enclosed fraction and the liberation size that follows | golden | `ae.physics.liberation` | 4 |
| GV-08 | Leachable fraction of an impurity from its partition and two inclusion sizes | golden | `ae.physics.liberation` | 4 |
| GV-09 | Impurity floor and removable inventory from a four-way partition | golden | `ae.physics.impurity_location` | 8 |
| GV-10 | Implied removal rate and the SiO2 percent ceiling a residual trace sum sets | golden | `ae.physics.impurity_location` | 4 |
| GV-11 | Shrinking-core characteristic times in all three rate-controlling regimes | golden | `ae.physics.leaching` | 3 |
| GV-12 | Conversion at a fixed dimensionless time in three regimes, with the g(X) round trip | golden | `ae.physics.leaching` | 9 |
| GV-13 | Characteristic time back-calculated from one timed leach assay | golden | `ae.physics.leaching` | 2 |
| GV-14 | Chloride vapour pressures at 400 K by one-point Clausius-Clapeyron, and the separation they imply | golden | `ae.physics.chlorination` | 15 |
| GV-15 | Gibbs energy of a stated reaction from a supplied thermochemical table | golden | `ae.physics.chlorination` | 1 |
| GV-16 | Sensible heat to take cristobalite from 1200 K to 1700 K | golden | `ae.physics.thermal` | 2 |
| GV-17 | Alpha quartz heat capacity at 700 K, base plus Landau excess, and their sum | golden | `ae.physics.thermal` | 4 |
| GV-18 | Flotation first-order recovery, two-product mass balance and a partition curve | golden | `ae.physics.separation` | 6 |
| GV-19 | Two-stage flowsheet with a cleaner reject recycle, solved to a fixed point by hand | golden | `ae.plant.streams` | 16 |
| GV-20 | OEE, loss decomposition, and the bottleneck of a two-unit line | golden | `ae.plant.capacity` | 17 |
| GV-21 | Four-stage yield cascade and the throughput factor each stage must carry | golden | `ae.plant.yield_cascade` | 5 |
| GV-22 | Cpk from six lot assays, its confidence interval, and the off-spec fraction under two tail models | golden | `ae.plant.yield_cascade` | 9 |
| GV-23 | M/M/1 and Allen-Cunneen waiting time, and the reduction the queueing approximation must satisfy | golden | `ae.plant.scheduling` | 3 |
| GV-24 | Cash cost per tonne of product, with the cascade yield dividing the feed-basis inputs | golden | `ae.econ.unit_economics` | 10 |
| GV-25 | Factored capital estimate with escalation, location factor and an AACE Class 5 band | golden | `ae.econ.capex` | 12 |
| GV-26 | NPV, IRR, payback, levelized cost and breakeven price on a five-period project | golden | `ae.econ.valuation` | 17 |
| GV-27 | Sobol indices of a linear additive function, checked against the exact variance decomposition | golden | `ae.econ.uncertainty` | 13 |
| RV-01 | Seeded Monte Carlo percentiles of a uniform variable, with the analytic quantile alongside | regression | `ae.econ.uncertainty` | 8 |
| RV-02 | Leave-one-deposit-out folds on a six-sample three-group set, with the mean baseline derived by hand | regression | `ae.ml.surrogate` | 18 |
| RV-03 | EVPI on a two-action problem whose closed-form answer is 1/16, showing the nested-Monte-Carlo bias | regression | `ae.agent.decisions` | 4 |

## Control results

Eleven defects were injected into the loader one at a time, each confirmed to be
reported, then removed and confirmed to pass. The script is
`tests/golden/control.py`, runnable from the repository root. Measured output:
eleven defects injected, eleven reported, then fourteen clean checks with zero
failures.

| defect injected | guard that reported it |
| --- | --- |
| expected value perturbed by 1e-8 against a 1e-9 tolerance | `test_vector_outputs_match_expected` |
| vector names an output the dispatch does not produce | `test_vector_outputs_match_expected` |
| tolerance whose reason is two characters | `test_every_expected_output_has_a_reasoned_tolerance` |
| expected output with no tolerance entry at all | `test_every_expected_output_has_a_reasoned_tolerance` |
| golden vector declaring `derivation: from_code` | `test_vector_declares_its_kind_and_provenance` |
| regression vector with a nine-character reason | `test_vector_declares_its_kind_and_provenance` |
| provenance entry naming none of the five tags | `test_every_provenance_tag_is_one_of_the_five` |
| arithmetic block of 41 characters | `test_vector_declares_its_kind_and_provenance` |
| tolerance giving both `abs` and `rel` | `_tolerance_of` |
| vector naming the wrong bottleneck (missing-key path) | `test_vector_outputs_match_expected` |
| suite shape assertion against a one-vector suite | `test_the_suite_has_the_declared_shape` |

The count is reported rather than a pass or fail verdict because a control whose
partial failure looks like success is worse than no control: had nine of eleven
been caught, printing only the first result would have read as success.

## Defects this suite found in its own construction

Recorded because being right is not the same as having checked.

1. Every tolerance bound was written with `repr()`, which emits an exponent form
   with no decimal point in the mantissa, and PyYAML's float resolver requires
   one. All 174 exponent-form bounds therefore parsed as strings rather than
   floats. A `float()` coercion inside the loader's tolerance helper made every
   test pass regardless, so the suite reported 103 passes while three quarters of
   its tolerances were the wrong type. The coercion was removed, the bounds
   rewritten in the form 1.0e-9, and two tests added: one asserting the type per
   vector, one pinning the count at 174 of 224. This is the worst defect found,
   because the masking mechanism was a convenience coercion that looked harmless.
2. `sobol_additive` in the dispatch layer tested `not diagnostics()` as its pass
   condition. `diagnostics()` returns a mapping of named flags and is never
   empty, so the expression was always false and every Sobol run was reported as
   failing its own diagnostics. The pass condition is now the conjunction of the
   two sum checks, the absence of materially negative indices, the absence of a
   first-order index exceeding its total, and the converged flag.
3. GV-14 asserted a Trouton enthalpy of 34816.0 J/mol for TiCl4. TiCl4 carries a
   MEASURED PubChem heat of vaporization of 36.2 kJ/mol in the module, so the
   Trouton estimate is not used for it and the predicted pressure is
   78509.86077995766 Pa rather than 79279.34025051698 Pa, a 1.0 percent
   difference. The output key was renamed from `trouton_enthalpy` to
   `transition_enthalpy` and a second output, `enthalpy_is_trouton`, added so the
   file records which basis is in use per species. Calling a measurement an
   estimate is the kind of error a provenance tag exists to prevent.
4. GV-27 asserted a worst Sobol index deviation of 0.000313 against seed 0. That
   figure is the worst across seeds 0 to 4 and belongs to seed 4; seed 0's own
   worst deviation is 1.8216682789187755e-05. Both are now measured and asserted
   separately.
5. Twelve vectors carried expected values rounded to ten significant digits while
   their tolerances sat at the floating point floor of 1e-12, so the rounding of
   the literal exceeded the tolerance by one to two orders. The expected values
   now carry full double precision. This is the failure mode where a tolerance
   chosen on analysis collides with a literal written for readability, and the
   right fix is the literal, not the tolerance.
6. Sixty-one tolerance reasons were written as bare cross-references of the form
   "As <other key>", which are not self-contained and fell below the loader's own
   substance floor. They were expanded so each reason stands alone.
7. `_leach_system` built its fixture `Feedstock` with `characterized=True`, which
   the model rejects because a one-element profile sits at tier `screened` while
   the flag requires tier `located`. The flag is now False, which is also the
   honest value: the fixture carries no real measurement.
8. `g_of_conversion` returns a one-element ndarray for a scalar argument, so
   `float(np.asarray(...))` raised. Fixed with `.ravel()[0]`.
9. Five dispatch calls were written against signatures that do not exist:
   `removable_ppm` takes no `efficiencies`, `krieger_dougherty_relative_viscosity`
   takes a provenance `Value` and not a float, `volume_to_mass_fraction` names its
   first parameter `phi`, `Flowsheet` has `connect` and not `link`, and
   `max_feasible_feed_fraction` takes an element name and not a removal fraction.

## On the Vikarabad deposit

No number in this suite is a property of the Vikarabad ore. That deposit is
uncharacterized in the citable record: no whole-rock assay, no LA-ICP-MS trace
element suite, no fluid inclusion study and no partition measurement for it exists
to cite. Every impurity concentration, partition fraction and removal efficiency
in these files is tagged ASSUMED and is a scenario gated on a future assay
campaign. The fixture `Source` object carried by the dispatch layer says so in its
own note, and its URL is deliberately `example.invalid` so it cannot be mistaken
for a reference.

---

## GV-01: Bond specific energy for a 12000 to 106 micron grind

- File: `data/golden/GV-01-bond-specific-energy.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.comminution`
- Dispatch entry: `bond_energy`

**Reference.** Bond, F.C. 1952, The Third Theory of Comminution, Transactions AIME 193:484-494. The 10 kWh/short_ton * um^0.5 coefficient is Bond's own, defined on a 100 um reference product, so sqrt(100) = 10.

**Inputs.**

```yaml
work_index: 12.7
work_index_unit: kWh/short_ton
f80_um: 12000.0
p80_um: 106.0
convention: short
```

**Provenance.**

- `work_index`: ASSUMED, 12.7 kWh/short_ton, a round mid-range quartzite figure chosen to make the
  arithmetic traceable, not a measurement on Vikarabad ore, which is uncharacterized
  in the citable record. No uncertainty is quoted because it is not a measurement.
- `f80_um`: ASSUMED, 12000 um, a nominal primary-crusher product.
- `p80_um`: ASSUMED, 106 um, a standard sieve aperture (140 mesh).
- `convention`: SOURCED, short ton, Bond's own basis (Bond 1952).

**Arithmetic chain.**

```
1/sqrt(106)   = 0.09712858623572641
1/sqrt(12000) = 0.00912870929175277
difference    = 0.08799987694397364
W = 10 * 12.7 * 0.08799987694397364 = 11.175984371884752 kWh/short_ton
metric: 11.175984371884752 * 1000 / 907.18474 = 12.31941398380887 kWh/tonne
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `specific_energy_kwh_per_short_ton` | 11.1759843719 | abs 1e-09 | Floating point only. The chain is two square roots, one subtraction and two multiplications in IEEE 754 double precision, so relative error is bounded at a few units in the last place, about 1e-15 relative or 1e-14 absolute here. 1e-9 absolute is six orders looser than that bound and still four orders tighter than the last digit quoted, so it detects any real formula change (a wrong reference-size coefficient shifts the answer by percent, not by ppb) while never firing on arithmetic reassociation. |
| `specific_energy_kwh_per_metric_tonne` | 12.3194139838 | abs 1e-09 | Floating point only, as above, plus one division by the exact constant 907.18474 kg per short ton. Exact constants introduce no additional error beyond one rounding. |

---

## GV-02: Operating work index inverted from measured mill power

- File: `data/golden/GV-02-bond-inversion.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.comminution`
- Dispatch entry: `bond_inversion`

**Reference.** Bond, F.C. 1952, The Third Theory of Comminution, Transactions AIME 193:484-494. Inversion of the same relation, dividing by the identical size term.

**Inputs.**

```yaml
specific_energy: 14.6
energy_unit: kWh/tonne
work_index_unit: kWh/tonne
f80_um: 2000.0
p80_um: 150.0
convention: metric
```

**Provenance.**

- `specific_energy`: ASSUMED, 14.6 kWh/tonne, a plausible measured mill draw per tonne. Treated as exact
  for this vector because the vector tests arithmetic, not a measurement uncertainty.
- `f80_um`: ASSUMED, 2000 um.
- `p80_um`: ASSUMED, 150 um (100 mesh).
- `convention`: ASSUMED, metric, matching the stated energy unit. The module raises on a convention
  mismatch rather than converting, which is the 10.23 percent silent error it exists
  to prevent.

**Arithmetic chain.**

```
1/sqrt(150)  = 0.08164965809277261
1/sqrt(2000) = 0.02236067977499790
difference   = 0.05928897831777471
denominator  = 10 * 0.05928897831777471 = 0.5928897831777471
Wi_op = 14.6 / 0.5928897831777471 = 24.625150262699417 kWh/tonne
Round trip: 10 * 24.625150262699417 * 0.05928897831777471 = 14.6 exactly by construction.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `operating_work_index` | 24.6251502627 | abs 1e-09 | Floating point only. One division by a value formed from two square roots. Double precision bounds the relative error near 1e-15; 1e-9 absolute is far looser than that and far tighter than any formula error. |
| `round_trip_specific_energy` | 14.6 | abs 1e-12 | A round trip through the module's forward function must return the input to within accumulated rounding, which for one division followed by one multiplication by the same denominator is at most a few ulp, near 1e-14 absolute at this magnitude. 1e-12 admits that and nothing larger. The check is meaningful because the forward and inverse are separate code paths: a sign or coefficient error in one and not the other breaks it. |

---

## GV-03: Rosin-Rammler quantiles, Sauter mean and geometric surface area

- File: `data/golden/GV-03-rosin-rammler.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.psd`
- Dispatch entry: `rosin_rammler`

**Reference.** Rosin, P. and Rammler, E. 1933, The Laws Governing the Fineness of Powdered Coal, Journal of the Institute of Fuel 7:29-36. Gamma function values below are from the standard series, computed here by the closed-form relations stated in the arithmetic block, not read from the module.

**Inputs.**

```yaml
d_prime_um: 45.0
n: 2.6
density_kg_m3: 2650.0
```

**Provenance.**

- `d_prime_um`: ASSUMED, 45 um, a round characteristic size. Not a measurement.
- `n`: ASSUMED, 2.6, within the 2 to 4 band typical of closed-circuit ball mill products.
  Chosen above 1 because the Sauter mean of a Rosin-Rammler distribution diverges at n
  <= 1 and the module raises there.
- `density_kg_m3`: SOURCED, 2650 kg/m^3, the accepted density of alpha quartz, quoted to three
  significant figures. Uncertainty is not propagated in this vector because the
  surface area scales exactly inversely with it.

**Arithmetic chain.**

```
Quantile: d_p = d' * (-ln(1-p))^(1/n),  1/n = 0.3846153846153846
  p = 0.10: -ln(0.90) = 0.10536051565782628
             0.10536051565782628^0.3846153846153846 = 0.4208301818844526
             d10 = 45 * 0.4208301818844526 = 18.937358184800367 um
  p = 0.50: ln 2 = 0.6931471805599453
             0.6931471805599453^0.3846153846153846 = 0.8685183997017331
             d50 = 45 * 0.8685183997017331 = 39.083327986577984 um
  p = 0.90: -ln(0.10) = 2.302585092994046
             2.302585092994046^0.3846153846153846 = 1.378204699352725
             d90 = 45 * 1.378204699352725 = 62.01921147087263 um
Span = (d90 - d10)/d50 = (62.01921147087263 - 18.937358184800367)/39.083327986577984
     = 43.081853286072266/39.083327986577984 = 1.1023076975652495
Cumulative undersize at d = d': 1 - exp(-1) = 0.6321205588285577
Sauter mean: d32 = d' / Gamma(1 - 1/n) = 45 / Gamma(0.6153846153846154)
  Gamma(0.6153846153846154) = 1.4549279883039163
  d32 = 45 / 1.4549279883039163 = 30.929365825492706 um
Volume-weighted mean: E_v[d] = d' * Gamma(1 + 1/n) = 45 * Gamma(1.3846153846153846)
  Gamma(1.3846153846153846) = 0.8882104347329064
  E_v[d] = 45 * 0.8882104347329064 = 39.96946956298079 um
Geometric surface area: S = 6/(rho * d32 * psi), psi = 1
  d32 = 30.929365825492706e-6 m
  S = 6 / (2650 * 30.929365825492706e-6) = 73.20392393981969 m^2/kg
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `d10_um` | 18.9373581848 | abs 1e-06 | Floating point plus one fractional power. The quantile is a closed form, so the only error is IEEE 754 rounding through log and pow, about 1e-15 relative, which is 2e-14 um here. 1e-6 um is eight orders looser than that bound and still eight orders tighter than any plausible formula error (dropping the 1/n exponent moves d10 by 26 um), so it separates the two cleanly. The looser 1e-6 rather than 1e-9 is used because pow with a non- dyadic exponent is permitted a slightly larger error than a single arithmetic operation. |
| `d50_um` | 39.0833279866 | abs 1e-06 | Same basis as d10_um. Floating point plus one fractional power. The quantile is a closed form, so the only error is IEEE 754 rounding through log and pow, about 1e-15 relative, which is 2e-14 um here. 1e-6 um is eight orders looser than that bound and still eight orders tighter than any plausible formula error (dropping the 1/n exponent moves d10 by 26 um), so it separates the two cleanly. The looser 1e-6 rather than 1e-9 is used because pow with a non-dyadic exponent is permitted a slightly larger error than a single arithmetic operation. |
| `d90_um` | 62.0192114709 | abs 1e-06 | Same basis as d10_um. Floating point plus one fractional power. The quantile is a closed form, so the only error is IEEE 754 rounding through log and pow, about 1e-15 relative, which is 2e-14 um here. 1e-6 um is eight orders looser than that bound and still eight orders tighter than any plausible formula error (dropping the 1/n exponent moves d10 by 26 um), so it separates the two cleanly. The looser 1e-6 rather than 1e-9 is used because pow with a non-dyadic exponent is permitted a slightly larger error than a single arithmetic operation. |
| `span` | 1.1023076976 | abs 1e-09 | A ratio of differences of three quantities each accurate to about 1e-14 um at a magnitude near 40 um, so the propagated absolute error in the dimensionless span is near 1e-15. 1e-9 admits that with six orders of margin. |
| `sauter_d32_um` | 30.9293658255 | abs 1e-06 | One Gamma function evaluation. Gamma is implemented by a series or by the Lanczos approximation, which is accurate to roughly 1e-15 relative but not to the last bit, so this quantity carries slightly more error than a plain arithmetic chain. At 31 um an error of 1e-15 relative is 3e-14 um; 1e-6 um admits that and still catches a wrong exponent sign, which would give 40 um. |
| `volume_weighted_mean_um` | 39.969469563 | abs 1e-06 | Same basis as sauter_d32_um. One Gamma function evaluation. Gamma is implemented by a series or by the Lanczos approximation, which is accurate to roughly 1e-15 relative but not to the last bit, so this quantity carries slightly more error than a plain arithmetic chain. At 31 um an error of 1e-15 relative is 3e-14 um; 1e-6 um admits that and still catches a wrong exponent sign, which would give 40 um. |
| `specific_surface_area_m2_per_kg` | 73.2039239398 | abs 1e-06 | One division of the Gamma-derived d32 into a constant. Absolute error inherits d32's relative error, so at 73 m^2/kg it is near 1e-13. 1e-6 is seven orders looser and still catches a factor-of-6 or sphericity error. |
| `cumulative_undersize_at_d_prime` | 0.6321205588285577 | abs 1e-12 | One exponential of exactly -1, so the only error is the rounding of exp(-1), bounded at 1 ulp near 1e-16. 1e-12 admits that with four orders of margin. This output is the sharpest check in the vector because 1 - exp(-1) is the definition of d': if the module scaled d' the value moves visibly. |

---

## GV-04: Log-normal PSD median conversions by weighting, Sauter mean and span

- File: `data/golden/GV-04-lognormal-psd.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.psd`
- Dispatch entry: `lognormal_psd`

**Reference.** Hatch, T. and Choate, S.P. 1929, Statistical description of the size properties of non-uniform particulate substances, Journal of the Franklin Institute 207:369-387. The moment relation d_k = d_gn exp(k s^2) used below is the Hatch-Choate form.

**Inputs.**

```yaml
d_gn_um: 8.0
sigma_g: 1.8
density_kg_m3: 2650.0
```

**Provenance.**

- `d_gn_um`: ASSUMED, 8 um COUNT median. The distinction matters: the module stores the count
  median and a caller who passes a volume median gets a distribution whose volume
  median is larger by exp(3 s^2).
- `sigma_g`: ASSUMED, 1.8. Values near 1 give a near-monodisperse powder; 1.8 is a moderately wide
  ground product. The module requires sigma_g >= 1.
- `density_kg_m3`: SOURCED, 2650 kg/m^3, alpha quartz.

**Arithmetic chain.**

```
s = ln(sigma_g) = ln(1.8) = 0.5877866649021191
s^2 = 0.345493163436756
Medians by weighting exponent k (number 0, area 2, volume 3):
  number: 8 * exp(0)                  = 8.0 um
  area:   8 * exp(2 * 0.345493163436756) = 8 * exp(0.690986326873512)
          = 8 * 1.9956829585543872 = 15.965463668435097 um
  volume: 8 * exp(3 * 0.345493163436756) = 8 * exp(1.036479490310268)
          = 8 * 2.8192742405231996 = 22.554193924185597 um
Sauter mean: d32 = d_gn exp(2.5 s^2) = 8 * exp(0.86373290859189)
          = 8 * 2.3719986419269956 = 18.975989135415965 um
Surface area: S = 6/(rho d32) = 6/(2650 * 18.975989135415965e-6)
          = 119.31662308820114 m^2/kg
Volume-weighted span: the volume-weighted quantiles are d50v exp(+/- z90 s) with
z90 = 1.2815515655446004, so
  span = exp(z90 s) - exp(-z90 s)
       = exp(0.7532789206115501) - exp(-0.7532789206115501)
       = 2.123952884323046 - 0.4708202368240027 = 1.6531326474990433
The span is independent of d_gn, which is the structural fact this output checks.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `s` | 0.5877866649021191 | abs 1e-12 | One natural logarithm of 1.8, accurate to 1 ulp near 1e-16. 1e-12 admits that with four orders of margin and is far tighter than any change in the sigma_g convention, which would move s by order 0.1. |
| `number_median_um` | 8.0 | abs 1e-12 | The count median is the stored input multiplied by exp(0), so the only error is one multiplication by exactly 1.0, which in IEEE 754 is exact. A non-zero deviation here would mean the weighting exponent is wrong, so the tightest tolerance in the vector belongs on this output. |
| `area_median_um` | 15.9654636684 | abs 1e-08 | Two floating point operations (one exp, one multiply) at a magnitude near 16 um, so the bound is near 2e-15 um. 1e-8 admits it with seven orders of margin and still separates the three medians, which differ by 6 um. |
| `volume_median_um` | 22.5541939242 | abs 1e-08 | Same basis as area_median_um. Two floating point operations (one exp, one multiply) at a magnitude near 16 um, so the bound is near 2e-15 um. 1e-8 admits it with seven orders of margin and still separates the three medians, which differ by 6 um. |
| `sauter_d32_um` | 18.9759891354 | abs 1e-08 | Same basis as area_median_um. Two floating point operations (one exp, one multiply) at a magnitude near 16 um, so the bound is near 2e-15 um. 1e-8 admits it with seven orders of margin and still separates the three medians, which differ by 6 um. |
| `specific_surface_area_m2_per_kg` | 119.3166230882 | abs 1e-06 | One division into the exp-derived d32, so the relative error is d32's, near 1e-15, which at 119 m^2/kg is 1e-13 absolute. 1e-6 is seven orders looser and still catches a wrong moment (using the volume median instead of d32 would give 99 m^2/kg). |
| `volume_span` | 1.6531326475 | abs 1e-09 | A difference of two exponentials, each accurate to 1 ulp, at magnitudes near 2 and 0.5, so the absolute bound is near 5e-16. 1e-9 admits it with six orders of margin. Reported to nine places because the value must not depend on d_gn, and a d_gn-dependent implementation would fail by a large margin. |

---

## GV-05: Furnas ternary maximum packing and optimal volume composition

- File: `data/golden/GV-05-furnas-ternary.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.packing`
- Dispatch entry: `furnas_packing`

**Reference.** McGeary, R.K. 1961, Mechanical Packing of Spherical Particles, Journal of the American Ceramic Society 44:513-522, doi:10.1111/j.1151-2916.1961.tb13716.x. The vibrated monomodal packing fraction 0.625 is McGeary's and is the module default.

**Inputs.**

```yaml
diameters_um:
- 490.0
- 70.0
- 10.0
```

**Provenance.**

- `diameters_um`: ASSUMED, 490/70/10 um, chosen so both successive ratios are exactly 7, the minimum
  separation McGeary found necessary. This keeps the geometry inside the model's
  stated domain, so no ratio warning is raised and the vector tests the equations
  rather than the guard.
- `phi_monomodal`: SOURCED, 0.625, McGeary 1961 vibrated spheres. This is the module default and is not
  passed by the vector.

**Arithmetic chain.**

```
Equation (1): phi_max = 1 - (1 - phi1)^N with phi1 = 0.625, so 1 - phi1 = 0.375.
  N = 3: 0.375^3 = 0.052734375
         phi_max = 1 - 0.052734375 = 0.947265625
Equation (2): raw term for class i (coarsest first) is phi1 (1 - phi1)^i:
  i = 0: 0.625 * 1        = 0.625
  i = 1: 0.625 * 0.375    = 0.234375
  i = 2: 0.625 * 0.140625 = 0.087890625
  sum = 0.947265625, which equals phi_max, an identity of the construction.
Normalised composition:
  0.625        / 0.947265625 = 0.6597938144329897
  0.234375     / 0.947265625 = 0.2474226804123711
  0.087890625  / 0.947265625 = 0.0927835051546392
  Exact as fractions: 64/97, 24/97, 9/97.
Size ratios: 490/70 = 7, 70/10 = 7.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `n_classes` | 3.0 | abs 0.0 | An integer count of size classes. Any deviation is a wrong answer and not a rounding difference, so the tolerance is exactly zero. |
| `phi_monomodal` | 0.625 | abs 0.0 | A constant passed through from the module default. Exactly representable in binary (0.625 = 5/8) and subjected to no arithmetic, so equality is the correct test and any non-zero tolerance would be a fudge factor. |
| `phi_max` | 0.947265625 | abs 0.0 | 0.625 and 0.375 are both exactly representable in binary (5/8 and 3/8), and 0.375^3 = 27/512 is exact in double precision, as is the subtraction from 1. The whole chain is therefore exact and the tolerance is zero. Choosing a non-zero tolerance here would be a fudge factor: there is no error to admit. |
| `composition_0` | 0.6597938144 | abs 1e-10 | The normalisation divides by 0.947265625, and 64/97 is not exactly representable, so one rounding of about 1e-17 enters. 1e-10 admits it with seven orders of margin and is tighter than the last digit quoted. |
| `composition_1` | 0.2474226804 | abs 1e-10 | Same basis as composition_0. The normalisation divides by 0.947265625, and 64/97 is not exactly representable, so one rounding of about 1e-17 enters. 1e-10 admits it with seven orders of margin and is tighter than the last digit quoted. |
| `composition_2` | 0.0927835052 | abs 1e-10 | Same basis as composition_0. The normalisation divides by 0.947265625, and 64/97 is not exactly representable, so one rounding of about 1e-17 enters. 1e-10 admits it with seven orders of margin and is tighter than the last digit quoted. |
| `composition_sum` | 1.0 | abs 1e-12 | Three roundings of about 1e-17 each summing to at most 3e-17. The module asserts closure to 1e-12 internally, so this tolerance matches the invariant the module itself enforces rather than inventing a looser one. |
| `size_ratio_0` | 7.0 | abs 1e-12 | 490e-6/70e-6 in double precision. Neither operand is exactly representable after the micron-to-metre conversion, so the quotient may differ from 7 by about 1 ulp, near 1e-15. 1e-12 admits that and is far tighter than the sevenfold threshold the ratio is compared against. |
| `size_ratio_1` | 7.0 | abs 1e-12 | Same basis as size_ratio_0. 490e-6/70e-6 in double precision. Neither operand is exactly representable after the micron-to-metre conversion, so the quotient may differ from 7 by about 1 ulp, near 1e-15. 1e-12 admits that and is far tighter than the sevenfold threshold the ratio is compared against. |

---

## GV-06: Krieger-Dougherty relative viscosity and volume to mass fraction conversion

- File: `data/golden/GV-06-krieger-dougherty.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.packing`
- Dispatch entry: `krieger_dougherty`

**Reference.** Krieger, I.M. and Dougherty, T.J. 1959, A Mechanism for Non-Newtonian Flow in Suspensions of Rigid Spheres, Transactions of the Society of Rheology 3:137-152, doi:10.1122/1.548848.

**Inputs.**

```yaml
phi: 0.6
phi_max: 0.75
intrinsic_viscosity: 2.5
filler_density_kg_m3: 2200.0
matrix_density_kg_m3: 1200.0
```

**Provenance.**

- `phi`: ASSUMED, 0.60 filler volume fraction, a high but achievable epoxy moulding compound
  loading.
- `phi_max`: ASSUMED, 0.75, a plausible maximum packing fraction for a graded spherical filler.
  Chosen above phi so the viscosity is finite; at phi = phi_max it diverges.
- `intrinsic_viscosity`: SOURCED, 2.5, the Einstein hard-sphere coefficient assumed in the Krieger-Dougherty
  derivation (Krieger and Dougherty 1959).
- `filler_density_kg_m3`: ASSUMED, 2200 kg/m^3, fused silica rather than crystalline quartz. Fused silica is the
  material actually used as a spherical filler.
- `matrix_density_kg_m3`: ASSUMED, 1200 kg/m^3, a cured epoxy.

**Arithmetic chain.**

```
eta_r = (1 - phi/phi_max)^(-[eta] phi_max)
phi/phi_max = 0.60/0.75 = 0.8, and 1 - 0.8 = 1/5 exactly in rational arithmetic.
exponent = -[eta] phi_max = -2.5 * 0.75 = -1.875
eta_r = (1/5)^(-1.875) = 5^1.875 = exp(1.875 ln 5)
  ln 5 = 1.6094379124341003
  1.875 * 1.6094379124341003 = 3.017696085813938
  exp(3.017696085813938) = 20.444135848948562
In double precision 1 - 0.6/0.75 evaluates to 0.20000000000000007 rather than to
0.2, because 0.6/0.75 is not binary-exact, and the powered result is then
20.444135848948548. The two routes differ by 1.4e-14, a relative 7e-16, which
sets the tolerance below.
Mass fraction from volume fraction:
  w = phi rho_f / (phi rho_f + (1 - phi) rho_m)
    = 0.6*2200 / (0.6*2200 + 0.4*1200)
    = 1320 / (1320 + 480) = 1320/1800 = 0.7333333333333333
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `relative_viscosity` | 20.4441358489 | abs 1e-06 | Measured, not assumed. Evaluating the exact rational route 5^1.875 and the floating point route (1 - 0.6/0.75)^-1.875 in this session gave 20.444135848948562 and 20.444135848948548, a difference of 1.4e-14. The subtraction 1 - phi/phi_max loses a bit because 0.6/0.75 is not binary- exact, and the exponent of -1.875 amplifies that relative error slightly. 1e-6 absolute admits the measured 1.4e-14 with eight orders of margin and does not fail on a reassociated implementation. A wrong intrinsic viscosity (2.0 in place of 2.5) gives 5^1.5 = 11.18, which this tolerance detects by seven orders of magnitude. |
| `mass_fraction` | 0.7333333333333333 | abs 1e-12 | Exact small integers throughout (1320/1800 = 11/15), so the only error is one division, at most 1 ulp near 1e-17. 1e-12 admits it with five orders of margin. The conversion direction is the substance of the check: swapping the densities gives 0.5, which fails by 0.23. |

---

## GV-07: Inclusion exposure, enclosed fraction and the liberation size that follows

- File: `data/golden/GV-07-liberation-geometry.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.liberation`
- Dispatch entry: `liberation_geometry`

**Reference.** Gaudin, A.M. 1939, Principles of Mineral Dressing, McGraw-Hill, chapter on liberation by size reduction. The cubic random-fracture form used here is the standard geometric idealisation and is labelled as such in the module.

**Inputs.**

```yaml
particle_size_um: 75.0
inclusion_size_um: 15.0
target_exposure: 0.6
```

**Provenance.**

- `particle_size_um`: ASSUMED, 75 um, the 200 mesh aperture, a common grind target.
- `inclusion_size_um`: ASSUMED, 15 um, a mineral inclusion size. Not a measurement on Vikarabad ore; no
  inclusion size distribution for that deposit exists in the citable record.
- `target_exposure`: ASSUMED, 0.60, a design exposure target.

**Arithmetic chain.**

```
ratio r = d_i/d_p = 15/75 = 0.2 exactly.
Enclosed fraction = (1 - r)^3 = 0.8^3 = 64/125 = 0.512 exactly as a rational.
In double precision 0.8 is not exactly representable and the cube evaluates to
0.5120000000000001, so the exposure evaluates to 0.4879999999999999.
Exposure = 1 - 0.512 = 0.488
Closure: 0.512 + 0.488 = 1 exactly, which is the identity this vector pins.
Liberation size for a target exposure E* = 0.60:
  d_p = d_i / (1 - (1 - E*)^(1/3))
  (1 - 0.60) = 0.4
  0.4^(1/3) = 0.7368062997280773
  1 - 0.7368062997280773 = 0.2631937002719227
  d_p = 15 / 0.2631937002719227 = 56.992245576176465 um
Direction check by hand: a 56.9922 um particle has r = 15/56.9922 = 0.2631937, so
(1 - 0.2631937)^3 = 0.4 and exposure 1 - 0.4 = 0.6, as required.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `enclosed_fraction` | 0.512 | abs 1e-12 | 0.2 and 0.8 are not binary-exact, so one cube of a rounded value introduces a relative error near 3e-16, which is 1.5e-16 absolute at 0.512. 1e-12 admits that with four orders of margin. A squared rather than cubed form gives 0.64, detected by 0.128. |
| `exposure` | 0.488 | abs 1e-12 | One subtraction of the enclosed fraction from 1, so the bound is the enclosed fraction's own error of about 2e-16 plus one rounding. 1e-12 admits that with four orders of margin. |
| `closure` | 1.0 | abs 1e-12 | The two outputs are computed by separate module functions rather than one from the other, so their sum being 1 is a real cross-check and not an identity of the arithmetic. Two values each carrying at most 2e-16 sum to at most 4e-16 of error. 1e-12 admits it and would catch any definitional mismatch, which would show as a discrepancy of order 0.1. |
| `liberation_size_um` | 56.9922455762 | abs 1e-06 | One cube root and one division. The cube root of 0.4 carries about 1 ulp, so the relative error in the divisor near 0.263 is about 6e-16, giving 3.4e-14 um at 57 um. 1e-6 um admits that with eight orders of margin. A square root in place of the cube root gives 15/(1 - 0.632456) = 40.82 um, detected by 16 um. |

---

## GV-08: Leachable fraction of an impurity from its partition and two inclusion sizes

- File: `data/golden/GV-08-liberation-leachable.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.liberation`
- Dispatch entry: `liberation_leachable`

**Reference.** Same geometric model as GV-07 (Gaudin 1939 cubic form), applied separately to the fluid and mineral inclusion populations and combined with the partition.

**Inputs.**

```yaml
particle_size_um: 75.0
fluid_inclusion_size_um: 5.0
mineral_inclusion_size_um: 20.0
partition:
  surface: 0.05
  fluid: 0.1
  mineral: 0.3
  lattice: 0.55
```

**Provenance.**

- `particle_size_um`: ASSUMED, 75 um.
- `fluid_inclusion_size_um`: ASSUMED, 5 um, smaller than the mineral inclusions, which is the usual observation and
  is what makes fluid inclusions harder to liberate.
- `mineral_inclusion_size_um`: ASSUMED, 20 um.
- `partition`: ASSUMED, a four-way split summing to exactly 1. This is a scenario, not a property of
  the Vikarabad deposit, which is uncharacterized in the citable record. A measured
  split requires LA-ICP-MS on inclusion-free domains plus a staged leach mass balance,
  neither of which has been run.

**Arithmetic chain.**

```
Surface-sited material is fully accessible, so it contributes its whole fraction.
Fluid inclusion exposure at 5 um in 75 um:
  r = 5/75 = 0.06666666666666667
  (1 - r)^3 = (14/15)^3 = 2744/3375 = 0.813037037037037
  E_fluid = 1 - 0.813037037037037 = 631/3375 = 0.18696296296296297
Mineral inclusion exposure at 20 um in 75 um:
  r = 20/75 = 0.26666666666666666
  (1 - r)^3 = (11/15)^3 = 1331/3375 = 0.39437037037037037
  E_mineral = 1 - 0.39437037037037037 = 2044/3375 = 0.6056296296296296
Leachable = surface + fluid*E_fluid + mineral*E_mineral
          = 0.05 + 0.10*0.18696296296296297 + 0.30*0.6056296296296296
          = 0.05 + 0.018696296296296297 + 0.18168888888888888
          = 0.25038518518518516
Ceiling: 1 - lattice = 1 - 0.55 = 0.45. The leachable fraction 0.2504 is below the
ceiling, so the cap does not bind and the vector tests the sum rather than the cap.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `exposure_fluid` | 0.18696296296296297 | abs 1e-12 | One division, one cube, one subtraction, all in double precision on values that are not binary-exact (1/15 and 14/15), so the bound is a few ulp, near 5e-16 absolute. 1e-12 admits it with three orders of margin. |
| `exposure_mineral` | 0.6056296296296296 | abs 1e-12 | Same basis as exposure_fluid. One division, one cube, one subtraction, all in double precision on values that are not binary-exact (1/15 and 14/15), so the bound is a few ulp, near 5e-16 absolute. 1e-12 admits it with three orders of margin. |
| `leachable_fraction` | 0.2503851851851852 | abs 1e-12 | A weighted sum of three terms, each carrying at most 5e-16, so the total bound is near 1.5e-15. 1e-12 admits that with three orders of margin. The value is the one a flowsheet designer reads, so the tolerance is set by the arithmetic bound and not by any engineering judgment about how precisely it is known: the partition inputs themselves are ASSUMED, and that uncertainty is stated in the provenance rather than smuggled into the tolerance. |
| `ceiling_one_minus_lattice` | 0.45 | abs 1e-12 | One subtraction from 1 of a value that is not binary-exact, so 1 ulp near 1e-17. 1e-12 admits it. The output exists so the vector records that the cap was checked and found not binding, which is why it is reported rather than assumed. |

---

## GV-09: Impurity floor and removable inventory from a four-way partition

- File: `data/golden/GV-09-impurity-floor.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.impurity_location`
- Dispatch entry: `impurity_floor`

**Reference.** Muller, A., Herrington, R., Armstrong, R. et al. 2012, Trace elements and cathodoluminescence of quartz in stockwork veins of Myanmar porphyry copper systems, in Gotze, J. and Mockel, R. eds., Quartz: Deposits, Mineralogy and Analytics, Springer. The four-way surface, fluid inclusion, mineral inclusion and lattice partition is the framework used there. The numbers in this vector are ASSUMED scenario values, not measurements.

**Inputs.**

```yaml
sample_id: AE-Q-IN-VKB-001
ore_type: vein_quartz
element: Al
bulk_ppm: 120.0
partition:
  surface: 0.1
  fluid: 0.05
  mineral: 0.35
  lattice: 0.5
efficiencies:
  surface: 0.95
  fluid: 0.8
  mineral: 0.6
```

**Provenance.**

- `bulk_ppm`: ASSUMED, 120 ppm Al. A scenario value. The Vikarabad deposit is uncharacterized in the
  citable record and no assay exists for it; this number is gated on a future 20 to 30
  sample campaign and is never a property of the ore.
- `partition`: ASSUMED, summing to exactly 1. The 0.50 lattice share is the parameter that sets the
  ceiling grade and is precisely the number the assay campaign has to measure, because
  lattice-bound Al cannot be removed by acid leaching.
- `efficiencies`: ASSUMED removal efficiencies per location. Surface material is easiest to remove
  (0.95), mineral inclusions hardest of the three non-lattice sites (0.60). Lattice
  removal is left at its default of zero, which the module requires unless a named
  mechanism is supplied.

**Arithmetic chain.**

```
Location inventories, each bulk times its partition share:
  surface: 120 * 0.10 = 12.0 ppm
  fluid:   120 * 0.05 = 6.0 ppm
  mineral: 120 * 0.35 = 42.0 ppm
  lattice: 120 * 0.50 = 60.0 ppm
  sum = 120.0 ppm, equal to the bulk, which is the closure this vector pins.
Floor, the residue a flowsheet with these efficiencies leaves behind:
  surface residue: 12.0 * (1 - 0.95) = 12.0 * 0.05 = 0.6 ppm
  fluid residue:    6.0 * (1 - 0.80) =  6.0 * 0.20 = 1.2 ppm
  mineral residue: 42.0 * (1 - 0.60) = 42.0 * 0.40 = 16.8 ppm
  lattice residue: 60.0 * (1 - 0.00) = 60.0 ppm
  floor = 0.6 + 1.2 + 16.8 + 60.0 = 78.6 ppm
Removable inventory, what a PERFECT physical flowsheet could take out, which is
bulk minus lattice and does not depend on the efficiencies:
  12.0 + 6.0 + 42.0 = 60.0 ppm
Cross-check: removable + lattice = 60.0 + 60.0 = 120.0 ppm = bulk.
Note the floor 78.6 exceeds the lattice inventory 60.0 by 18.6 ppm, the part of the
removable inventory these imperfect efficiencies fail to reach. That gap, not the
floor, is what an extra stage buys.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `conc_surface_ppm` | 12.0 | abs 1e-09 | One multiplication of exact decimals that are not binary-exact (0.10), so 1 ulp near 1e-15 ppm. 1e-9 ppm admits it with six orders of margin and is far below any assay resolution. |
| `conc_fluid_ppm` | 6.0 | abs 1e-09 | Same basis as conc_surface_ppm. One multiplication of exact decimals that are not binary-exact (0.10), so 1 ulp near 1e-15 ppm. 1e-9 ppm admits it with six orders of margin and is far below any assay resolution. |
| `conc_mineral_ppm` | 42.0 | abs 1e-09 | Same basis as conc_surface_ppm. One multiplication of exact decimals that are not binary-exact (0.10), so 1 ulp near 1e-15 ppm. 1e-9 ppm admits it with six orders of margin and is far below any assay resolution. |
| `conc_lattice_ppm` | 60.0 | abs 1e-09 | Same basis as conc_surface_ppm. One multiplication of exact decimals that are not binary-exact (0.10), so 1 ulp near 1e-15 ppm. 1e-9 ppm admits it with six orders of margin and is far below any assay resolution. |
| `conc_sum_ppm` | 120.0 | abs 1e-09 | Four terms each carrying at most 1e-15 ppm, so 4e-15 ppm. 1e-9 admits it. This output is the closure check: if the partition did not sum to 1, or if the module normalised it silently, the sum would move by ppm and not by ulp. |
| `floor_ppm` | 78.6 | abs 1e-09 | Four products and three additions on values near 100 ppm, so the bound is about 1e-14 ppm. 1e-9 ppm admits it with five orders of margin. The floor is the number a purity claim rests on, so the tolerance is set by the arithmetic and the input uncertainty is carried in the provenance instead: the inputs are ASSUMED and the floor inherits their status, which no tolerance can repair. |
| `removable_ppm` | 60.0 | abs 1e-09 | A sum of three exact products. The output is here because it must NOT depend on the efficiencies: an implementation that subtracted the floor from the bulk would return 41.4 rather than 60.0, an error of 18.6 ppm which this tolerance detects by ten orders of magnitude. |
| `removable_plus_lattice_ppm` | 120.0 | abs 1e-09 | A second closure identity, summing two independently returned quantities back to the bulk. Four ulp at most, near 4e-15 ppm. |

---

## GV-10: Implied removal rate and the SiO2 percent ceiling a residual trace sum sets

- File: `data/golden/GV-10-purity-ceiling.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.impurity_location`
- Dispatch entry: `purity_ceiling`

**Reference.** Xia, Y., Zhang, X., Wang, L. et al. 2024, purification study reporting a trace element sum falling from 128.86 to 24.23 ug/g, doi:10.3390/min14070727. The two input concentrations below are that paper's reported figures, as recorded in src/ae/physics/impurity_location.py; the floor figure is this suite's own GV-09 output and is a scenario value, not a measurement.

**Inputs.**

```yaml
feed_sum_ppm: 128.86
product_sum_ppm: 24.23
floor_sum_ppm: 78.6
```

**Provenance.**

- `feed_sum_ppm`: SOURCED, 128.86 ug/g, Xia et al. 2024. Quoted to five significant figures as
  published. No uncertainty is given in the source, so none is claimed here; the
  figure is used as an exact input to an arithmetic identity.
- `product_sum_ppm`: SOURCED, 24.23 ug/g, Xia et al. 2024.
- `floor_sum_ppm`: COMPUTED, 78.6 ppm, the floor derived in GV-09 from ASSUMED partition and efficiency
  inputs. Carried here to show what such a floor would imply for a purity claim. It is
  a scenario, gated on the assay campaign.

**Arithmetic chain.**

```
Implied removal fraction = 1 - product/feed
  24.23 / 128.86 = 0.18803352475554863
  1 - 0.18803352475554863 = 0.8119664752444513
  as a percentage: 81.19664752444513 percent
SiO2 percent ceiling from a residual trace sum, treating the trace sum as the only
non-SiO2 constituent:
  from the product sum: 100 - 24.23e-4 = 100 - 0.002423 = 99.997577 percent
  from the GV-09 floor: 100 - 78.6e-4  = 100 - 0.00786   = 99.99214 percent
The unit step is the substance of the check: 1 ppm is 1e-4 percent, so a trace sum
of 24.23 ppm costs 0.002423 percentage points, not 0.2423 and not 0.00002423.
Getting that factor wrong by 100 is the difference between 4N and 5N in a
marketing claim, which is why the conversion is pinned here.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `implied_removal_fraction` | 0.8119664752444513 | abs 1e-12 | One division and one subtraction from 1. The division carries 1 ulp near 2e-17; the subtraction from 1 of a value near 0.19 is exact to within 1 ulp of the result, near 1e-16. 1e-12 admits that with four orders of margin. |
| `implied_removal_percent` | 81.1966475244 | abs 1e-09 | The fraction times 100, so the absolute error scales by 100 to near 1e-14. 1e-9 admits it with five orders of margin. Reported separately from the fraction because the published figure in the module docstring is a percentage (81.2), and the vector checks the scaling explicitly. |
| `sio2_percent_from_product` | 99.997577 | abs 1e-12 | A subtraction of 0.002423 from 100. This is the catastrophic-cancellation case in the vector: the result keeps only about 11 significant digits of the small term, because 100 has an exponent 5 orders above 0.002423. The absolute error is therefore about 1 ulp of 100, near 1.4e-14. 1e-12 admits that with two orders of margin, which is why this tolerance is not set to 1e-15: at that level the check would fail on a legitimate reordering of the subtraction. |
| `sio2_percent_from_floor` | 99.99214 | abs 1e-12 | Same basis as sio2_percent_from_product. A subtraction of 0.002423 from 100. This is the catastrophic-cancellation case in the vector: the result keeps only about 11 significant digits of the small term, because 100 has an exponent 5 orders above 0.002423. The absolute error is therefore about 1 ulp of 100, near 1.4e-14. 1e-12 admits that with two orders of margin, which is why this tolerance is not set to 1e-15: at that level the check would fail on a legitimate reordering of the subtraction. |

---

## GV-11: Shrinking-core characteristic times in all three rate-controlling regimes

- File: `data/golden/GV-11-leach-characteristic-times.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.leaching`
- Dispatch entry: `leach_characteristic_times`

**Reference.** Levenspiel, O. 1999, Chemical Reaction Engineering, 3rd edition, Wiley, chapter 25, shrinking core model for a particle of unchanging size. The three tau expressions are that chapter's standard results.

**Inputs.**

```yaml
sample_id: AE-Q-IN-VKB-001
bulk_fe_ppm: 30.0
fe_lattice_fraction: 0.1
reagent: HCl
particle_radius_m: 0.0001
concentration_mol_m3: 1000.0
solid_molar_density_mol_m3: 30000.0
stoich_b: 1.0
temperature_K: 353.15
k_film_m_s: 0.0001
diffusivity_m2_s: 1.0e-09
k_surface_m_s: 0.0001
```

**Provenance.**

- `particle_radius_m`: ASSUMED, 1e-4 m (100 um), a round number chosen so the tau arithmetic is traceable.
- `concentration_mol_m3`: ASSUMED, 1000 mol/m^3, which is 1 molar.
- `solid_molar_density_mol_m3`: ASSUMED, 30000 mol/m^3. This is the molar density of the REACTIVE SOLID PHASE, not of
  the quartz matrix, a distinction the module enforces and which is easy to get wrong
  by a factor of more than one.
- `stoich_b`: ASSUMED, 1.0 mol of solid per mol of reagent, chosen to keep the arithmetic clear.
- `temperature_K`: ASSUMED, 353.15 K (80 degC). All three activation energies are set to exactly zero in
  this vector so that k(T) equals the prefactor and no exponential has to be evaluated
  by hand. That is what makes the vector hand-traceable.
- `k_film_m_s`: ASSUMED, 1e-4 m/s.
- `diffusivity_m2_s`: ASSUMED, 1e-9 m^2/s, the order of magnitude of an aqueous ionic diffusivity in a
  porous product layer.
- `k_surface_m_s`: ASSUMED, 1e-4 m/s.
- `bulk_fe_ppm`: ASSUMED, 30 ppm Fe. Required to build the feedstock object; it does not enter the tau
  arithmetic.

**Arithmetic chain.**

```
With Ea = 0 the Arrhenius factor exp(-0/(RT)) is exactly 1, so k_g = 1e-4 m/s,
D_e = 1e-9 m^2/s and k_s = 1e-4 m/s at any temperature.
Film diffusion control:
  tau = rho R / (3 b k_g C_A)
      = 30000 * 1e-4 / (3 * 1.0 * 1e-4 * 1000)
      = 3.0 / 0.3 = 10.0 s
Product layer diffusion control:
  tau = rho R^2 / (6 b D_e C_A)
      = 30000 * (1e-4)^2 / (6 * 1.0 * 1e-9 * 1000)
      = 30000 * 1e-8 / 6e-6
      = 3.0e-4 / 6.0e-6 = 50.0 s
Surface reaction control:
  tau = rho R / (b k_s C_A)
      = 30000 * 1e-4 / (1.0 * 1e-4 * 1000)
      = 3.0 / 0.1 = 30.0 s
The ordering film < surface < product layer at this radius is the physical content:
product layer control is slowest here because it scales as R^2 while the other two
scale as R, so the ranking flips with particle size. That is the reason the module
reports all three rather than one.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `tau_film_s` | 10.0 | abs 1e-09 | All operands are exactly representable powers of ten times small integers, so the chain 30000*1e-4/(3*1e-4*1000) is exact in binary except for the final division, at most 1 ulp near 2e-15 s. 1e-9 s admits that with six orders of margin. It is not set to zero because the module carries the quantity through pint unit conversions, which may reassociate the products. |
| `tau_product_layer_s` | 50.0 | abs 1e-09 | As tau_film_s, with one additional squaring of 1e-4, which is exact. The R^2 dependence is the substance of the check: an implementation using R rather than R^2 would return 5e5 s, detected by 14 orders of magnitude. |
| `tau_surface_reaction_s` | 30.0 | abs 1e-09 | As tau_film_s. The absence of the factor 3 is what distinguishes this from film control; an implementation carrying the 3 would return 10 s, detected by 20 s. |

---

## GV-12: Conversion at a fixed dimensionless time in three regimes, with the g(X) round trip

- File: `data/golden/GV-12-leach-conversion.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.leaching`
- Dispatch entry: `leach_conversion`

**Reference.** Levenspiel, O. 1999, Chemical Reaction Engineering, 3rd edition, Wiley, chapter 25. The three g(X) forms are that chapter's; the closed-form inversion of the cubic product-layer case by the trigonometric method is standard for a depressed cubic with three real roots.

**Inputs.**

```yaml
t_over_tau: 0.875
conversion_x: 0.8
```

**Provenance.**

- `t_over_tau`: ASSUMED, 0.875 = 7/8, chosen because it makes the surface-reaction inversion exact in
  binary arithmetic: 1 - 0.875 = 0.125 = 1/8 and its cube is exactly 1/512. A
  dimensionless time was chosen instead of a time and a tau so that this vector tests
  the conversion relations alone, with GV-11 testing tau.
- `conversion_x`: ASSUMED, 0.80, a separate point at which the three g(X) functions are evaluated
  forward. 0.80 is deliberately NOT the inverse image of 0.875 in any regime, so the
  forward and inverse halves of the vector cannot cancel a shared error.

**Arithmetic chain.**

```
Film diffusion, g(X) = X, so X(theta) = theta:
  X_film = 0.875
Surface reaction, g(X) = 1 - (1-X)^(1/3), inverted as X = 1 - (1 - theta)^3:
  1 - 0.875 = 0.125
  0.125^3 = 0.001953125 exactly (1/512)
  X_sr = 1 - 0.001953125 = 0.998046875
Product layer, g(X) = 1 - 3(1-X)^(2/3) + 2(1-X). Substituting u = (1-X)^(1/3)
gives 2u^3 - 3u^2 + 1 - theta = 0, a cubic with three real roots, of which the
physical one lies in [0,1] and is obtained by the trigonometric formula:
  phi = arccos(2 theta - 1) = arccos(0.75) = 0.7227342478134157 rad
  u = 0.5 + cos(phi/3 + 4 pi/3)
    phi/3 = 0.24091141593780524
    4 pi/3 = 4.188790204786391
    sum = 4.429701620724196 rad
    cos(4.429701620724196) = -0.27893734916576235
    u = 0.5 - 0.27893734916576235 = 0.22106265083423765
  X_pl = 1 - u^3 = 1 - 0.010803043390790... = 0.9891969566092098
Round trip: feeding each X back through the module's own g(X) must return 0.875.
By hand for the surface-reaction case: 1 - (0.001953125)^(1/3) = 1 - 0.125 = 0.875.
Forward evaluation of g at X = 0.80, where 1 - X = 0.2:
  film:    g = 0.80
  surface: g = 1 - 0.2^(1/3) = 1 - 0.5848035476425733 = 0.4151964523574267
  product: g = 1 - 3*0.2^(2/3) + 2*0.2
             = 1 - 3*0.34199518933533946 + 0.4
             = 1 - 1.0259855680060184 + 0.4 = 0.3740144319939818
The three forward values differ by a factor of two, which is why a vector that
used only one of them could not distinguish the regimes.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `x_film` | 0.875 | abs 0.0 | The identity X = theta, one assignment of an exactly representable value (7/8), so equality is the correct test and any tolerance would be a fudge factor. |
| `x_surface_reaction` | 0.998046875 | abs 0.0 | Every operand is a dyadic rational: 1 - 0.875 = 0.125 and 0.125^3 = 1/512 are both exact in binary floating point, as is the final subtraction from 1. The input was chosen for exactly this reason, so equality is the correct test. |
| `x_product_layer` | 0.9891969566 | abs 1e-09 | One arccos, one cosine, one cube. Transcendental functions are accurate to about 1 ulp, and the cosine is evaluated at an argument near 4.43 rad where its derivative is about 0.96, so no catastrophic amplification occurs. The propagated bound is near 1e-15. 1e-9 admits it with six orders of margin. Deliberately looser than the surface-reaction tolerance because the trigonometric route genuinely carries transcendental rounding while the dyadic route does not. |
| `g_round_trip_film` | 0.875 | abs 1e-12 | An identity mapping composed with itself, exact but routed through numpy array conversion, so 1e-12 admits any ulp-level difference. |
| `g_round_trip_surface_reaction` | 0.875 | abs 1e-12 | The forward g is a separate function from the inversion, so this is a real cross-check. The cube root of an exact 1/512 is exactly 0.125, so the round trip is exact to 1 ulp, near 1e-16. |
| `g_round_trip_product_layer` | 0.875 | abs 1e-12 | Measured in this session: substituting the trigonometric root back into g returned 0.8749999999999997, an error of 3e-16 against the target 0.875. The error arises from the two fractional powers in g. 1e-12 admits the measured 3e-16 with three orders of margin and would still catch a wrong coefficient in the cubic, which shifts g by order 0.1. |
| `g_at_x_film` | 0.8 | abs 1e-12 | The identity g = X on an exactly representable 0.8 input rounded once, so at most 1 ulp near 1e-17. 1e-12 admits that with five orders of margin. |
| `g_at_x_surface_reaction` | 0.41519645235742675 | abs 1e-12 | One cube root and one subtraction. The cube root of 0.2 carries 1 ulp near 6e-17; the subtraction from 1 leaves an absolute error near 1e-16. 1e-12 admits it with four orders of margin. |
| `g_at_x_product_layer` | 0.37401443199398166 | abs 1e-12 | Two fractional powers of 0.2 and two additions, so the bound is a few ulp near 3e-16. 1e-12 admits it. The coefficients 3 and 2 are the substance: dropping the 2(1-X) term gives -0.0259856, a sign change, detected immediately. |

---

## GV-13: Characteristic time back-calculated from one timed leach assay

- File: `data/golden/GV-13-leach-tau-from-point.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.leaching`
- Dispatch entry: `leach_tau_from_point`

**Reference.** Levenspiel, O. 1999, Chemical Reaction Engineering, 3rd edition, Wiley, chapter 25. tau = t / g(X) follows directly from t/tau = g(X); no additional source is needed.

**Inputs.**

```yaml
regime: surface_reaction
conversion_x: 0.8
t_s: 3600.0
```

**Provenance.**

- `regime`: ASSUMED, surface reaction control. In a real fit the regime would be identified from
  the size exponent and the apparent activation energy, not assumed; this vector tests
  the arithmetic of the back-calculation only.
- `conversion_x`: MEASURED in the hypothetical, 0.80, meaning 80 percent of the leachable inventory
  removed. A real single-point fit on one assay pair has no degrees of freedom left to
  test the regime, which the module states as a caveat.
- `t_s`: MEASURED in the hypothetical, 3600 s (one hour).

**Arithmetic chain.**

```
g(X) for surface reaction control at X = 0.80:
  1 - X = 0.2
  0.2^(1/3) = 0.5848035476425733
  g = 1 - 0.5848035476425733 = 0.4151964523574267
tau = t / g = 3600 / 0.4151964523574267 = 8670.594316400606 s
As a sanity figure: 8670.6 s is 2.41 hours, so 80 percent conversion in one hour
implies complete conversion of a single particle in about 2.4 hours under this
regime. Under product layer control the same assay pair would give
3600/0.3740144319939818 = 9625.297026126329 s, a 10 percent different answer from
the same data, which is why the regime cannot be assumed away.
Round trip: the module's forward conversion at t = 3600 s with this tau must return
X = 0.80.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `tau_s` | 8670.5943164006 | abs 1e-06 | One cube root, one subtraction and one division. The cube root carries 1 ulp, giving a relative error near 1.5e-16 in g, which propagates to about 1.3e-12 s at tau = 8670 s. 1e-6 s admits that with six orders of margin and is far tighter than the 10 percent regime ambiguity noted in the arithmetic, so the tolerance tests the arithmetic rather than hiding the physics. |
| `round_trip_conversion` | 0.8 | abs 1e-12 | The forward conversion is a separate code path from tau_from_single_point, so the round trip is a genuine cross-check rather than an algebraic identity. Each direction carries a cube root, so the composed bound is a few ulp near 3e-16. 1e-12 admits that with three orders of margin. |

---

## GV-14: Chloride vapour pressures at 400 K by one-point Clausius-Clapeyron, and the separation they imply

- File: `data/golden/GV-14-chlorination-volatility.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.chlorination`
- Dispatch entry: `chlorination_volatility`

**Reference.** Transition temperatures are the module's recorded PubChem values: TiCl4 boils at 136.45 degC (CID 24193), AlCl3 sublimes at 180 degC (CID 24012), NaCl boils at 1465 degC (CID 5234), all retrieved via the PUG-View REST API and accessed 2026-09-16 as recorded in the module. TiCl4 also carries a MEASURED PubChem heat of vaporization of 36.2 kJ/mol at 136.45 degC. AlCl3 and NaCl carry no measured enthalpy in the module, so for those two the enthalpy of transition is estimated by Trouton's rule at 85 J/(mol K), which the module tags ASSUMED with low confidence. The vector reports which of the two applies per species, because calling all three Trouton estimates would assert an estimate where a measurement exists.

**Inputs.**

```yaml
species:
- TiCl4
- AlCl3
- NaCl
temperature_K: 400.0
total_pressure_Pa: 101325.0
```

**Provenance.**

- `temperature_K`: ASSUMED, 400 K (126.85 degC), chosen deliberately below all three transition
  temperatures so that every predicted pressure is below one atmosphere and the total-
  pressure cap does NOT bind. A vector at 700 K would return the cap for two of three
  species and would test the clamp instead of the relation.
- `total_pressure_Pa`: SOURCED, 101325 Pa, the standard atmosphere by definition, exact.
- `TiCl4 enthalpy`: SOURCED and MEASURED, 36.2 kJ/mol, PubChem heat of vaporization at 136.45 degC (CID
  24193), quoted to three significant figures, so the implied uncertainty is of order
  plus or minus 0.05 kJ/mol.
- `AlCl3 and NaCl enthalpy`: ASSUMED via Trouton's rule at 85 J/(mol K), uncertainty of order plus or minus 15
  J/(mol K) across ordinary liquids, which propagates to roughly a factor of two in
  pressure at this temperature offset. For those two species the vector therefore
  checks the arithmetic of the estimate and not the accuracy of the estimate.

**Arithmetic chain.**

```
One-point Clausius-Clapeyron:
  ln(p/P0) = -(dH/R) (1/T - 1/T_transition),  P0 = 101325 Pa, R = 8.314462618
where dH is the MEASURED enthalpy of transition when the module has one, and
otherwise the Trouton estimate dH = 85 * T_transition (in K).
TiCl4, T_b = 136.45 + 273.15 = 409.59999999999997 K, MEASURED dH = 36200.0 J/mol:
  dH/R = 36200.0 / 8.314462618 = 4353.859252626927
  1/400 - 1/409.6 = 0.0025 - 0.00244140625 = 5.859375000000005e-05
  exponent = -4353.859252626927 * 5.859375000000005e-05 = -0.2551089405836092
  p = 101325 * exp(-0.2551089405836092) = 101325 * 0.7748320827037518
    = 78509.86077995766 Pa
  Had the Trouton rule been applied to TiCl4 as well, dH would have been
  85 * 409.6 = 34816.0 J/mol and the pressure 79279.34025051698 Pa, 1.0 percent
  higher. The two are close here because Trouton's rule is accurate for a
  non-associating liquid like TiCl4, which is the case it was derived for.
AlCl3, T_sub = 180 + 273.15 = 453.15 K, Trouton estimate:
  dH = 85 * 453.15 = 38517.75 J/mol
  dH/R = 38517.75 / 8.314462618 = 4632.620503532343
  1/400 - 1/453.15 = 0.0025 - 0.0022067747986318 = 0.00029322520136820015
  exponent = -4632.620503532343 * 0.00029322520136820015 = -1.358401080010724
  p = 101325 * exp(-1.358401080010724) = 101325 * 0.2570714852576491
    = 26047.768243731298 Pa
NaCl, T_b = 1465 + 273.15 = 1738.15 K, Trouton estimate:
  dH = 85 * 1738.15 = 147742.75 J/mol
  dH/R = 147742.75 / 8.314462618 = 17769.36848331621
  1/400 - 1/1738.15 = 0.0025 - 0.0005753243390961655 = 0.0019246756609038345
  exponent = -34.2002710294704
  p = 101325 * exp(-34.2002710294704) = 1.4214370287424754e-10 Pa
The separation this predicts: at 400 K the TiCl4 partial pressure exceeds the NaCl
partial pressure by log10(78509.86077995766/1.4214370287424754e-10) = 14.74 orders
of magnitude, which is why chlorination roasting can strip Ti while leaving Na
entirely behind. The NaCl figure rests on a Trouton estimate extrapolated 1338 K
below its boiling point and is not a measurement; it should be read as "many orders
of magnitude lower", not as 1.42e-10 Pa.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `TiCl4_pressure_Pa` | 78509.86077995766 | rel 1e-12 | A RELATIVE tolerance is correct here because the quantity is an exponential: the absolute error scales with the value, and the same absolute tolerance cannot serve both 79279 Pa and 1.4e-10 Pa in one vector. The chain is one reciprocal difference, one multiplication and one exp, each 1 ulp, so the relative bound is near 4e-16. 1e-12 relative admits that with three orders of margin and still detects a wrong enthalpy basis: applying Trouton to TiCl4 in place of its measured 36.2 kJ/mol moves the pressure from 78509.86 Pa to 79279.34 Pa, a 1.0 percent shift, ten orders above this tolerance. |
| `TiCl4_capped` | 0.0 | abs 0.0 | A boolean encoded as 0.0 or 1.0. There is no rounding to admit, so the tolerance is exactly zero. The output exists to record that the total- pressure cap did not bind, which is a precondition for the pressure values being the raw Clausius-Clapeyron relation rather than the clamp. |
| `TiCl4_transition_enthalpy_J_per_mol` | 36200.0 | abs 1e-09 | A SOURCED constant passed straight through, 36.2 kJ/mol converted to 36200.0 J/mol, so the only error is the pint unit conversion, one multiplication by 1000, which is exact in binary. 1e-9 J/mol admits any conversion rounding. The output exists to pin which enthalpy basis is in use, checked together with TiCl4_enthalpy_is_trouton below. |
| `TiCl4_enthalpy_is_trouton` | 0.0 | abs 0.0 | A boolean encoded as 0.0, recording that TiCl4's enthalpy is the MEASURED PubChem value and not a Trouton estimate. Exactly zero tolerance: this is a provenance assertion, and a 1.0 here would mean the measured value was silently discarded in favour of an estimate, which is the substitution this output exists to catch. |
| `TiCl4_transition_T_K` | 409.6 | abs 1e-09 | A degC to K conversion, 136.45 + 273.15, where neither operand is binary- exact, so 1 ulp near 6e-14 K. 1e-9 K admits it. The output is reported so that the enthalpy above can be checked against its own input. |
| `AlCl3_pressure_Pa` | 26047.768243731298 | rel 1e-12 | Same basis as TiCl4_pressure_Pa. A RELATIVE tolerance is correct here because the quantity is an exponential: the absolute error scales with the value, and the same absolute tolerance cannot serve both 79279 Pa and 1.4e-10 Pa in one vector. The chain is one reciprocal difference, one multiplication and one exp, each 1 ulp, so the relative bound is near 4e-16. 1e-12 relative admits that with three orders of margin and still detects a wrong Trouton constant, which moves the pressure by tens of percent. |
| `AlCl3_capped` | 0.0 | abs 0.0 | A boolean encoded as 0.0 or 1.0, recording that the total-pressure cap did not bind. There is no rounding to admit, so the tolerance is exactly zero. |
| `AlCl3_transition_enthalpy_J_per_mol` | 38517.75 | abs 1e-09 | A Trouton estimate, one multiplication of 85 by the transition temperature in K. 85 * 453.15 = 38517.75 is exact to the last digit quoted here, so the bound is 1 ulp near 4e-12 J/mol. 1e-9 admits that with two orders of margin. The tolerance bounds the ARITHMETIC of the estimate; the estimate itself carries a plus or minus 15 J/(mol K) uncertainty on the Trouton entropy, stated in the provenance. |
| `AlCl3_enthalpy_is_trouton` | 1.0 | abs 0.0 | A boolean encoded as 1.0, recording that AlCl3 has no measured enthalpy in the module and falls back to Trouton. Exactly zero tolerance, as a provenance assertion. A 0.0 would mean an unsourced measurement appeared. |
| `AlCl3_transition_T_K` | 453.15 | abs 1e-09 | 180 + 273.15, one addition, 1 ulp near 6e-14 K. |
| `NaCl_pressure_Pa` | 1.4214370287424754e-10 | rel 1e-12 | A relative tolerance is not merely convenient here, it is the only workable choice: the value is 1.4e-10 Pa, so any absolute tolerance loose enough for the 79279 Pa output in the same vector would pass ANY value for this one. The relative bound is the same 4e-16 as the others because the exponent, although large in magnitude at -34.2, is evaluated once. |
| `NaCl_capped` | 0.0 | abs 0.0 | A boolean encoded as 0.0 or 1.0, recording that the total-pressure cap did not bind. There is no rounding to admit, so the tolerance is exactly zero. |
| `NaCl_transition_enthalpy_J_per_mol` | 147742.75 | abs 1e-09 | A Trouton estimate, 85 * 1738.15 = 147742.75, one multiplication, 1 ulp near 1.5e-11 J/mol. 1e-9 admits that with two orders of margin, and the Trouton uncertainty is carried in the provenance rather than here. |
| `NaCl_enthalpy_is_trouton` | 1.0 | abs 0.0 | A boolean encoded as 1.0, recording the Trouton fallback for NaCl. Exactly zero tolerance, as a provenance assertion. |
| `NaCl_transition_T_K` | 1738.15 | abs 1e-09 | One degC to K conversion, 1465 + 273.15, where 273.15 is not binary-exact, so 1 ulp near 2e-13 K. 1e-9 K admits that with four orders of margin. |

---

## GV-15: Gibbs energy of a stated reaction from a supplied thermochemical table

- File: `data/golden/GV-15-chlorination-gibbs.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.chlorination`
- Dispatch entry: `chlorination_gibbs`

**Reference.** The relation dG = sum(nu dHf) - T sum(nu S) is the standard thermodynamic identity and needs no source. The thermochemical values below are a FIXTURE TABLE supplied to the function by this vector, not a real reaction and not taken from any database. The module's own THERMO table is deliberately not used, because it is incomplete: JANAF entries for Cl2(g) and CO(g) were not reachable from this sandbox, which the module records, and a vector built on an incomplete table would be testing the MISSING sentinel rather than the arithmetic.

**Inputs.**

```yaml
temperature_K: 1073.15
stoichiometry:
  Z(g): 1.0
  X(s): -1.0
  Y(g): -1.0
thermo:
  X(s):
    Hf_kJ_per_mol: -400.0
    S_J_per_mol_K: 80.0
  Y(g):
    Hf_kJ_per_mol: -100.0
    S_J_per_mol_K: 210.0
  Z(g):
    Hf_kJ_per_mol: -250.0
    S_J_per_mol_K: 300.0
```

**Provenance.**

- `temperature_K`: ASSUMED, 1073.15 K (800 degC), a representative chlorination roasting temperature.
- `thermo`: ASSUMED fixture values with round magnitudes, chosen so that the sums are exact small
  integers. They describe no real substance. Labelling them X, Y and Z rather than
  TiCl4 or AlCl3 is deliberate: a reader must not be able to mistake this table for
  sourced data.
- `stoichiometry`: ASSUMED, X(s) + Y(g) -> Z(g), one mole each.

**Arithmetic chain.**

```
sum(nu dHf) = (+1)(-250) + (-1)(-400) + (-1)(-100)
            = -250 + 400 + 100 = 250.0 kJ/mol
sum(nu S)   = (+1)(300) + (-1)(80) + (-1)(210)
            = 300 - 80 - 210 = 10.0 J/(mol K)
T sum(nu S) = 1073.15 * 10.0 = 10731.5 J/mol = 10.7315 kJ/mol
dG = 250.0 - 10.7315 = 239.2685 kJ/mol
The sign is positive, so this fixture reaction is not spontaneous at 800 K under
standard states. That is the physically meaningful content of the check: a sign
error in the T dS term would give 260.73 kJ/mol, still positive, so a vector
testing only the sign would not catch it; the magnitude is what pins the term.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `delta_g_kJ_per_mol` | 239.2685 | abs 1e-09 | Three multiplications and four additions on operands near 100 to 400, then one subtraction of a 10.7 kJ/mol term from a 250 kJ/mol term, so no cancellation beyond one decimal digit. The accumulated bound is a few ulp of 250, near 1e-13 kJ/mol. 1e-9 kJ/mol admits that with four orders of margin, and a wrong sign on the entropy term shifts the answer by 21.5 kJ/mol, ten orders above the tolerance. |

---

## GV-16: Sensible heat to take cristobalite from 1200 K to 1700 K

- File: `data/golden/GV-16-thermal-enthalpy.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.thermal`
- Dispatch entry: `thermal_enthalpy`

**Reference.** Holland, T.J.B. and Powell, R. 2011, An improved and extended internally consistent thermodynamic dataset for phases of petrological interest, Journal of Metamorphic Geology 29:333-383, doi:10.1111/j.1525-1314.2010.00923.x, dataset ds62. The heat capacity coefficients for cristobalite recorded in the module are a = 72.7, b = 1.304e-3, c = -4129000.0, d = 0.0 in the Holland-Powell form Cp = a + bT + c/T^2 + d/sqrt(T).

**Inputs.**

```yaml
polymorph: cristobalite
t_from_K: 1200.0
t_to_K: 1700.0
include_landau: false
molar_mass_kg_per_mol: 0.0600843
```

**Provenance.**

- `polymorph`: SOURCED, cristobalite, chosen because its d coefficient is exactly zero, which removes
  the sqrt(T) term from the antiderivative and makes the integral hand-traceable in
  three terms rather than four.
- `t_from_K`: ASSUMED, 1200 K, above the 1143.15 K tridymite to cristobalite transition recorded in
  the module, so cristobalite is the stable phase over the whole interval and no
  transition enthalpy is omitted.
- `t_to_K`: ASSUMED, 1700 K, below the 1743.15 K cristobalite upper limit and below the 1996 K
  melting point, so no latent heat enters.
- `include_landau`: ASSUMED false. The Landau excess term in this module describes the alpha to beta
  quartz transition at 847 K and does not apply to cristobalite. Including it here
  would add heat capacity that belongs to a different phase.
- `molar_mass_kg_per_mol`: SOURCED, 0.0600843 kg/mol for SiO2, from the module's M_SIO2 constant. Used only to
  convert the molar result to a mass basis.

**Arithmetic chain.**

```
Antiderivative of Cp dT with d = 0:
  F(T) = a T + (b/2) T^2 - c/T
At T = 1700 K:
  a T      = 72.7 * 1700 = 123590.0
  (b/2)T^2 = 0.5 * 1.304e-3 * 2890000 = 1884.2800000000002
  -c/T     = 4129000.0 / 1700 = 2428.823529411765
  F(1700)  = 127903.10352941176 J/mol
At T = 1200 K:
  a T      = 72.7 * 1200 = 87240.0
  (b/2)T^2 = 0.5 * 1.304e-3 * 1440000 = 938.88
  -c/T     = 4129000.0 / 1200 = 3440.8333333333335
  F(1200)  = 91619.71333333333 J/mol
dH = F(1700) - F(1200) = 36283.39019607843 J/mol
On a mass basis: 36283.39019607843 / 0.0600843 = 603874.7259446882 J/kg
               = 603.8747259446882 kJ/kg
Sense check: 36283.39019607843 J/mol over 500 K is a mean Cp of 72.56678039215686
J/(mol K), which sits
between the endpoint values of Cp, as it must for a function that rises with T over
most of this interval. The c/T^2 term falls as T rises, so the mean is close to
a rather than far above it.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `delta_h_J_per_mol` | 36283.3901960784 | abs 1e-06 | Six multiplications, two divisions and one subtraction of two quantities near 1e5 whose difference is 3.6e4, so the cancellation costs less than half a decimal digit. The accumulated bound is a few ulp of 1.3e5, near 1e-10 J/mol. 1e-6 J/mol admits that with four orders of margin. A dropped c/T term would change the answer by 1012 J/mol, nine orders above the tolerance. |
| `delta_h_kJ_per_kg` | 603.8747259447 | abs 1e-06 | One division by the molar mass and one unit scaling, so the relative error is inherited from delta_h_J_per_mol, near 1e-15, which at 604 kJ/kg is 6e-13 absolute. 1e-6 kJ/kg admits that with six orders of margin. Reported on both bases because a plant heat balance is written per tonne and a thermodynamic table per mole, and the conversion factor is a place errors hide. |

---

## GV-17: Alpha quartz heat capacity at 700 K, base plus Landau excess, and their sum

- File: `data/golden/GV-17-thermal-heat-capacity.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.thermal`
- Dispatch entry: `thermal_heat_capacity`

**Reference.** Holland, T.J.B. and Powell, R. 2011, doi:10.1111/j.1525-1314.2010.00923.x, dataset ds62, for the quartz Cp coefficients a = 92.9, b = -6.42e-4, c = -714900.0, d = -716.1 recorded in the module. The Landau tricritical form with Tc(0) = 847.0 K and S_D = 4.95 J/(mol K) is the module's recorded parameterisation of the alpha to beta quartz transition.

**Inputs.**

```yaml
polymorph: alpha_quartz
temperature_K: 700.0
```

**Provenance.**

- `temperature_K`: ASSUMED, 700 K, chosen inside the alpha quartz field (below the 847 K transition) and
  far enough from it that the Landau term is finite and moderate rather than
  divergent. At T approaching Tc the excess grows without bound and no tolerance would
  be meaningful.

**Arithmetic chain.**

```
Base heat capacity, Holland-Powell form Cp = a + bT + c/T^2 + d/sqrt(T):
  a          = 92.9
  bT         = -6.42e-4 * 700 = -0.4494
  c/T^2      = -714900.0 / 490000 = -1.4589795918367348
  d/sqrt(T)  = -716.1 / 26.457513110645905 = -27.066035912190763
  Cp_base    = 92.9 - 0.4494 - 1.4589795918367348 - 27.066035912190763
             = 63.92558449597251 J/(mol K)
Landau excess. The order parameter is q = ((Tc - T)/Tc0)^(1/4) with Tc = Tc0 at
zero pressure:
  (847.0 - 700.0)/847.0 = 0.17355371900826447
  q = 0.17355371900826447^0.25 = 0.645443870875331
  q^2 = 0.4165977904505309
  Cp_excess = T S_D / (2 Tc0 q^2)
            = 700 * 4.95 / (2 * 847.0 * 0.4165977904505309)
            = 3465.0 / 705.7166570231993 = 4.909902530309829 J/(mol K)
Total = 63.92558449597251 + 4.909902530309829 = 68.83548702628234 J/(mol K)
The excess is 7.68 percent of the base at 700 K, which is the size of the effect a
kiln heat balance would miss by using a tabulated Cp that omits the transition.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `cp_base_J_per_mol_K` | 63.925584496 | abs 1e-09 | One square root, two divisions, one multiplication and three additions of terms between 1 and 93, so the bound is a few ulp of 93, near 4e-14 J/(mol K). 1e-9 admits that with five orders of margin. A dropped d/sqrt(T) term would give 90.99, detected by 27 units. |
| `landau_excess_J_per_mol_K` | 4.9099025303 | abs 1e-09 | One fourth root and two divisions. The fourth root carries 1 ulp, and its square appears in the denominator, so the relative error is about 1e-15, giving 5e-15 J/(mol K) at this magnitude. 1e-9 admits it with six orders of margin. The exponent 1/4 is tricritical and specific: a 1/2 exponent would give 2.97 J/(mol K), detected by 1.9 units. |
| `cp_total_J_per_mol_K` | 68.8354870263 | abs 1e-09 | The sum of the two terms above, so the bound is the sum of their bounds, near 5e-14 J/(mol K). 1e-9 admits it. |
| `sum_check_J_per_mol_K` | 68.83548702628234 | abs 1e-12 | This output adds the two SEPARATELY returned quantities and compares against the total the module returns with include_landau true, so it tests that the total really is base plus excess and not a differently parameterised whole. It is a self-consistency check between three module calls, so the tolerance is set at the arithmetic floor of one addition, a few ulp near 1.4e-14, rather than at the looser 1e-9 used for the physical values. |

---

## GV-18: Flotation first-order recovery, two-product mass balance and a partition curve

- File: `data/golden/GV-18-separation.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.physics.separation`
- Dispatch entry: `separation_recovery`

**Reference.** Garcia-Zuniga, H. 1935, first-order flotation rate equation, as restated in Wills, B.A. and Finch, J.A. 2016, Wills' Mineral Processing Technology, 8th edition, Elsevier. The two-product formula and the Ecart Probable definition of a partition curve are standard results in that text.

**Inputs.**

```yaml
time_s: 45.0
k_per_s: 0.08
ultimate_recovery: 0.85
feed_grade: 2.5
concentrate_grade: 18.0
tailing_grade: 0.4
x50: 100.0
ep: 25.0
```

**Provenance.**

- `time_s`: ASSUMED, 45 s residence time.
- `k_per_s`: ASSUMED, 0.08 1/s, so k t = 3.6 exactly, chosen for traceability.
- `ultimate_recovery`: ASSUMED, 0.85. Real rate fits need four to six timed concentrates; a single pair
  cannot separate k from R_inf, which the module states.
- `feed_grade`: ASSUMED, 2.5 percent, in any consistent unit.
- `concentrate_grade`: ASSUMED, 18.0.
- `tailing_grade`: ASSUMED, 0.4.
- `x50`: ASSUMED, 100 um cut size.
- `ep`: ASSUMED, 25 um Ecart Probable, a realistic classifier sharpness.

**Arithmetic chain.**

```
First-order recovery, R = R_inf (1 - exp(-k t)):
  k t = 0.08 * 45 = 3.6
  exp(-3.6) = 0.02732372244729256
  1 - 0.02732372244729256 = 0.9726762775527075
  R = 0.85 * 0.9726762775527075 = 0.8267748359198013
Two-product formula, mass yield Y = (f - t)/(c - t):
  f - t = 2.5 - 0.4 = 2.1
  c - t = 18.0 - 0.4 = 17.6
  Y = 2.1/17.6 = 0.11931818181818181
Element recovery R = Y c / f = 0.11931818181818181 * 18.0 / 2.5
  = 2.1477272727272725 / 2.5 = 0.859090909090909
Cross-check by the alternative form Y c/f = (f-t)c / ((c-t)f)
  = 2.1 * 18 / (17.6 * 2.5) = 37.8/44.0 = 0.859090909090909, which agrees to all
printed digits by both routes.
Partition curve. The module uses a logistic whose scale follows from the
definition that the partition rises from 0.25 to 0.75 across 2 Ep:
  s = Ep / ln 3 = 25 / 1.0986122886681098 = 22.755980665670933
  at x = x50 the exponent is zero, so the partition is exactly 0.5
  at x = x50 + Ep the exponent is Ep/s = ln 3, so the partition is
    1/(1 + exp(-ln 3)) = 1/(1 + 1/3) = 3/4 = 0.75 exactly by construction
The 0.75 value is the definition of Ep, so this output is a definitional check and
not an arithmetic one: if the module used Ep/1.349 (the normal-distribution
convention) rather than Ep/ln 3, the value at x50 + Ep would be 0.7699.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `flotation_recovery` | 0.8267748359 | abs 1e-09 | One exponential, one subtraction from 1 and one multiplication. exp carries 1 ulp, and 1 - 0.0273 loses nothing, so the bound is near 2e-16. 1e-9 admits it with seven orders of margin, and a missing R_inf factor would give 0.9727, detected by 0.146. |
| `mass_yield` | 0.11931818181818182 | abs 1e-12 | Two exact subtractions of one-decimal values (which are not binary-exact, so each carries 1 ulp near 4e-16) and one division. The bound is near 3e-16. 1e-12 admits it with three orders of margin. |
| `element_recovery` | 0.8590909090909091 | abs 1e-12 | One multiplication and one division on top of the mass yield, so the bound is near 1e-15. 1e-12 admits it. The alternative algebraic route in the arithmetic block agrees to all printed digits, which is why a tolerance this tight is defensible rather than lucky. |
| `logistic_scale` | 22.7559806657 | abs 1e-09 | One division by ln 3, which carries 1 ulp, so the relative error is near 2e-16 and the absolute error near 5e-15 at 22.7560. 1e-9 admits it with six orders of margin. |
| `partition_at_x50` | 0.5 | abs 1e-12 | The logistic at zero exponent is 1/(1 + exp(0)) = 0.5, with exp(0) exactly 1 in IEEE 754, so the result is exactly 0.5. 1e-12 is used rather than 0.0 only because the argument (x - x50)/s is formed by a subtraction of equal floats, which is exactly zero, then divided, so the tolerance could be zero; 1e-12 guards against an implementation that shifts the curve by a half-bin. |
| `partition_at_x50_plus_ep` | 0.75 | abs 1e-12 | 1/(1 + exp(-ln 3)) where exp(-ln 3) differs from 1/3 by 1 ulp, so the result differs from 0.75 by about 4e-17. 1e-12 admits that with four orders of margin and is four orders tighter than the 0.0199 discrepancy a different Ep convention would produce, so the check distinguishes the conventions. |

---

## GV-19: Two-stage flowsheet with a cleaner reject recycle, solved to a fixed point by hand

- File: `data/golden/GV-19-stream-balance-recycle.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.plant.streams`
- Dispatch entry: `stream_balance_recycle`

**Reference.** The fixed point of a single recycle loop is elementary algebra and needs no source. The convergence criterion used by the module (damped successive substitution, convergent when the loop gain is below unity) is the standard result for this iteration and the loop gain is computed explicitly below.

**Inputs.**

```yaml
name: rougher-cleaner with recycle
element: Fe
feed_kg_s: 100.0
feed_ppm: 1000.0
units:
- name: rougher
  mass_yield: 0.8
  element_removal: 0.6
- name: cleaner
  mass_yield: 0.75
  element_removal: 0.5
links:
- - rougher
  - product
  - cleaner
- - cleaner
  - reject
  - rougher
damping: 0.5
tol: 1.0e-10
max_iter: 200
```

**Provenance.**

- `feed_kg_s`: ASSUMED, 100 kg/s, a round basis. Nothing depends on the absolute scale.
- `feed_ppm`: ASSUMED, 1000 ppm Fe. A scenario value, not a Vikarabad assay.
- `mass_yield`: ASSUMED, 0.80 rougher and 0.75 cleaner.
- `element_removal`: ASSUMED, 0.60 of the Fe entering the rougher leaves with its reject, 0.50 for the
  cleaner. Both are below the values a real magnetic or flotation stage would need to
  be qualified against.
- `damping`: ASSUMED, 0.5, the module default. The fixed point does not depend on the damping; only
  the iteration count does.
- `tol`: ASSUMED, 1e-10 on the recycle mass flow, the module default.

**Arithmetic chain.**

```
Solids. Let R be the recycle flow. The rougher sees F + R, passes 0.80 of it to the
cleaner, and the cleaner rejects 0.25 of what it receives back to the rougher:
  R = 0.25 * 0.80 * (F + R) = 0.20 (F + R)
  R (1 - 0.20) = 0.20 F
  R = 0.20 * 100 / 0.80 = 25.0 kg/s
Loop gain is 0.20, well below unity, so the iteration converges.
  rougher feed   = 100 + 25 = 125.0 kg/s
  rougher product = 0.80 * 125 = 100.0 kg/s
  rougher reject  = 125 - 100 = 25.0 kg/s (terminal)
  cleaner product = 0.75 * 100 = 75.0 kg/s (terminal)
  cleaner reject  = 100 - 75 = 25.0 kg/s (recycled, consistent with R above)
Closure: terminal out = 25.0 + 75.0 = 100.0 kg/s = feed, so the closure error is 0.
Overall yield = 75.0/100.0 = 0.75. WITHOUT the recycle the same two stages give
0.80 * 0.75 = 0.60, so the recycle raises yield from 0.60 to 0.75 by returning
material that would otherwise have been discarded.
Iron. Feed Fe = 100 * 1000e-6 = 0.1 kg/s. The rougher passes 0.40 of the Fe it
sees, the cleaner rejects 0.50 of what it gets, so the Fe loop gain is
0.40 * 0.50 = 0.20, the same as the solids gain in this case:
  Fe_recycle = 0.20 (0.1 + Fe_recycle) -> Fe_recycle = 0.02/0.80 = 0.025 kg/s
  rougher feed Fe = 0.1 + 0.025 = 0.125 kg/s over 125 kg/s = 1000 ppm
  rougher product Fe = 0.40 * 0.125 = 0.05 kg/s over 100 kg/s = 500 ppm
  rougher reject Fe  = 0.60 * 0.125 = 0.075 kg/s over 25 kg/s = 3000 ppm
  cleaner product Fe = 0.50 * 0.05 = 0.025 kg/s over 75 kg/s = 333.3333333 ppm
  cleaner reject Fe  = 0.50 * 0.05 = 0.025 kg/s over 25 kg/s = 1000 ppm
Element closure: terminal Fe out = 0.075 + 0.025 = 0.1 kg/s = feed Fe, exactly.
That the recycle stream returns at exactly the feed grade of 1000 ppm is a
coincidence of these numbers, not a general result; it happens because the solids
and element loop gains are equal here.
Feasibility. A unit cannot reject more of an element than arrives, which caps the
feed concentration at (1 - mass_yield)/removal:
  rougher: (1 - 0.80)/0.60 = 0.3333333333 mass fraction
  cleaner: (1 - 0.75)/0.50 = 0.5 mass fraction
Both caps are five orders above the 0.001 mass fraction actually present, so the
balance is feasible with large margin.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `converged` | 1.0 | abs 0.0 | A boolean encoded as 1.0. No rounding to admit, so the tolerance is exactly zero. If the solver did not converge every other output in this vector is meaningless, so this one is checked exactly. |
| `overall_yield` | 0.75 | abs 1e-09 | The fixed point is approached iteratively, not solved in closed form, so the error is set by the solver tolerance of 1e-10 on the recycle mass flow rather than by floating point. A 1e-10 kg/s error in a 25 kg/s recycle propagates to about 1e-12 in the yield. 1e-9 admits that with three orders of margin. This is the reason the tolerance is looser than the 1e-12 used for closed-form vectors: the convergence criterion, not the arithmetic, is the binding term. |
| `closure_error` | 0.0 | abs 1e-10 | The module raises a MassBalanceError if this exceeds its own CLOSURE_TOL, so any value it returns is already below that threshold. The tolerance here matches the solver tolerance of 1e-10, which is the level at which the iteration was stopped. |
| `element_closure` | 0.0 | abs 1e-10 | As closure_error, set by the 1e-10 solver tolerance. The element balance is enforced by construction inside each unit (the reject composition is the residual), so the only error is the unconverged part of the recycle. |
| `recycle_kg_s` | 25.0 | abs 1e-09 | The quantity the iteration converges on, stopped at 1e-10 absolute by the tol input. 1e-9 admits one order above the stopping criterion, which covers the final damped step. Reported because it is the algebraic answer 25.0 kg/s derived above, so agreement means the solver found the right fixed point and not merely a stationary one. |
| `rougher_feed_kg_s` | 125.0 | abs 1e-09 | Feed plus recycle, so it inherits the recycle's 1e-10 convergence error. |
| `rougher_product_kg_s` | 100.0 | abs 1e-09 | Same basis as rougher_feed_kg_s. Feed plus recycle, so it inherits the recycle's 1e-10 convergence error. |
| `cleaner_product_kg_s` | 75.0 | abs 1e-09 | Same basis as rougher_feed_kg_s. Feed plus recycle, so it inherits the recycle's 1e-10 convergence error. |
| `rougher_reject_kg_s` | 25.0 | abs 1e-09 | Same basis as rougher_feed_kg_s. Feed plus recycle, so it inherits the recycle's 1e-10 convergence error. |
| `rougher_feed_ppm` | 1000.0 | abs 1e-06 | A concentration in ppm, so the same relative convergence error of 1e-12 becomes 1e-9 absolute at 1000 ppm. 1e-6 ppm admits that with three orders of margin and is still six orders below any real assay resolution (ICP-MS on a 1000 ppm Fe stream resolves perhaps 1 ppm), so the tolerance tests the solver and not the instrument. |
| `rougher_product_ppm` | 500.0 | abs 1e-06 | Same basis as rougher_feed_ppm. A concentration in ppm, so the same relative convergence error of 1e-12 becomes 1e-9 absolute at 1000 ppm. 1e-6 ppm admits that with three orders of margin and is still six orders below any real assay resolution (ICP-MS on a 1000 ppm Fe stream resolves perhaps 1 ppm), so the tolerance tests the solver and not the instrument. |
| `cleaner_product_ppm` | 333.3333333333 | abs 1e-06 | As rougher_feed_ppm. This output is 1000/3 ppm, which is not exactly representable, so it also carries one rounding near 6e-14 ppm. The convergence term dominates. |
| `rougher_reject_ppm` | 3000.0 | abs 1e-06 | Same basis as rougher_feed_ppm. A concentration in ppm, so the same relative convergence error of 1e-12 becomes 1e-9 absolute at 1000 ppm. 1e-6 ppm admits that with three orders of margin and is still six orders below any real assay resolution (ICP-MS on a 1000 ppm Fe stream resolves perhaps 1 ppm), so the tolerance tests the solver and not the instrument. |
| `recycle_ppm` | 1000.0 | abs 1e-06 | Same basis as rougher_feed_ppm. A concentration in ppm, so the same relative convergence error of 1e-12 becomes 1e-9 absolute at 1000 ppm. 1e-6 ppm admits that with three orders of margin and is still six orders below any real assay resolution (ICP-MS on a 1000 ppm Fe stream resolves perhaps 1 ppm), so the tolerance tests the solver and not the instrument. |
| `max_feasible_feed_fraction_rougher` | 0.3333333333333333 | abs 1e-12 | A closed-form ratio (1 - y)/r computed once, not iterated, so the bound is one division, 1 ulp near 6e-17. 1e-12 admits it. Tighter than the other outputs in this vector precisely because this one does not pass through the solver. |
| `max_feasible_feed_fraction_cleaner` | 0.5 | abs 1e-12 | Same basis as max_feasible_feed_fraction_rougher. A closed-form ratio (1 - y)/r computed once, not iterated, so the bound is one division, 1 ulp near 6e-17. 1e-12 admits it. Tighter than the other outputs in this vector precisely because this one does not pass through the solver. |

---

## GV-20: OEE, loss decomposition, and the bottleneck of a two-unit line

- File: `data/golden/GV-20-capacity-oee.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.plant.capacity`
- Dispatch entry: `capacity_oee`

**Reference.** Nakajima, S. 1988, Introduction to TPM: Total Productive Maintenance, Productivity Press. The availability times performance times quality decomposition of OEE is that text's definition.

**Inputs.**

```yaml
availability: 0.92
performance: 0.88
quality: 0.97
planned_hours: 7500.0
units:
- name: kiln
  rate_t_per_h: 12.0
  tonnes_per_tonne_product: 1.4
- name: leach
  rate_t_per_h: 8.0
  tonnes_per_tonne_product: 1.0
```

**Provenance.**

- `availability`: ASSUMED, 0.92, a plausible figure for a rotary kiln with planned maintenance windows.
- `performance`: ASSUMED, 0.88.
- `quality`: ASSUMED, 0.97.
- `planned_hours`: ASSUMED, 7500 h/yr, which is 85.6 percent of the 8760 h year, leaving 1260 h for
  planned shutdown. The module caps this at 8760.
- `rate_t_per_h`: ASSUMED, 12 t/h kiln and 8 t/h leach, both on a FEED basis.
- `tonnes_per_tonne_product`: ASSUMED, 1.4 t of kiln feed per t of final product and 1.0 for the leach. The 1.4
  encodes the mass loss upstream of the leach; getting this ratio wrong is how a
  nameplate capacity comes out 40 percent too high.

**Arithmetic chain.**

```
OEE = 0.92 * 0.88 * 0.97
  0.92 * 0.88 = 0.8096
  0.8096 * 0.97 = 0.785312
Loss decomposition, which must sum to 1 - OEE:
  availability loss = 1 - 0.92 = 0.08
  performance loss  = 0.92 * (1 - 0.88) = 0.92 * 0.12 = 0.1104
  quality loss      = 0.92 * 0.88 * (1 - 0.97) = 0.8096 * 0.03
                    = 0.024287999999999997 in double precision (0.024288 exactly)
  sum = 0.08 + 0.1104 + 0.024288 = 0.21468800000000002 in double precision
  1 - OEE = 1 - 0.785312 = 0.214688
  The two routes differ by 2e-17, one ulp, which sets the tolerance below.
The nesting matters: a naive decomposition using (1 - performance) rather than
availability times (1 - performance) would give 0.12 and the three losses would
sum to 0.23, overstating the total loss by 7 percent of capacity.
Effective feed capacities:
  kiln:  12 * 7500 * 0.785312 = 90000 * 0.785312 = 70678.08 t/yr of kiln feed
  leach:  8 * 7500 * 0.785312 = 60000 * 0.785312 = 47118.72 t/yr of leach feed
Product capacities, dividing by the feed-per-product ratio:
  kiln:  70678.08 / 1.4 = 50484.34285714286 t/yr of product
  leach: 47118.72 / 1.0 = 47118.72 t/yr of product
The line rate is the minimum, 47118.72 t/yr, so the LEACH is the bottleneck even
though its nameplate feed rate is lower AND it has the smaller ratio. Ranking on
nameplate feed rate alone would have flagged the leach too, but ranking on
effective feed capacity would have given the same wrong-for-the-right-reason
answer; the product basis is what makes the comparison valid.
Bottleneck margin, the headroom of the second-tightest unit over the tightest:
  (50484.34285714286 - 47118.72) / 47118.72 = 3365.6228571428583 / 47118.72
  = 0.07142857142857145, which is 1/14 to 15 digits. Exactly: 90000/(1.4*60000) - 1
  = 15/14 - 1 = 1/14.
Utilisations at the line rate:
  kiln:  47118.72 * 1.4 / 70678.08 = 65966.208/70678.08 = 0.9333333333333332,
         which is 14/15 to the last ulp
  leach: 47118.72 * 1.0 / 47118.72 = 1.0
Slack:
  kiln:  50484.34285714286 - 47118.72 = 3365.6228571428583 t/yr
  leach: 0.0 t/yr
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `oee` | 0.785312 | abs 1e-12 | Two multiplications of two-decimal values, none binary-exact, so two roundings near 1e-17 each. 1e-12 admits that with five orders of margin. |
| `availability_loss` | 0.08 | abs 1e-12 | One subtraction from 1, 1 ulp near 1e-17. |
| `performance_loss` | 0.1104 | abs 1e-12 | One subtraction and one multiplication, 2 ulp near 3e-17. |
| `quality_loss` | 0.024288 | abs 1e-12 | Two multiplications and one subtraction, 3 ulp near 4e-17. |
| `loss_sum` | 0.214688 | abs 1e-12 | The three losses summed. This output is the decomposition identity, so it must match one_minus_oee below to the same tolerance; three terms each carrying 4e-17 sum to at most 1.2e-16. 1e-12 admits that and would catch the un-nested decomposition described in the arithmetic block, which differs by 0.0159. |
| `one_minus_oee` | 0.214688 | abs 1e-12 | One subtraction from 1 of the OEE, so the OEE's error plus 1 ulp, near 1e-16. Reported alongside loss_sum so the identity is checked between two independently computed quantities rather than asserted. |
| `effective_capacity_kiln` | 70678.08 | abs 1e-06 | Two multiplications at a magnitude near 7e4, so the bound is a few ulp of 7e4, near 3e-11 t/yr. 1e-6 t/yr (one gram per year) admits that with five orders of margin. A tonnage tolerance of 1e-6 is far below any real measurement, which is intended: the check is arithmetic, and the input uncertainty is carried in the provenance block instead. |
| `effective_capacity_leach` | 47118.72 | abs 1e-06 | Same basis as effective_capacity_kiln. Two multiplications at a magnitude near 7e4, so the bound is a few ulp of 7e4, near 3e-11 t/yr. 1e-6 t/yr (one gram per year) admits that with five orders of margin. A tonnage tolerance of 1e-6 is far below any real measurement, which is intended: the check is arithmetic, and the input uncertainty is carried in the provenance block instead. |
| `product_capacity_kiln` | 50484.3428571429 | abs 1e-06 | One further division by 1.4, which is not binary-exact, so 1 more ulp. The bound remains near 3e-11 t/yr. 1e-6 admits it. |
| `product_capacity_leach` | 47118.72 | abs 1e-06 | Division by exactly 1.0, which is exact, so the bound is effective_capacity_leach's. |
| `line_rate_t_per_yr` | 47118.72 | abs 1e-06 | A minimum over the product capacities, so it carries whichever bound the winner had, near 3e-11 t/yr. |
| `bottleneck_is_leach` | 1.0 | abs 0.0 | An identity encoded as 1.0 under a key naming the bottleneck. The dispatch emits the key "bottleneck_is_<name>", so a wrong bottleneck produces a MISSING key rather than a wrong value, and the test fails on the missing key. That is why the tolerance can be exactly zero. |
| `bottleneck_margin` | 0.07142857142857142 | abs 1e-12 | A ratio of differences near 3e3 over 4.7e4. The subtraction loses about one decimal digit of the 5e4 magnitudes, so the bound is near 1e-15 rather than 1e-17. 1e-12 admits that with three orders of margin. The exact value 1/14 is derived in the arithmetic block, so this checks against a rational and not against a printed float. |
| `utilisation_kiln` | 0.9333333333333333 | abs 1e-12 | One multiplication and one division giving 14/15, which is not binary- exact, so 2 ulp near 2e-16. 1e-12 admits it. |
| `utilisation_leach` | 1.0 | abs 1e-12 | A ratio of a quantity to itself, exactly 1.0 in IEEE 754. 1e-12 guards only against a different route to the same number. |
| `slack_kiln` | 3365.6228571429 | abs 1e-06 | A difference of two tonnages near 5e4 whose result is 3.4e3, so one decimal digit of cancellation and a bound near 1e-11 t/yr. 1e-6 admits it. |
| `slack_leach` | 0.0 | abs 1e-09 | The bottleneck's slack is its capacity minus itself, exactly zero in IEEE 754. The tolerance is 1e-9 t/yr rather than zero because the two sides are converted through pint separately, which can introduce one rounding. |

---

## GV-21: Four-stage yield cascade and the throughput factor each stage must carry

- File: `data/golden/GV-21-yield-cascade.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.plant.yield_cascade`
- Dispatch entry: `yield_cascade`

**Reference.** The product of stage yields and the reciprocal-of-downstream-product throughput factor are elementary and need no source. The engineering point they encode, that an upstream unit must be sized for the inverse of everything downstream of it, is standard process design practice.

**Inputs.**

```yaml
stage_yields:
- 0.95
- 0.88
- 0.92
- 0.99
```

**Provenance.**

- `stage_yields`: ASSUMED, four stages at 0.95, 0.88, 0.92 and 0.99. Chosen so that no two partial
  products coincide, which means a throughput factor computed with an off-by-one index
  would give a visibly different number rather than a coincidentally correct one. That
  off-by-one is a real failure mode recorded in this repository's corrections file,
  where a throughput factor was wrong in the fourth decimal.

**Arithmetic chain.**

```
Cascade yield, the product of all four:
  0.95 * 0.88 = 0.836
  0.836 * 0.92 = 0.76912
  0.76912 * 0.99 = 0.7614288
Throughput factor for stage i is the reciprocal of the product of stage i and
everything downstream of it, because a tonne of final product requires
1/(that product) tonnes entering stage i:
  stage 0: product 0.95*0.88*0.92*0.99 = 0.7614288  -> f0 = 1.3133204312734166
  stage 1: product      0.88*0.92*0.99 = 0.801504   -> f1 = 1.2476544097097457
  stage 2: product           0.92*0.99 = 0.9108     -> f2 = 1.097935880544576
  stage 3: product                0.99 = 0.99       -> f3 = 1.0101010101010102
Each factor is strictly larger than the next, and f0 equals 1/cascade_yield, which
is the identity that makes the sequence checkable: 1/0.7614288 = 1.3133204312734166.
The off-by-one failure mode: computing stage 1's factor as 1/(0.95*0.92*0.99),
which is the product EXCLUDING its own yield and including an upstream one, gives
1.1557219795206064 rather than 1.2476544097097457, a 7.4 percent undersizing of
that unit.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `cascade_yield` | 0.7614288 | abs 1e-12 | Three multiplications of two-decimal values, none binary-exact, so three roundings of about 1e-17 each. 1e-12 admits that with five orders of margin and is four orders tighter than the fourth-decimal error this class of calculation actually suffered in this repository. |
| `throughput_factor_0` | 1.3133204313 | abs 1e-09 | One division into a four-term product, so four roundings plus one, near 6e-16 relative or 8e-16 absolute at 1.31. 1e-9 admits that with six orders of margin and is six orders tighter than the 0.0920 discrepancy an off-by- one index produces, so it separates a rounding difference from the real defect. |
| `throughput_factor_1` | 1.2476544097 | abs 1e-09 | As throughput_factor_0, one fewer multiplication. The 0.0919 gap to the off- by-one value is seven orders above this tolerance. |
| `throughput_factor_2` | 1.0979358805 | abs 1e-09 | Same basis as throughput_factor_0. One division into a four-term product, so four roundings plus one, near 6e-16 relative or 8e-16 absolute at 1.31. 1e-9 admits that with six orders of margin and is six orders tighter than the 0.0920 discrepancy an off-by-one index produces, so it separates a rounding difference from the real defect. |
| `throughput_factor_3` | 1.0101010101 | abs 1e-09 | One division, 1/0.99, whose exact value 100/99 is not binary-exact, so 1 ulp near 2e-16. 1e-9 admits it. This is the output that pins the index convention at the downstream end: a factor of 1.0 here would mean the last stage's own yield was excluded. |

---

## GV-22: Cpk from six lot assays, its confidence interval, and the off-spec fraction under two tail models

- File: `data/golden/GV-22-spc-off-spec.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.plant.yield_cascade`
- Dispatch entry: `spc`

**Reference.** Montgomery, D.C. 2012, Introduction to Statistical Quality Control, 7th edition, Wiley, section 8.7, for the large-sample variance of the Cpk estimator, Var(Cpk) = (1/(9n))(1 + 9 Cpk^2 / 2).

**Inputs.**

```yaml
assays:
- 10.0
- 12.0
- 14.0
- 16.0
- 18.0
- 20.0
usl: 30.0
confidence: 0.95
cpk_target: 1.33
```

**Provenance.**

- `assays`: ASSUMED, six lots at 10 to 20 ppm in steps of 2. An arithmetic sequence was chosen so
  the mean and the sum of squared deviations are exact integers, which makes the
  standard deviation a clean sqrt(14). These are not measurements; no lot assays exist
  for the Vikarabad deposit.
- `usl`: ASSUMED, 30 ppm upper specification limit, one-sided, which is the form an impurity
  spec takes (there is no lower limit on purity).
- `confidence`: ASSUMED, 0.95, two-sided, so the z multiplier is the 0.975 quantile.
- `cpk_target`: SOURCED as a convention, 1.33, the customary four-sigma capability target. It is a
  convention rather than a measurement and is used here only to invert for the process
  mean it would require.

**Arithmetic chain.**

```
n = 6, mean = (10+12+14+16+18+20)/6 = 90/6 = 15.0
Sum of squared deviations from the mean:
  (-5)^2 + (-3)^2 + (-1)^2 + 1^2 + 3^2 + 5^2 = 25+9+1+1+9+25 = 70
Sample variance (ddof = 1) = 70/5 = 14.0
Sample standard deviation = sqrt(14) = 3.7416573867739413
One-sided Cpk = (USL - mean)/(3 sigma) = (30 - 15)/(3 * 3.7416573867739413)
  = 15/11.224972160321824 = 1.3363062095621219
Confidence interval, Montgomery's large-sample variance:
  Cpk^2 = 1.7857142857142856
  Var = (1/(9*6)) * (1 + 9*1.7857142857142856/2)
      = (1/54) * (1 + 8.035714285714285)
      = 9.035714285714285/54 = 0.1673280423280423
  sd(Cpk) = sqrt(0.1673280423280423) = 0.40905750491592535
  z at the 0.975 quantile = 1.9599639845400532
  half width = 1.9599639845400532 * 0.40905750491592535 = 0.8017379772410295
  CI = 1.3363062095621219 -/+ 0.8017379772410295
     = (0.5345682323210924, 2.1380441868031514)
The interval spans a factor of four in Cpk from six lots, which is the point:
a Cpk of 1.34 from six lots is not evidence of a capable process.
Off-spec fraction, normal model:
  z = (USL - mean)/sigma = 15/3.7416573867739413 = 4.008918628686366
  z/sqrt(2) = 2.8347335475692037
  upper tail = 0.5 erfc(2.8347335475692037) = 3.0498714448247723e-05
Off-spec fraction, lognormal model matched to the same mean and variance by the
method of moments:
  cv^2 = (sigma/mean)^2 = (3.7416573867739413/15)^2 = 0.06222222222222223
  s^2 = ln(1 + cv^2) = ln(1.0622222222222222) = 0.06036314972709054
  s = sqrt(0.06036314972709054) = 0.2456891322934137
  mu_log = ln(15) - 0.5 s^2 = 2.70805020110221 - 0.03018157486354527
         = 2.6778686262386646
  z_ln = (ln 30 - mu_log)/s = (3.4011973816621555 - 2.6778686262386646)/0.2456891322934137
       = 2.9440811999761434
  upper tail = 0.5 erfc(z_ln/sqrt(2)) = 0.0016195750169419518
The lognormal tail is 53.10 times the normal tail at the same mean and standard
deviation. That ratio, not either number alone, is the reason the tail model has
to be declared: an impurity distribution is bounded below at zero and is usually
right-skewed, so the normal model understates the off-spec rate.
Required mean for the target Cpk:
  mean = USL - 3 Cpk_target sigma = 30 - 3*1.33*3.7416573867739413
       = 30 - 14.929212973228026 = 15.070787026771974
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `mean` | 15.0 | abs 1e-12 | An exact integer quotient 90/6 in floating point, so exactly 15.0. 1e-12 guards only against a different summation order. |
| `sigma` | 3.7416573868 | abs 1e-09 | One square root of an exact integer 14, so 1 ulp near 4e-16. 1e-9 admits it with six orders of margin. The ddof convention is what this pins: a population standard deviation (ddof = 0) would be sqrt(70/6) = 3.4157, detected by 0.33. |
| `n_lots` | 6.0 | abs 0.0 | An integer count of lots. Any other value means samples were dropped or duplicated, which is a wrong answer and not a rounding difference, so the tolerance is exactly zero. |
| `cpk` | 1.3363062096 | abs 1e-09 | One division by 3 sigma, so it inherits sigma's relative error of 1e-16, giving 1.3e-16 absolute. 1e-9 admits it with six orders of margin. |
| `ci_low` | 0.5345682323 | abs 1e-08 | The chain runs Cpk -> Cpk^2 -> Var -> sqrt -> times the normal quantile -> subtract. The normal quantile is the least accurate step: it is computed by a rational approximation or an iterative inverse-erf, whose accuracy is typically 1e-15 relative rather than 1 ulp. Propagating 1e-15 relative through a 0.8017 half width gives 8e-16 absolute. 1e-8 admits that with seven orders of margin and covers a slightly different quantile implementation (scipy against a bisection on erf), which is the practical reason this tolerance is two orders looser than cpk's. |
| `ci_high` | 2.1380441868 | abs 1e-08 | Same basis as ci_low. The chain runs Cpk -> Cpk^2 -> Var -> sqrt -> times the normal quantile -> subtract. The normal quantile is the least accurate step: it is computed by a rational approximation or an iterative inverse- erf, whose accuracy is typically 1e-15 relative rather than 1 ulp. Propagating 1e-15 relative through a 0.8017 half width gives 8e-16 absolute. 1e-8 admits that with seven orders of margin and covers a slightly different quantile implementation (scipy against a bisection on erf), which is the practical reason this tolerance is two orders looser than cpk's. |
| `off_spec_normal` | 3.0498714448247723e-05 | rel 1e-09 | A RELATIVE tolerance, because the value is 3e-05 and an absolute tolerance suited to the other outputs in this vector would pass any tail probability. The chain is one division and one complementary error function. erfc is accurate to about 1e-15 relative in its argument, and at z/sqrt(2) = 2.835 the tail is not in the regime where erfc loses precision (that begins above about z = 6, where the result underflows toward zero), so the relative bound is near 1e-15. 1e-9 relative admits that with six orders of margin. |
| `off_spec_lognormal` | 0.0016195750169 | rel 1e-09 | A relative tolerance for the same reason as off_spec_normal. One extra log1p, one log and one sqrt enter, each 1 ulp, so the relative bound rises to about 5e-15. 1e-9 relative admits it and is five orders tighter than the factor-of-53.10 difference from the normal model, so the two models cannot be confused by a tolerance this size. |
| `required_mean_at_target_cpk` | 15.0707870268 | abs 1e-09 | One multiplication and one subtraction from 30, where the subtracted term is 14.93, so no significant cancellation. The bound is a few ulp of 30, near 7e-15. 1e-9 admits it with five orders of margin. |

---

## GV-23: M/M/1 and Allen-Cunneen waiting time, and the reduction the queueing approximation must satisfy

- File: `data/golden/GV-23-queueing.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.plant.scheduling`
- Dispatch entry: `queueing`

**Reference.** Allen, A.O. 1990, Probability, Statistics and Queueing Theory with Computer Science Applications, 2nd edition, Academic Press, for the G/G/1 waiting time approximation Wq = (rho/(1-rho)) * ((ca^2 + cs^2)/2) / mu. The M/M/1 result Wq = rho/(mu(1-rho)) is exact and standard.

**Inputs.**

```yaml
arrival_rate: 0.6
service_rate: 0.9
cv_arrival: 1.0
cv_service: 0.5
```

**Provenance.**

- `arrival_rate`: ASSUMED, 0.6 lots per hour.
- `service_rate`: ASSUMED, 0.9 lots per hour, giving rho = 2/3 exactly, comfortably below 1 so a steady
  state exists. The module raises at rho >= 1 rather than returning a large number.
- `cv_arrival`: ASSUMED, 1.0, Poisson arrivals.
- `cv_service`: ASSUMED, 0.5, a service process half as variable as exponential, which is what a batch
  kiln with a fixed cycle plus variable loading looks like.

**Arithmetic chain.**

```
rho = 0.6/0.9 = 0.6666666666666666 (2/3)
1 - rho = 0.33333333333333337 in double precision
M/M/1 exact: Wq = rho/(mu (1 - rho)) = 0.6666666666666666/(0.9*0.33333333333333337)
  = 0.6666666666666666/0.30000000000000004 = 2.222222222222222 h
Exactly, as a rational: (2/3)/(0.9 * 1/3) = (2/3)/(3/10) = 20/9 = 2.2222222222222223 h
Allen-Cunneen: Wq = (rho/(1-rho)) * ((ca^2 + cs^2)/2) / mu
  rho/(1-rho) = 0.6666666666666666/0.33333333333333337 = 1.9999999999999998
  (1.0^2 + 0.5^2)/2 = 1.25/2 = 0.625
  Wq = 1.9999999999999998 * 0.625 / 0.9 = 1.2500000000000004/0.9
     = 1.3888888888888886 h
Reduction check: at ca = cs = 1 the approximation must reduce to M/M/1:
  (1+1)/2 = 1, so Wq = 1.9999999999999998 * 1 / 0.9 = 2.222222222222222 h,
which equals the M/M/1 value to the last digit printed. That reduction is the
structural property the vector checks; a missing factor of 1/2 in the variability
term would double the M/M/1 case and break it.
Interpretation: halving the service coefficient of variation from 1.0 to 0.5 cuts
queue time from 2.2222 h to 1.3889 h, a 37.5 percent reduction, with no change to
utilisation. Variability reduction buys capacity that a rate increase would have
to pay for in capital.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `mm1_waiting_hours` | 2.2222222222 | abs 1e-09 | Two divisions and one subtraction. The subtraction 1 - 2/3 is the lossy step: 2/3 is not binary-exact, so 1 - rho has a relative error near 1.5e-16, which the division amplifies to about 2.4e-15 absolute at 2.22 h. Measured in this session the value came out 2.222222222222222 against the exact 20/9 = 2.2222222222222223, a difference of 4.4e-16. 1e-9 h admits that with six orders of margin. |
| `allen_cunneen_waiting_hours` | 1.3888888889 | abs 1e-09 | The same lossy subtraction plus two multiplications, so the bound is near 3e-15. 1e-9 h admits it with six orders of margin and is eight orders tighter than the 0.8333 h difference from the M/M/1 case, so the vector distinguishes the two formulas. |
| `allen_cunneen_at_unit_cv` | 2.2222222222 | abs 1e-09 | This output exists to check the reduction to M/M/1 and must match mm1_waiting_hours. The two are computed by different functions, so agreement is a real cross-check. Both carry the same 2.4e-15 bound from the shared subtraction, and the tolerance is set at 1e-9 for consistency with the value it is compared against rather than tighter. |

---

## GV-24: Cash cost per tonne of product, with the cascade yield dividing the feed-basis inputs

- File: `data/golden/GV-24-cash-cost.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.econ.unit_economics`
- Dispatch entry: `cash_cost`

**Reference.** The structure (consumption per tonne of feed, divided by the cascade yield to reach a product basis, plus fixed costs per tonne, less byproduct credits) is standard mining and chemical cost accounting. Every price below is a FIXTURE value, not a quoted tariff: no Telangana power tariff or Indian HCl price was verified in this session, so none is asserted.

**Inputs.**

```yaml
site:
  site_id: IN-TG-VKB
  country: IN
  region: Telangana
  currency: USD
  energy_price: 0.08
  operator_rate: 6.0
  reagent_prices:
    HCl: 0.22
  construction_cost_index: 1.15
demands:
- name: electricity
  per_tonne_feed: 180.0
  unit: kWh/tonne
  basis: feed
- name: HCl
  per_tonne_feed: 25.0
  unit: kg/tonne
  basis: feed
cascade_yield: 0.65
product_tonnes_per_year: 8000.0
annual_labour: 2400000.0
annual_maintenance: 800000.0
credits:
- name: iron oxide byproduct
  value: 15.0
capex: 40000000.0
fixed_charge_rate: 0.12
```

**Provenance.**

- `energy_price`: ASSUMED, 0.08 USD/kWh. NOT SOURCED. A Telangana industrial tariff was not retrievable
  in this session, so this is a fixture number and the resulting cost is a scenario,
  not an estimate for this site.
- `operator_rate`: ASSUMED, 6.00 USD/h fully loaded. Not used in the arithmetic below, since labour
  enters as an annual total; required to construct the Site object.
- `reagent_prices`: ASSUMED, 0.22 USD/kg HCl. NOT SOURCED.
- `construction_cost_index`: ASSUMED, 1.15. Not used in cash cost; it enters capex.
- `demands`: ASSUMED, 180 kWh and 25 kg HCl per tonne of FEED. The basis is the point: a per-
  product basis would give a cost 35 percent lower and is the error the yield division
  exists to prevent.
- `cascade_yield`: ASSUMED, 0.65 overall, consistent in structure with GV-21 though not numerically equal
  to it.
- `product_tonnes_per_year`: ASSUMED, 8000 t/yr.
- `annual_labour`: ASSUMED, 2.4 MUSD/yr.
- `annual_maintenance`: ASSUMED, 0.8 MUSD/yr.
- `credits`: ASSUMED, 15 USD/t of product from an iron oxide byproduct. Speculative: no offtake for
  such a byproduct has been identified.
- `fixed_charge_rate`: ASSUMED, 0.12 per year, within the 0.08 to 0.15 band the module documents as typical.
  This is a levelized charge, not a discounted cash flow.

**Arithmetic chain.**

```
Variable inputs, quoted per tonne of FEED and converted to a product basis by
dividing by the cascade yield:
  electricity: 180 kWh/t feed * 0.08 USD/kWh = 14.4 USD/t feed
               14.4 / 0.65 = 22.153846153846153 USD/t product
  HCl:         25 kg/t feed * 0.22 USD/kg = 5.5 USD/t feed
               5.5 / 0.65 = 8.461538461538462 USD/t product
Fixed costs, annual totals over annual product tonnes:
  labour:      2,400,000 / 8,000 = 300.0 USD/t
  maintenance:   800,000 / 8,000 = 100.0 USD/t
  freight:     0.0, explicitly waived on an ex-works basis (buyer collects). The
               module requires freight to be supplied or explicitly waived, so
               that a zero is a decision and not an omission.
Gross cost = 22.153846153846153 + 8.461538461538462 + 300.0 + 100.0 + 0.0
           = 430.61538461538464 USD/t
Cash cost = gross less credits = 430.61538461538464 - 15.0
          = 415.61538461538464 USD/t
Capital recovery, levelized: 40,000,000 * 0.12 / 8,000 = 600.0 USD/t
Full cost = 415.61538461538464 + 600.0 = 1015.6153846153846 USD/t
The capital recovery term is 1.44 times the whole cash cost, which is the
structural fact worth reading off this vector: at 8000 t/yr against a 40 MUSD
capital base, this business is a capital story and not an operating cost story.
The module deliberately excludes capital recovery from the line breakdown so that
the cash cost lines sum to the cash cost and nothing else.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `line_electricity` | 22.1538461538 | abs 1e-09 | One multiplication and one division by 0.65, which is not binary-exact, so 2 ulp near 7e-15 USD/t. 1e-9 USD/t admits that with six orders of margin and is six orders below the cent, so it tests the arithmetic and makes no claim about the accuracy of the ASSUMED input price. |
| `line_HCl` | 8.4615384615 | abs 1e-09 | Same basis as line_electricity. One multiplication and one division by 0.65, which is not binary-exact, so 2 ulp near 7e-15 USD/t. 1e-9 USD/t admits that with six orders of margin and is six orders below the cent, so it tests the arithmetic and makes no claim about the accuracy of the ASSUMED input price. |
| `line_labour` | 300.0 | abs 1e-09 | One division of exact integers, 2400000/8000 = 300 exactly in IEEE 754. 1e-9 guards only against a different route. |
| `line_maintenance` | 100.0 | abs 1e-09 | Same basis as line_labour. One division of exact integers, 2400000/8000 = 300 exactly in IEEE 754. 1e-9 guards only against a different route. |
| `line_freight` | 0.0 | abs 0.0 | An exact zero, placed there by an explicit waiver rather than computed. There is no rounding to admit. The output is checked because an implementation that silently omitted freight rather than recording a waiver would drop the line entirely, and the test then fails on a missing key. |
| `gross_cost` | 430.6153846154 | abs 1e-09 | A sum of five lines, two of which carry 7e-15, so the bound is near 2e-14 USD/t. 1e-9 admits it with five orders of margin. |
| `cash_cost` | 415.6153846154 | abs 1e-09 | Gross less one credit, one subtraction at a magnitude near 430, so the bound is near 6e-14 USD/t. |
| `capital_recovery` | 600.0 | abs 1e-09 | Exact integers throughout: 40e6 * 0.12 / 8000. 0.12 is not binary-exact, so 1 ulp near 7e-14 USD/t enters. 1e-9 admits it. Reported as the DIFFERENCE between full and cash cost, so the output also checks that capital recovery is not double-counted inside the cash cost. |
| `full_cost` | 1015.6153846154 | abs 1e-09 | One addition of two quantities near 400 and 600, so the bound is near 1e-13 USD/t. |
| `breakdown_closes` | 1.0 | abs 0.0 | A boolean from the module's own internal reconciliation, which checks the lines sum to the cash cost within its own 1e-9 tolerance. Encoded as 1.0, so there is nothing to round. |

---

## GV-25: Factored capital estimate with escalation, location factor and an AACE Class 5 band

- File: `data/golden/GV-25-capex-factored.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.econ.capex`
- Dispatch entry: `capex`

**Reference.** AACE International Recommended Practice 18R-97, Cost Estimate Classification System, for the Class 5 accuracy range of minus 50 to plus 100 percent, as recorded in the module's AACE_ACCURACY table. The six-tenths rule for capacity scaling is standard and is used here with an exponent of 0.65, stated as an input. The two index values below are FIXTURE numbers: no Chemical Engineering Plant Cost Index value was verified in this session, so no real index level is asserted.

**Inputs.**

```yaml
site:
  site_id: IN-TG-VKB
  country: IN
  region: Telangana
  currency: USD
  energy_price: 0.08
  operator_rate: 6.0
  reagent_prices:
    HCl: 0.22
  construction_cost_index: 1.15
equipment:
- name: kiln
  purchased_cost: 3000000.0
  installation_factor: 2.2
- name: mill
  purchased_cost: 1500000.0
  installation_factor: 1.8
indirect_factor: 0.3
contingency_fraction: 0.25
base_index: 708.0
target_index: 826.0
scaling:
  known_cost: 2000000.0
  known_size: 5000.0
  target_size: 18000.0
  size_unit: tonne/year
  exponent: 0.65
escalation:
  cost: 1000000.0
```

**Provenance.**

- `purchased_cost`: ASSUMED, 3.0 MUSD kiln and 1.5 MUSD mill. NOT SOURCED: no vendor quotes were obtained.
  This is why the estimate is Class 5.
- `installation_factor`: ASSUMED, 2.2 for the kiln and 1.8 for the mill, inside the 1.0 to 6.0 range the module
  allows and near the middle of the 1.4 (bare tank) to 3.5 (instrumented) band it
  documents as typical.
- `indirect_factor`: ASSUMED, 0.30, mid-range of the 0.20 to 0.40 the module documents.
- `contingency_fraction`: ASSUMED, 0.25.
- `base_index`: ASSUMED, 708.0. NOT SOURCED as a real CEPCI value. The ratio to the target index is
  what the arithmetic tests.
- `target_index`: ASSUMED, 826.0, giving a ratio of exactly 7/6.
- `construction_cost_index`: ASSUMED, 1.15 as a location factor relative to the index basis, meaning construction
  at this site costs 15 percent more than at the reference location.
- `exponent`: ASSUMED, 0.65. The choice matters: at a 3.6-fold scale-up, an exponent of 0.6 against
  0.9 changes the scaled cost by more than 50 percent, which the module documents. No
  fitted exponent for this equipment class was sourced.

**Arithmetic chain.**

```
Installed cost on a raw basis, purchased times installation factor:
  kiln: 3,000,000 * 2.2 = 6,600,000
  mill: 1,500,000 * 1.8 = 2,700,000
  raw installed = 9,300,000; raw purchased = 4,500,000
Index ratio = 826/708 = 1.1666666666666667 (exactly 7/6)
Location factor = 1.15
Installed, escalated and located:
  9,300,000 * 1.1666666666666667 * 1.15 = 10,850,000 * 1.15 = 12,477,500
  In double precision this evaluates to 12477499.999999998.
Purchased, escalated and located:
  4,500,000 * 1.1666666666666667 * 1.15 = 5,250,000 * 1.15 = 6,037,500
  (6037499.999999999 in double precision)
Indirects at 0.30 of installed: 12,477,500 * 0.30 = 3,743,250
Subtotal: 12,477,500 + 3,743,250 = 16,220,750
Contingency at 0.25 of the subtotal: 16,220,750 * 0.25 = 4,055,187.5
Total project cost: 16,220,750 + 4,055,187.5 = 20,275,937.5
AACE Class 5 band, minus 50 to plus 100 percent:
  low  = 20,275,937.5 * 0.50 = 10,137,968.75
  high = 20,275,937.5 * 2.00 = 40,551,875.0
The band spans a factor of four. Quoting the Class 4 band (minus 30 to plus 50)
on an estimate built from no quotes and no plot plan would understate the upside
by a factor of two, which this module's docstring records as a defect it fixed.
Six-tenths scaling, exponent 0.65:
  ratio = 18,000/5,000 = 3.6
  3.6^0.65 = 2.299305221174797
  2,000,000 * 2.299305221174797 = 4,598,610.442349594
Escalation alone: 1,000,000 * 7/6 = 1,166,666.6666666667
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `index_ratio` | 1.1666666666666667 | abs 1e-12 | One division giving 7/6, not binary-exact, so 1 ulp near 2e-16. 1e-12 admits it with four orders of margin. |
| `location_factor` | 1.15 | abs 1e-12 | A constant passed through from the Site. 1.15 is not binary-exact, so 1 ulp near 2e-16, but no operation is applied. 1e-12 admits any conversion rounding. |
| `total_purchased` | 6037500.0 | abs 0.01 | A RELATIVE-scale absolute tolerance of one cent on a 6.04 MUSD figure, which is 1.7e-9 relative. The arithmetic is two multiplications on a 4.5e6 magnitude, so the floating point bound is a few ulp of 6e6, near 2e-9 USD, and the measured value 6037499.999999999 differs from the exact 6037500 by 9.3e-10 USD. One cent admits that with seven orders of margin. A cent rather than 1e-9 USD is chosen because currency below the cent is not meaningful and because the value is the product of three inexact decimals, so pinning it to the nanodollar would test the multiplication order rather than the estimate. |
| `total_installed` | 12477500.0 | abs 0.01 | As total_purchased. The measured value 12477499.999999998 differs from the exact 12477500 by 1.9e-9 USD, which one cent admits with seven orders of margin. |
| `indirect_cost` | 3743250.0 | abs 0.01 | One further multiplication by 0.30, so the bound is near 5e-10 USD. One cent admits it. |
| `contingency` | 4055187.5 | abs 0.01 | One addition and one multiplication by 0.25, which is binary-exact, so the bound is inherited from the subtotal, near 2e-9 USD. One cent admits it. |
| `total_project_cost` | 20275937.5 | abs 0.01 | The sum of three components each within 2e-9 USD, so the bound is near 6e-9 USD. One cent admits that with six orders of margin and is nine orders below the plus 100 percent Class 5 uncertainty, which is the real uncertainty on this number and is stated in the provenance rather than in the tolerance. |
| `accuracy_low` | 10137968.75 | abs 0.01 | The total times 0.50, which is binary-exact, so the total's bound halves. One cent admits it. |
| `accuracy_high` | 40551875.0 | abs 0.01 | The total times 2.00, exact, so the bound doubles to 1.2e-8 USD. One cent admits it. |
| `scaled_cost` | 4598610.442349594 | abs 0.01 | One fractional power and one multiplication. The power 3.6^0.65 carries about 1 ulp relative, 2e-16, which at 4.6 MUSD is 1e-9 USD. One cent admits that with seven orders of margin. An exponent of 0.6 rather than 0.65 gives 4,313,318 USD, a 285,292 USD difference, seven orders above the tolerance. |
| `escalated_cost` | 1166666.6666666667 | abs 0.01 | One multiplication by 7/6, so the bound is near 2e-10 USD. One cent admits it. |
| `reconciles` | 1.0 | abs 0.0 | A boolean from the module's own internal check that the components sum to the total within its 1e-6 tolerance. Encoded as 1.0, nothing to round. |

---

## GV-26: NPV, IRR, payback, levelized cost and breakeven price on a five-period project

- File: `data/golden/GV-26-valuation.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.econ.valuation`
- Dispatch entry: `valuation`

**Reference.** Discounted cash flow arithmetic is elementary and needs no source. The identity checked here, that the breakeven price equals the levelized cost when the tax rate is zero, follows from both being the price at which discounted revenue equals discounted cost, and is derived explicitly below.

**Inputs.**

```yaml
capex_schedule:
- 50.0
- 50.0
construction_periods: 2
ramp_fractions:
- 0.5
- 1.0
nameplate_tonnes: 100.0
price: 10.0
cash_cost_per_tonne: 6.0
life_periods: 3
discount_rate: 0.1
```

**Provenance.**

- `capex_schedule`: ASSUMED, 50 units in each of two construction periods. Round numbers chosen so every
  discount factor is a power of 1.1 that can be written out.
- `construction_periods`: ASSUMED, 2 periods with no output. The module's docstring records that setting this to
  zero understates the capital cost of a project of this shape by 38 percent, which is
  why the vector uses a non-zero value.
- `ramp_fractions`: ASSUMED, 50 percent of nameplate in the first operating period, full rate thereafter.
- `nameplate_tonnes`: ASSUMED, 100 t per period.
- `price`: ASSUMED, 10 currency units per tonne.
- `cash_cost_per_tonne`: ASSUMED, 6 units per tonne, giving a 40 percent margin.
- `life_periods`: ASSUMED, 3 operating periods. Deliberately short so the cash flow stream can be
  written out in full and checked term by term.
- `discount_rate`: ASSUMED, 0.10 per period. No tax, no working capital, no salvage and no fixed cost,
  all at their module defaults of zero, so the vector isolates the discounting
  arithmetic.

**Arithmetic chain.**

```
Output profile, zero through construction then ramped:
  [0, 0, 50, 100, 100] tonnes
Revenue at 10/t:   [0, 0, 500, 1000, 1000]
Variable at 6/t:   [0, 0, 300, 600, 600]
EBITDA:            [0, 0, 200, 400, 400]
Capex:             [50, 50, 0, 0, 0]
Net cash flow = EBITDA - capex (tax, working capital and salvage are all zero):
                   [-50, -50, 200, 400, 400]
Discount factors at 10 percent: 1, 1.1, 1.21, 1.331, 1.4641
Present values term by term:
  t=0: -50/1       = -50.0
  t=1: -50/1.1     = -45.45454545454545
  t=2: 200/1.21    = 165.2892561983471
  t=3: 400/1.331   = 300.525920360631
  t=4: 400/1.4641  = 273.2053821460282
NPV = -50 - 45.45454545454545 + 165.2892561983471 + 300.525920360631
      + 273.2053821460282 = 643.5660132504609
IRR is the rate at which that sum is zero. Solved by bisection on the polynomial
in this session, independently of the module: 1.4675038570565175 per period. At
that rate the discounted sum is -1.78e-15, which is the check.
Payback, undiscounted: cumulative net cash flow is
  [-50, -100, 100, 500, 900], first non-negative at t=2, interpolating within the
  crossing period: 1 + 100/200 = 1.5 periods.
PV of output: 50/1.21 + 100/1.331 + 100/1.4641
            = 41.32231404958677 + 75.13148009015775 + 68.30134553650706
            = 184.7551396762516 tonnes
PV of capex: 50 + 50/1.1 = 95.45454545454545
PV of all costs (capex plus variable plus fixed):
  cost stream = [50, 50, 300, 600, 600]
  PV = 50 + 45.45454545454545 + 247.93388429752062 + 450.78888054094654
     + 409.8080732190423 = 1203.9853835120548
Levelized cost = PV(costs)/PV(output) = 1203.9853835120548/184.7551396762516
               = 6.516654343807763 per tonne
Breakeven price. At zero tax the NPV is linear in price, so the breakeven price is
the one at which PV(revenue) = PV(costs), that is
  price * PV(output) = PV(costs), price = PV(costs)/PV(output),
the SAME expression as the levelized cost. So breakeven and LCOP must agree
exactly at zero tax, and the vector asserts that identity. With tax they would
differ, because tax is not linear in price once losses are carried forward.
Cross-check: the breakeven price is 6.5166543438 against a cash cost of 6.00, so
the capital burden is 0.5166543438 per tonne, which is PV(capex)/PV(output) =
95.45454545454545/184.7551396762516 = 0.5166543438077635, and
6.0 + 0.5166543438077635 = 6.5166543438077635, agreeing with the levelized cost
to the last digit. Two independent routes to the same number.
Peak funding requirement, the most negative cumulative undiscounted position:
  min(0, -50, -100, 100, 500, 900) = -100.0
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `output_0` | 0.0 | abs 0.0 | An exact zero placed by the construction-period logic. Nothing to round. Checked because an off-by-one in the ramp offset would put output in a construction period. |
| `output_1` | 0.0 | abs 0.0 | An exact zero placed by the construction-period logic, the second of two construction periods. Nothing to round. Checked because an off-by-one in the ramp offset would put output into a construction period. |
| `output_2` | 50.0 | abs 1e-12 | 0.5 * 100, both exactly representable, so the product is exact. 1e-12 guards only against a different route. |
| `output_3` | 100.0 | abs 1e-12 | The product 1.0 times 100, both exactly representable in binary, so the result is exact. 1e-12 guards only against a different route to the same number. |
| `output_4` | 100.0 | abs 1e-12 | The ramp holds its last value, so 1.0 * 100 again. Checked so that a ramp shorter than the life is confirmed to extend rather than to fall back to zero. |
| `net_cash_flow_0` | -50.0 | abs 1e-12 | A single capex entry negated, with zero EBITDA, tax and working capital, so the value is exactly -50. 1e-12 admits any summation-order difference. |
| `net_cash_flow_1` | -50.0 | abs 1e-12 | A single capex entry negated, with zero EBITDA, tax and working capital, so the value is exactly -50. 1e-12 admits any summation-order difference. |
| `net_cash_flow_2` | 200.0 | abs 1e-09 | Revenue less variable cost, both exact products of round numbers, so the difference is exact. 1e-9 admits any pint conversion rounding. |
| `net_cash_flow_3` | 400.0 | abs 1e-09 | Revenue less variable cost, both exact products of round numbers, so the difference is exact. 1e-9 admits any pint unit-conversion rounding on the way out. |
| `net_cash_flow_4` | 400.0 | abs 1e-09 | Revenue less variable cost, both exact products of round numbers, so the difference is exact. 1e-9 admits any pint unit-conversion rounding on the way out. |
| `npv` | 643.5660132505 | abs 1e-09 | Five divisions by powers of 1.1 (none binary-exact) and four additions, at magnitudes near 300, so the bound is a few ulp of 643, near 5e-13. 1e-9 admits that with three orders of margin. This is the vector's least forgiving arithmetic output and the tolerance reflects that the discount factors compound their rounding. |
| `irr` | 1.4675038571 | abs 1e-08 | The module finds the IRR by bisection with a stated tolerance of 1e-10 on the rate, so the error floor is the solver's, not floating point's. The independently derived value came from a separate 400-iteration bisection in this session, which converges to about 1e-16 on the rate but locates the same root. 1e-8 admits the module's 1e-10 stopping criterion with two orders of margin and allows for the two bisections bracketing from different sides. |
| `payback_undiscounted` | 1.5 | abs 1e-09 | A linear interpolation within the crossing period, one division of exact integers 100/200, so exact. 1e-9 admits any rounding in the cumulative sum. The interpolation is the substance: reporting the integer period 2 rather than 1.5 would fail by 0.5. |
| `levelized_cost` | 6.5166543438 | abs 1e-09 | A ratio of two discounted sums, each carrying the 5e-13 bound of the NPV chain, so the quotient carries about 1e-14 relative or 7e-14 absolute at 6.52. 1e-9 admits that with four orders of margin. |
| `breakeven_price` | 6.5166543438 | abs 1e-08 | Found by bisection on price with the module's stated tolerance of 1e-8, so the solver's stopping criterion sets the floor, not floating point. The tolerance equals that criterion rather than something tighter, because a tighter value would be asserting a precision the solver does not deliver. |
| `npv_at_breakeven` | 0.0 | abs 1e-06 | The NPV recomputed at the breakeven price must be zero. It is not exactly zero: the breakeven price is known only to the solver's 1e-8, and the NPV derivative with respect to price is PV(output) = 184.76, so a 1e-8 price error gives an NPV error near 1.8e-6. The tolerance is that propagated figure rounded up, which is why it is 1e-6 and not 1e-8. This is the clearest case in the suite of a tolerance derived by propagation rather than chosen. |
| `peak_funding_requirement` | -100.0 | abs 1e-09 | A minimum over a cumulative sum of exact values at a zero discount rate, so exact. 1e-9 admits any rounding. The output is checked because it answers a different question from the NPV (how much cash must be raised, not whether the project is worth it) and a project can have a positive NPV and an unfundable peak. |

---

## GV-27: Sobol indices of a linear additive function, checked against the exact variance decomposition

- File: `data/golden/GV-27-sobol-additive.yaml`
- Kind: **golden**, derivation `independent`
- Module under test: `ae.econ.uncertainty`
- Dispatch entry: `sobol_additive`

**Reference.** Saltelli, A., Annoni, P., Azzini, I. et al. 2010, Variance based sensitivity analysis of model output: Design and estimator for the total sensitivity index, Computer Physics Communications 181:259-270. The exact indices below come from the variance of a uniform variable, Var(aU(0,1)) = a^2/12, and not from any estimator.

**Inputs.**

```yaml
coefficients:
- 1.0
- 2.0
- 3.0
n_base: 4096
seed: 0
```

**Provenance.**

- `coefficients`: ASSUMED, y = x1 + 2 x2 + 3 x3 with each xi uniform on (0,1). Chosen because a purely
  additive linear model has an analytically known decomposition, so the vector checks
  the estimator against truth rather than against itself. The coefficients are
  distinct so the three indices are distinct and a permuted output ordering would be
  detected.
- `n_base`: ASSUMED, 4096 base samples, giving 4096*(3+2) = 20480 model evaluations. A power of
  two is required by the module because the Sobol sequence loses its balance
  properties otherwise.
- `seed`: ASSUMED, 0. The seed affects the estimate but not the truth, which is the property
  that makes this a golden and not a regression vector: the expected values are
  analytic and would be the same under any seed, with only the tolerance depending on
  the sample size.

**Arithmetic chain.**

```
Each xi is uniform on (0,1) with variance 1/12. For an additive model the variance
decomposes exactly:
  V1 = 1^2/12 = 1/12  = 0.08333333333333333
  V2 = 2^2/12 = 4/12  = 0.3333333333333333
  V3 = 3^2/12 = 9/12  = 0.75
  total variance = (1 + 4 + 9)/12 = 14/12 = 1.1666666666666667
First-order indices are Vi over the total:
  S1 = 1/14 = 0.07142857142857142
  S2 = 4/14 = 0.2857142857142857
  S3 = 9/14 = 0.6428571428571429
  sum = 1 exactly, because the model has no interactions.
Total-order indices equal the first-order indices for an additive model, since
there is no interaction variance to add:
  ST1 = S1, ST2 = S2, ST3 = S3
Interaction share, defined as (ST - S)/ST, is therefore exactly zero for each
input, and the additive fraction (the sum of the first-order indices) is exactly 1.
These are the structural facts a Sobol implementation must reproduce and they are
the strongest available test of one, because any estimator bias shows up as a
departure from a known constant rather than as a difference from another estimate.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `n_evaluations` | 20480.0 | abs 0.0 | An integer, n_base*(k+2) = 4096*5. A different count means a different estimator design, not a rounding difference. |
| `output_variance` | 1.1666666667 | abs 0.002 | A MONTE CARLO tolerance, not a floating point one. The sample variance of 20480 quasi-random draws of a variable whose own variance is 1.1667 estimates the true variance with a standard error of roughly sigma^2 sqrt(2/N) = 1.1667 * 0.0099 = 0.0115 for independent draws, and the Sobol sequence does better than that. Measured across seeds 0 to 4 at this n_base the returned variances were 1.166733 and nearby, within 6e-5 of the truth. 0.002 is a factor of thirty above the measured spread and a factor of six below the independent-sampling standard error, so it is loose enough not to fail on a seed change and tight enough to catch a variance computed on the wrong sample. |
| `first_x1` | 0.0714285714 | abs 0.005 | MEASURED Monte Carlo scatter. Across seeds 0, 1, 2, 3 and 4 at n_base = 4096 the maximum absolute deviation of any first or total index from its analytic value was 0.000313 (measured in this session; at n_base = 1024 it was 0.000840). 0.005 is sixteen times the measured maximum, which covers seeds not tested while remaining far below the 0.21 separation between the three true indices. A tolerance derived from the observed seed-to-seed spread is the only defensible choice here: a floating point tolerance would fail on every seed and an eyeballed 0.05 would admit a genuinely wrong index. |
| `first_x2` | 0.2857142857 | abs 0.005 | Same measured basis as first_x1: per-seed worst index deviation at n_base 4096 was 1.8216682789187755e-05 (seed 0), 3.070839859953267e-06 (seed 1), 4.6773696027457845e-12 (seed 2), 7.663454326678476e-08 (seed 3) and 0.000313270921173725 (seed 4). 0.005 is 15.96 times the worst of those. |
| `first_x3` | 0.6428571429 | abs 0.005 | Same measured basis as first_x1: the worst per-seed index deviation at n_base 4096 was 0.000313270921173725 at seed 4, and 0.005 is 15.96 times that, far below the 0.21 separation between the three true indices. |
| `total_x1` | 0.0714285714 | abs 0.005 | Same measured basis as first_x1: the per-seed worst deviation of any index from its analytic value at n_base 4096 ran from 4.6773696027457845e-12 (seed 2) to 0.000313270921173725 (seed 4), and 0.005 is 15.96 times the worst. The total-order estimator is a different formula from the first- order one, so it is checked separately against the same truth; for an additive model the two must agree. |
| `total_x2` | 0.2857142857 | abs 0.005 | Same measured basis as total_x1, the same per-seed worst deviations with a maximum of 0.000313270921173725 at seed 4 against a 0.005 tolerance. Checked separately from first_x2 because the total-order estimator is a different formula. |
| `total_x3` | 0.6428571429 | abs 0.005 | Same measured basis as total_x1, the same per-seed worst deviations with a maximum of 0.000313270921173725 at seed 4 against a 0.005 tolerance. Checked separately from first_x3 because the total-order estimator is a different formula. |
| `interaction_x1` | 0.0 | abs 0.01 | The interaction share is (ST - S)/ST, a DIFFERENCE of two noisy estimates divided by one of them, so its error is larger than either index's. With each index within 0.000313 of truth, the numerator can be off by 0.000626, and dividing by ST1 = 0.0714 amplifies that to 0.0088. 0.01 admits the propagated figure. Note this is the one output whose tolerance had to be LOOSENED relative to the indices themselves, and the amplification factor 1/ST is the reason: the same absolute index error is a much larger relative error on the smallest index. |
| `interaction_x2` | 0.0 | abs 0.01 | Same basis as interaction_x1: (ST - S)/ST is a difference of two noisy estimates divided by one of them. The amplification is milder here, 1/0.2857142857142857 = 3.5 rather than 14, so the propagated bound is 0.002192896448216075 and 0.01 has more margin than on x1. |
| `interaction_x3` | 0.0 | abs 0.01 | Same basis as interaction_x1: (ST - S)/ST is a difference of two noisy estimates divided by one of them. The amplification here is 1/0.6428571428571429 = 1.556, the mildest of the three, giving a propagated bound of 0.0009746206436515889. |
| `additive_fraction` | 1.0 | abs 0.005 | The sum of the three first-order indices. Errors in a Sobol estimate are not independent across indices (they share the same base sample), so the sum's error is not sqrt(3) times one index's; the measured deviations summed to less than 0.0005 in this session. 0.005 admits that with an order of margin. |
| `diagnostics_pass` | 1.0 | abs 0.0 | A boolean: the module's diagnostics flag fires when the first-order sum exceeds 1.05, the total-order sum falls below 0.95, an index is below -0.05, or a first-order index exceeds its total by more than 0.05. For a correct additive estimate none of these fire. Encoded as 1.0 with nothing to round. |

---

## RV-01: Seeded Monte Carlo percentiles of a uniform variable, with the analytic quantile alongside

- File: `data/golden/RV-01-monte-carlo-percentiles.yaml`
- Kind: **regression**, derivation `from_code`
- Module under test: `ae.econ.uncertainty`
- Dispatch entry: `monte_carlo_percentiles`

**Reference.** David, H.A. and Nagaraja, H.N. 2003, Order Statistics, 3rd edition, Wiley, for the asymptotic standard error of a sample quantile, se = sqrt(p(1-p)/n)/f(x_p).

**Why this is a regression vector.** The expected values in this file are the percentiles of a PARTICULAR seeded sample and can only come from running the code. They depend on the seed, on the number of draws, on numpy's default_rng bit stream and on numpy's percentile interpolation convention. None of those is derivable with a calculator, so this is a REGRESSION vector and is counted separately from the golden vectors. What the file does check independently is that each seeded percentile lies within the order-statistic standard error of its ANALYTIC value, which is derived in the arithmetic block; the seeded value pins the bit stream, and the analytic bound tests the estimator.

**Inputs.**

```yaml
name: x
low: 100.0
high: 200.0
kind: uniform
n_draws: 50000
seed: 0
percentiles:
- 10
- 50
- 90
n_boot: 400
```

**Provenance.**

- `low`: ASSUMED, 100. The uniform distribution was chosen because its quantile function and
  its density are both exact, so the analytic comparison below has no approximation of
  its own.
- `high`: ASSUMED, 200.
- `n_draws`: ASSUMED, 50000, above the module's documented floor of 10000 for percentile reporting.
- `seed`: ASSUMED, 0. This is the number that makes the expected values reproducible and is also
  the reason they are not golden.
- `n_boot`: ASSUMED, 400 bootstrap resamples for the percentile standard error, the module
  default.

**Arithmetic chain.**

```
Analytic quantiles of U(100, 200):
  P10 = 100 + 0.10*100 = 110.0
  P50 = 100 + 0.50*100 = 150.0
  P90 = 100 + 0.90*100 = 190.0
Order-statistic standard error, se = sqrt(p(1-p)/n) / f(x_p) with density
f = 1/(200-100) = 0.01:
  at p = 0.10: sqrt(0.10*0.90/50000) = sqrt(1.8e-06) = 0.0013416407864998738
               se = 0.0013416407864998738/0.01 = 0.13416407864998739
  at p = 0.50: sqrt(0.25/50000) = 0.0022360679774997895
               se = 0.22360679774997896
  at p = 0.90: same as p = 0.10 by symmetry, se = 0.13416407864998739
So a correct estimator should land within roughly 0.13 of 110 and 0.22 of 150.
MEASURED seeded values, which are the regression content of this file:
  seed 0: P10 = 109.78876517893794, P50 = 150.05657561025876,
          P90 = 189.79034145691648, mean = 150.06755258237402
Deviations from analytic: -0.21123482106206382, +0.056575610258761344 and
-0.2096585430835205, which are 1.574, 0.253 and 1.563 standard errors
respectively. That the P10 and P90 deviations are both about 1.6 se in
the same direction is not a defect; it is what one draw of a correlated sample
looks like.
Across seeds 0 to 4 the measured P10 ranged 109.63914543983736 to
110.0019941642077, a spread of 0.363, consistent with an se of 0.134.
The module's own bootstrap standard error at seed 0 returned 0.1325916214003937
for P10 and 0.23663988865573488 for P50, against the analytic 0.1342 and 0.2236.
The ratios are 0.988 and 1.058, so the bootstrap agrees with the closed form to
within 6 percent, which is the real check in this file: it tests the uncertainty reporting, not just the point estimate.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `n_draws` | 50000.0 | abs 0.0 | An integer count of completed draws. Any other value means draws were lost. |
| `n_failed` | 0.0 | abs 0.0 | An integer count of model failures, which must be zero for an identity model. A non-zero value would mean the error-recording path fired on a function that cannot fail. |
| `mean` | 150.0675525824 | abs 1e-09 | A REGRESSION tolerance at the floating point level, not a statistical one. The seed fixes the sample exactly, so the mean of that sample is a deterministic function of the bit stream and reproduces to full precision on the same numpy version. 1e-9 admits summation-order differences across platforms. The statistical question (is 150.0676 close to 150?) is answered by the analytic_reference block: the standard error of the mean here is (100/sqrt(12))/sqrt(50000) = 0.1290994448735806, and the measured mean sits 0.0676 from 150, which is 0.52 of that. This tolerance says nothing about that question. |
| `p10` | 109.7887651789 | abs 1e-09 | As mean: a fixed seed makes this exact, so the tolerance is floating point and the value is a bit-stream pin. It is emphatically NOT a statement that the tenth percentile of U(100,200) is known to 1e-9; that quantity is 110 and this sample estimates it to 0.21. |
| `p50` | 150.0565756103 | abs 1e-09 | Same basis as p10. Same basis as mean. A REGRESSION tolerance at the floating point level, not a statistical one. The seed fixes the sample exactly, so the mean of that sample is a deterministic function of the bit stream and reproduces to full precision on the same numpy version. 1e-9 admits summation-order differences across platforms. The statistical question (is 150.0676 close to 150?) is answered by the analytic_reference block: the standard error of the mean here is (100/sqrt(12))/sqrt(50000) = 0.1290994448735806, and the measured mean sits 0.0676 from 150, which is 0.52 of that. This tolerance says nothing about that question. Additionally for this output: : a fixed seed makes this exact, so the tolerance is floating point and the value is a bit-stream pin. It is emphatically NOT a statement that the tenth percentile of U(100,200) is known to 1e-9; that quantity is 110 and this sample estimates it to 0.21. |
| `p90` | 189.7903414569 | abs 1e-09 | Same basis as p10. Same basis as mean. A REGRESSION tolerance at the floating point level, not a statistical one. The seed fixes the sample exactly, so the mean of that sample is a deterministic function of the bit stream and reproduces to full precision on the same numpy version. 1e-9 admits summation-order differences across platforms. The statistical question (is 150.0676 close to 150?) is answered by the analytic_reference block: the standard error of the mean here is (100/sqrt(12))/sqrt(50000) = 0.1290994448735806, and the measured mean sits 0.0676 from 150, which is 0.52 of that. This tolerance says nothing about that question. Additionally for this output: : a fixed seed makes this exact, so the tolerance is floating point and the value is a bit-stream pin. It is emphatically NOT a statement that the tenth percentile of U(100,200) is known to 1e-9; that quantity is 110 and this sample estimates it to 0.21. |
| `bootstrap_se_p10` | 0.1325916214 | abs 1e-09 | Also seeded (the bootstrap resampling uses the same seed argument), so it reproduces exactly. The interesting comparison, against the analytic 0.1342, is checked by the test with a 10 percent relative band rather than by this tolerance, because the bootstrap is a consistent but not exact estimator of the asymptotic standard error at n_boot = 400. |
| `bootstrap_se_p50` | 0.2366398887 | abs 1e-09 | Same basis as bootstrap_se_p10. Also seeded (the bootstrap resampling uses the same seed argument), so it reproduces exactly. The interesting comparison, against the analytic 0.1342, is checked by the test with a 10 percent relative band rather than by this tolerance, because the bootstrap is a consistent but not exact estimator of the asymptotic standard error at n_boot = 400. |

**Closed-form reference, carried alongside the seeded values.**

| quantity | exact value |
| --- | --- |
| `p10` | 110.0 |
| `p50` | 150.0 |
| `p90` | 190.0 |
| `order_statistic_se_p10` | 0.1341640786 |
| `order_statistic_se_p50` | 0.2236067977 |
| `order_statistic_se_p90` | 0.1341640786 |

---

## RV-02: Leave-one-deposit-out folds on a six-sample three-group set, with the mean baseline derived by hand

- File: `data/golden/RV-02-surrogate-lodo.yaml`
- Kind: **regression**, derivation `mixed`
- Module under test: `ae.ml.surrogate`
- Dispatch entry: `surrogate_lodo`

**Reference.** Roberts, D.R., Bahn, V., Ciuti, S. et al. 2017, Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure, Ecography 40:913-929, for the argument that grouped validation is the honest estimate when samples cluster. The mean-baseline RMSE arithmetic below needs no source.

**Why this is a regression vector.** The fold structure, the sample counts and the MEAN-baseline RMSE of each fold are derived independently below with a calculator. The fitted model's RMSE and the ridge-baseline RMSE are not: they depend on the estimator's internals (here a gradient boosting machine with a fixed seed, and a RidgeCV alpha grid), so those outputs are REGRESSION pins. The file is therefore counted as a regression vector even though most of its expected values are independently derived, because the strictest classification is the honest one: any file containing a value that could only come from running the code is a regression vector.

**Inputs.**

```yaml
X:
- - 1.0
  - 0.5
- - 2.0
  - 0.5
- - 3.0
  - 1.5
- - 4.0
  - 1.5
- - 5.0
  - 2.5
- - 6.0
  - 2.5
y:
- 10.0
- 12.0
- 20.0
- 22.0
- 30.0
- 32.0
groups:
- A
- A
- B
- B
- C
- C
feature_names:
- feature_1
- feature_2
target_name: target
kind: gbm
seed: 0
```

**Provenance.**

- `X`: ASSUMED. Feature 1 is a sample index 1 to 6 and feature 2 is the group centroid (0.5,
  1.5, 2.5). The target is an EXACT linear function of both, y = 2*feature_1 +
  6*feature_2 + 5, verified in this session against all six rows: 2+3+5 = 10, 4+3+5 =
  12, 6+9+5 = 20, 8+9+5 = 22, 10+15+5 = 30, 12+15+5 = 32. This is a synthetic fixture
  and proves nothing about any real deposit.
- `y`: ASSUMED, the exact linear image described above, in ascending order.
- `groups`: ASSUMED, three deposits of two samples each. Three is the minimum that leaves a non-
  trivial training set on every fold; the module documents that five deposits is where
  a grouped estimate becomes meaningful, so this fixture is below that threshold by
  design and the vector makes no claim about model skill.
- `kind`: ASSUMED, gbm. A tree ensemble was chosen deliberately: it extrapolates as a constant,
  so on a leave-one-group-out split where the held-out group lies outside the training
  range it cannot do better than predicting the nearest training mean. That failure is
  the pedagogical content of the fixture.
- `seed`: ASSUMED, 0.

**Arithmetic chain.**

```
Fold structure. Leave-one-deposit-out on three groups of two gives three folds,
each with 4 training and 2 test samples:
  fold A: train indices [2,3,4,5], test [0,1]
  fold B: train indices [0,1,4,5], test [2,3]
  fold C: train indices [0,1,2,3], test [4,5]
Global mean of y = (10+12+20+22+30+32)/6 = 126/6 = 21.0
Target interquartile range, linear interpolation on the sorted y:
  sorted y = [10,12,20,22,30,32], n = 6
  q25 index = (6-1)*0.25 = 1.25, so q25 = 12 + 0.25*(20-12) = 14.0
  q75 index = (6-1)*0.75 = 3.75, so q75 = 22 + 0.75*(30-22) = 28.0
  IQR = 28.0 - 14.0 = 14.0
Mean-baseline RMSE per fold, predicting the TRAINING mean for every test sample:
  fold A: train y = [20,22,30,32], mean = 104/4 = 26.0
          test y = [10,12], errors -16 and -14
          MSE = (256 + 196)/2 = 226.0, RMSE = sqrt(226) = 15.033296378372908
  fold B: train y = [10,12,30,32], mean = 84/4 = 21.0
          test y = [20,22], errors -1 and +1
          MSE = (1 + 1)/2 = 1.0, RMSE = 1.0
  fold C: train y = [10,12,20,22], mean = 64/4 = 16.0
          test y = [30,32], errors +14 and +16
          MSE = (196 + 256)/2 = 226.0, RMSE = sqrt(226) = 15.033296378372908
The A and C folds are symmetric, which is why their baselines are identical, and
the B fold is easy because the held-out group sits between the two training
groups. That asymmetry is the whole reason a per-fold spread must be reported and
not just an average: averaging 15.033296378372908, 1.0 and 15.033296378372908
gives 10.355530918915273, which describes none of the three folds.
MEASURED model RMSE, which cannot be derived by hand:
  gbm fold A 15.033296378372908, fold B 1.0, fold C 15.033296378372908
The gbm exactly equals the mean baseline on every fold. That is not a coincidence
and it is not a bug: with 4 training samples and min_samples_leaf defaults, the
ensemble cannot split usefully and predicts the training mean, so it inherits the
baseline exactly. A model with no skill over the mean is what this fixture is
designed to exhibit.
MEASURED ridge-baseline RMSE, also not hand-derivable (RidgeCV selects alpha from
a 25-point logspace grid by internal cross-validation):
  fold A 0.0022628130162319375, fold B 0.0017341949999050144,
  fold C 0.0022628130162319375
The linear baseline recovers the exactly-linear target almost perfectly while the
tree ensemble does not. This is the case the module's "a ridge baseline can win and
that is reported" behaviour exists for.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `n_folds` | 3.0 | abs 0.0 | An integer. Three groups give three folds; any other count means the grouping was ignored. |
| `global_mean` | 21.0 | abs 1e-12 | An exact integer quotient 126/6. 1e-12 admits summation order. |
| `target_iqr` | 14.0 | abs 1e-12 | Two linear interpolations on exact integers giving exactly 14.0. 1e-12 admits the interpolation-convention arithmetic. The convention matters: a nearest-rank quartile would give 20 - 12 = 8 rather than 14, detected by 6. |
| `n_train_A` | 4.0 | abs 0.0 | An integer sample count. Independently derived from the group structure. |
| `n_test_A` | 2.0 | abs 0.0 | An integer sample count, independently derived from the group structure: three groups of two give two test samples per fold. Any other value means the grouping was not respected. |
| `n_train_B` | 4.0 | abs 0.0 | An integer sample count, independently derived from the group structure: six samples less the two held out give four in training. |
| `n_test_B` | 2.0 | abs 0.0 | An integer sample count, independently derived from the group structure: two samples per held-out deposit. |
| `n_train_C` | 4.0 | abs 0.0 | An integer sample count, independently derived from the group structure: six samples less the two held out give four in training. |
| `n_test_C` | 2.0 | abs 0.0 | An integer sample count, independently derived from the group structure: two samples per held-out deposit. |
| `baseline_mean_rmse_A` | 15.0332963784 | abs 1e-09 | INDEPENDENTLY DERIVED as sqrt(226). One square root of an exact integer, so 1 ulp near 2e-15. 1e-9 admits it with six orders of margin. This is a golden value inside a regression file and is labelled as such. |
| `baseline_mean_rmse_B` | 1.0 | abs 1e-09 | Independently derived as sqrt(1) = 1.0, exact. 1e-9 admits any rounding in the mean. |
| `baseline_mean_rmse_C` | 15.0332963784 | abs 1e-09 | INDEPENDENTLY DERIVED as sqrt(226) = 15.033296378372908, as for fold A, which fold C mirrors: it trains on y = [10,12,20,22] with mean 16.0 and tests on [30,32], giving errors of +14 and +16, MSE (196+256)/2 = 226.0. One square root of an exact integer, so 1 ulp near 2e-15, and 1e-9 admits that with six orders of margin. |
| `rmse_A` | 15.0332963784 | abs 1e-06 | A REGRESSION pin on a fitted model. The gbm is deterministic at a fixed seed, so it reproduces exactly on the same scikit-learn version, but a version change can alter the histogram binning in the last bits. 1e-6 admits that while still detecting any real change in fit. It is looser than the hand-derived baselines in the same file precisely because its provenance is weaker, and the file says so. |
| `rmse_B` | 1.0 | abs 1e-06 | Same basis as rmse_A. A REGRESSION pin on a fitted model. The gbm is deterministic at a fixed seed, so it reproduces exactly on the same scikit- learn version, but a version change can alter the histogram binning in the last bits. 1e-6 admits that while still detecting any real change in fit. It is looser than the hand-derived baselines in the same file precisely because its provenance is weaker, and the file says so. |
| `rmse_C` | 15.0332963784 | abs 1e-06 | Same basis as rmse_A. A REGRESSION pin on a fitted model. The gbm is deterministic at a fixed seed, so it reproduces exactly on the same scikit- learn version, but a version change can alter the histogram binning in the last bits. 1e-6 admits that while still detecting any real change in fit. It is looser than the hand-derived baselines in the same file precisely because its provenance is weaker, and the file says so. |
| `baseline_ridge_rmse_A` | 0.0022628130162319375 | rel 1e-06 | A RELATIVE tolerance on a regression pin, because the value is 2.3e-3 and a 1e-6 absolute tolerance would be a 0.04 percent band that a RidgeCV alpha- grid change could break, while an absolute tolerance scaled to the other outputs would pass anything. The alpha is selected by internal cross- validation over a 25-point grid, so this value is the least portable in the suite and is pinned relatively. |
| `baseline_ridge_rmse_B` | 0.0017341949999050144 | rel 1e-06 | Same basis as baseline_ridge_rmse_A. A RELATIVE tolerance on a regression pin, because the value is 2.3e-3 and a 1e-6 absolute tolerance would be a 0.04 percent band that a RidgeCV alpha-grid change could break, while an absolute tolerance scaled to the other outputs would pass anything. The alpha is selected by internal cross-validation over a 25-point grid, so this value is the least portable in the suite and is pinned relatively. |
| `baseline_ridge_rmse_C` | 0.0022628130162319375 | rel 1e-06 | Same basis as baseline_ridge_rmse_A. A RELATIVE tolerance on a regression pin, because the value is 2.3e-3 and a 1e-6 absolute tolerance would be a 0.04 percent band that a RidgeCV alpha-grid change could break, while an absolute tolerance scaled to the other outputs would pass anything. The alpha is selected by internal cross-validation over a 25-point grid, so this value is the least portable in the suite and is pinned relatively. |

---

## RV-03: EVPI on a two-action problem whose closed-form answer is 1/16, showing the nested-Monte-Carlo bias

- File: `data/golden/RV-03-evpi.yaml`
- Kind: **regression**, derivation `mixed`
- Module under test: `ae.agent.decisions`
- Dispatch entry: `evpi`

**Reference.** Howard, R.A. 1966, Information Value Theory, IEEE Transactions on Systems Science and Cybernetics 2(1):22-26, doi:10.1109/TSSC.1966.300074, verified against Crossref in this repository's corrections record. Raiffa, H. and Schlaifer, R. 1961, Applied Statistical Decision Theory, Division of Research, Graduate School of Business Administration, Harvard University, for the decision-analytic framing.

**Why this is a regression vector.** The EVPI of this problem has an exact closed form, 1/16 = 0.0625, derived below with a calculator. The value the module returns does NOT equal it and cannot be expected to: the nested Monte Carlo estimator is biased UPWARD at finite inner sample size, and the baseline is itself a Monte Carlo estimate. The expected value in this file is therefore the MEASURED value at a stated seed and sample size, so the file is a REGRESSION vector. The closed form is carried alongside as analytic_reference and the test checks the seeded value against the pin AND the bias against its stated direction, which is the part that tests the model rather than the snapshot.

**Inputs.**

```yaml
theta_low: -0.5
theta_high: 1.5
actions:
- name: wait
  cost: 0.0
- name: build
  cost: 0.0
null_action: wait
n_outer: 2048
n_inner: 256
seed: 0
```

**Provenance.**

- `theta_low`: ASSUMED, -0.5. The parameter is the payoff of building, uniform on (-0.5, 1.5), so a
  quarter of the prior mass makes building the wrong choice. That fraction is what
  perfect information is worth.
- `theta_high`: ASSUMED, 1.5.
- `actions`: ASSUMED, two zero-cost actions: wait, which always pays zero, and build, which pays
  theta. Zero action costs keep the closed form clean.
- `n_outer`: ASSUMED, 2048 outer draws.
- `n_inner`: ASSUMED, 256 inner draws. There is no remaining uncertainty in this problem once theta
  is resolved, so the inner loop averages a constant and the usual upward bias from
  taking a maximum over noisy inner estimates does NOT arise here. That is deliberate:
  it isolates the outer-loop sampling error.
- `seed`: ASSUMED, 0.

**Arithmetic chain.**

```
Prior expectations. Waiting always pays 0. Building pays theta, so
  E[build] = (low + high)/2 = (-0.5 + 1.5)/2 = 0.5
The best action under the prior is therefore build, with value 0.5.
With perfect information the decision maker builds when theta > 0 and waits
otherwise, so the value is E[max(0, theta)]:
  E[max(0, theta)] = (1/(high - low)) * integral from 0 to 1.5 of theta dtheta
                   = (1/2) * (1.5^2/2) = (1/2) * 1.125 = 0.5625
EVPI = 0.5625 - 0.5 = 0.0625 = 1/16 exactly.
Switch fraction. Perfect information changes the action exactly when theta < 0,
which has probability (0 - (-0.5))/2 = 0.25.
Sampling error of the estimator:
  Var(max(0, theta)) = E[max^2] - (E[max])^2
  E[max^2] = (1/2) * (1.5^3/3) = (1/2) * 1.125 = 0.5625
  Var = 0.5625 - 0.5625^2 = 0.5625 - 0.31640625 = 0.24609375
  sd = sqrt(0.24609375) = 0.49607837082461076
  SE of the resolved value at n_outer = 2048:
    0.49607837082461076/sqrt(2048) = 0.010961886875314282
  Var(theta) = (high - low)^2/12 = 4/12 = 0.3333333333333333
  The baseline uses max(n_outer*2, 512) = 4096 draws, so its SE is
    sqrt(0.3333333333333333/4096) = 0.009021097956087902
  Combined SE of the difference:
    sqrt(0.010961886875314282^2 + 0.009021097956087902^2) = 0.014196590161039404
So the estimator should land within about 3 combined SE, 0.042589770483118212,
of 0.0625.
MEASURED at seed 0, n_outer = 2048, n_inner = 256:
  evpi = 0.06732450807636592
  baseline_value = 0.4928314937557836 (against the true 0.5)
  resolved_value = 0.5601560018321495 (against the true 0.5625)
  switch_fraction = 0.2548828125 (against the true 0.25)
The deviation of the EVPI estimate from 1/16 is +0.004824508076365919, which is
0.3398356944617673 of a combined SE. Both the baseline and the resolved value came in LOW, and because EVPI is
their difference the two errors partly cancel, which is why the EVPI estimate is
closer to truth in SE terms than either input.
Seed dependence, measured at n_outer = 8192: seed 0 gave 0.05622818277750663,
seed 1 gave 0.07114787799362726, seed 2 gave 0.06457173732175125. The spread of
0.014919695216120632 across three seeds at four times the sample size confirms the sampling
error is the dominant term and not a bias, since the values straddle 0.0625.
Inner-sample independence check: at n_outer = 2048 with n_inner raised from 256 to
2048 the measured EVPI was 0.06732450807636547 against 0.06732450807636592, a
difference of 4.5e-16. That confirms the claim in the provenance that the inner
loop averages a constant in this problem, so no upward nesting bias is present
and the entire deviation from 1/16 is outer-loop sampling error.
```

**Expected outputs and tolerances.**

| output | expected | tolerance | reason the tolerance is what it is |
| --- | --- | --- | --- |
| `evpi` | 0.06732450807636592 | abs 1e-12 | A REGRESSION pin. The seed fixes the sample, so the value is deterministic to floating point on the same numpy version, and 1e-12 admits summation- order differences across platforms. This tolerance makes NO claim about statistical accuracy; the 0.0141950 combined standard error in analytic_reference is the honest uncertainty and the test separately checks that the measured value lies within 3 of those of the closed-form 1/16. |
| `baseline_value` | 0.4928314937557836 | abs 1e-12 | A seeded regression pin, as for evpi. Its own sampling error against the true 0.5 is sqrt(Var(theta)/4096) = 0.009021097956087902, which the test checks separately. |
| `resolved_value` | 0.5601560018321495 | abs 1e-12 | A seeded regression pin. Its sampling error against the true 0.5625 is 0.010961886875314282. |
| `switch_fraction` | 0.2548828125 | abs 1e-12 | A seeded regression pin on a ratio of integers (switches over n_outer), so it is exactly representable and reproduces bit for bit. The statistical check against the true 0.25 uses the binomial standard error sqrt(0.25*0.75/2048) = 0.009568319307746789, which the test applies separately. |

**Closed-form reference, carried alongside the seeded values.**

| quantity | exact value |
| --- | --- |
| `evpi` | 0.0625 |
| `baseline_value` | 0.5 |
| `resolved_value` | 0.5625 |
| `switch_fraction` | 0.25 |
| `combined_standard_error` | 0.0141965902 |

