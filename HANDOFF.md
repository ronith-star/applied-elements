# HANDOFF: what is validated, what is scaffolding, what needs data

Measured at commit `fbefb47`. Every number below was produced by a command run in
this repository during the session that wrote this file. Where a quantity could
not be measured or sourced, it says so.

Read this file before trusting any output of this platform.

## Summary

| | count | how counted |
| --- | --- | --- |
| modules in `src/ae` | 27 | `find src/ae -name '*.py' ! -name '__init__.py'` |
| modules with an external literature benchmark | 13 | join of `validation_record.csv` to source modules |
| modules with analytic or self-consistency benchmarks only | 2 | same |
| modules with hand-traceable golden vectors only | 7 | same |
| modules with no benchmark and no golden vector | 5 | same |
| marked tests in the validation record | 162 | `python scripts/export_validation.py` |
| of which golden | 115 | same |
| of which benchmark | 47 | same |
| benchmarks stating a numeric error | 22 of 47 | same |

Counting "validated" strictly, meaning a model checked against a number someone
else published: **13 of 27 modules**. Counting "stubbed or unvalidated", meaning
no benchmark of any kind: **5 of 27 modules**, plus the ML surrogate and the agent
decision layer, which have benchmarks but no real data behind them and are
described under STUBBED below.

## VALIDATED

### A warning about the literature benchmark count, read this first

`scripts/export_validation.py` reports 28 benchmarks classified `literature`.
**That number overstates the external evidence and should not be quoted.**

The classifier `_benchmark_kind()` assigns `literature` by a chain of keyword
tests, and its final fallback is: if the module under test carries a DOI anywhere
in its reference block, classify the benchmark as `literature`. I probed the
classifier directly this session by importing it and re-running it on each
benchmark's docstring with and without the module DOI list. Measured result:

| route to the `literature` class | count |
| --- | --- |
| a reported-value phrase in the test's own docstring | 6 |
| module-DOI fallback only, no reported-value phrase | 22 |

Of the 28 `literature` rows, 15 carry no DOI in their own `dois` column and rely
entirely on the module's reference block.

The fallback mislabels real cases. The test
`uncertainty::test_zero_first_order_with_large_total_is_the_case_tornado_misses`
classifies as `literature` with the module DOIs
present and `unclassified` without them, and none of the four keyword lists fire.
That test validates against the Ishigami function's closed form; there is no
external measurement in it at all. The same pattern holds for
`separation::test_benchmark_whims_field_exponent_sensitivity`, whose own printed
output says "No measured exponent exists; this is the cost of the assumption, not
a validated result", and for `separation::test_benchmark_mass_balance_residuals`,
whose reference is exact conservation and whose printed residuals are
`0.000e+00` because they are implementation residuals, not model errors.

So the honest statement is: 13 modules carry at least one benchmark against a
published measurement, and the list below gives the specific measurement and the
error I measured by running the test. The classifier's `literature` count is a
loose upper bound.

### Benchmarks against published measurements, with the error I measured

Every error below was printed by the test when I ran it and read from that output.
Where the test prints no error, the row says what it does print instead.

**`physics/leaching`**
`test_benchmark_xia_2024_total_removal`: literature 81.20 percent removal, model
81.1966 percent, error 0.0041 percent (Xia et al. 2024,
`doi:10.3390/min14070727`). The test's own docstring is explicit that this
validates the accounting identity, not the kinetic model: the paper reports no
per-stage or conversion-time data.
`test_benchmark_yang_2020_iron_removal`: literature 74.0 percent, model 73.979
percent, error 0.028 percent (`doi:10.1515/htmp-2020-0081`; Fe2O3 0.0857 to
0.0223 percent).
`test_benchmark_yang_2020_activation_energy_band`: 27.72 kJ/mol (ultrasound) and
20.44 kJ/mol (regular) both land inside the module's `PRODUCT_LAYER` band of 20
to 40 kJ/mol. A consistency check, no error computed.

**`physics/impurity_location`**
`test_benchmark_xia_2024_endpoints`: removal published 81.20 percent, model
81.1966, error 0.0041 percent relative; SiO2 published 99.998 wt percent, model
99.997577, error 4.23e-04 percent relative. The test states in its own output
that the SiO2 agreement is a rounding check on a by-difference convention, not an
independent measurement.
`test_benchmark_qu_2025_two_ore_bodies` (`doi:10.5194/ejm-37-953-2025`): HT
322.96 to 18.72 ug/g (94.20 percent removal), PX 4944.73 to 114.56 ug/g (97.68
percent). The test reports a real negative finding: by-difference SiO2 exceeds
the published upper bound in both cases (HT +0.0281, PX +0.0035 percent
relative), so Qu et al.'s SiO2 is not 100 minus the cation trace sum, and trace
sum in ug/g is the comparable metric. It also records that PX's 114.56 ug/g
residual is 2.29 times the 50 ppm HPQ trace sum limit despite 97.68 percent
removal: feed grade, not removal efficiency, decides HPQ viability.
`test_benchmark_prior_floors_against_xia_residual`: the module's `ASSUMED` vein
quartz lattice fractions imply a 39.58 ug/g residual from a 128.86 ug/g feed,
that is 69.29 percent removal, against the published 81.20 percent. **Error 14.67
percent relative.** The test's own conclusion, printed: the priors overstate
lattice content and must be replaced by LA-ICP-MS measurement.

**`physics/comminution`** (`doi:10.37190/ppmp/172458`, Arellano-Pina et al. 2023)
`test_benchmark_ore_a_work_index_round_trip`: published 12.30 kWh/metric ton,
model 12.300000, error 1.44e-14 percent. This is a round trip through the
equation, so it validates the implementation and the unit convention, not the
model's predictive accuracy.
`test_benchmark_reduced_procedure_spread`: the paper's own four standard versus
reduced procedure spreads, 4.1, 3.8, 5.3 and 4.2 percent, recomputed as 4.065,
3.774, 5.348 and 4.242 percent, differences 0.035 to 0.048 percentage points.
Mean procedure spread 4.36 percent, which the test labels the accuracy floor of
any Bond work index.
`test_benchmark_prior_against_sourced_measurement`: the `ASSUMED` vein quartz
prior 13.60 kWh/short ton against measured Ore A 11.158 kWh/short ton, **error
21.88 percent**. The prior's uniform bracket of 11 to 17 does contain the
measurement.
`test_benchmark_grindability_equation_against_ore_a`: labelled
`[benchmark, weak]` in its own output. The paper publishes only Wi, so Eq. E2
cannot be checked directly; the inverse solve gives G = 1.6333 g/rev reproducing
12.300000 against target 12.30 at error 0.00e+00 percent, and G lands in the
physically observed 0.5 to 3 g/rev range. The printed output says outright: "Ore
A's own G is not published, so this checks self-consistency, not accuracy."

**`physics/thermal`**
`test_benchmark_theoretical_melt_energy_against_lbnl`
(`doi:10.2172/927883`, Galitsky and Worrell 2008): LBNL 2.2 MMBtu/short ton =
710.7 kWh/tonne, model 599.7 kWh/tonne, **error -15.62 percent**. The sign is
expected negative: there is no carbonate decomposition in pure silica.
`test_benchmark_electric_melter_efficiency_against_lbnl`: the model floor of
599.7 kWh/tonne implies 18.0 percent efficiency against a specialty-glass
electric melter at 10.3 MMBtu/short ton (3327.5 kWh/tonne) and 69.7 percent
against state of the art at 780 kWh/short ton (859.8 kWh/tonne). LBNL's stated
band for continuous furnaces is 33 to 40 percent, so the model floor brackets it
rather than matching it.
`test_benchmark_quartz_cp_against_dulong_petit`: classical limit 3 x 3 x R =
74.830 J/(mol K), model ds62 polynomial 68.898, error -7.93 percent, which must
be negative because a solid approaches the limit from below. This is a benchmark
against a physical law, not a published measurement.
`test_benchmark_transition_enthalpies_are_small_and_positive`: quartz to
tridymite 2.756 kJ/mol (45.9 kJ/kg) at 1143.15 K, quartz to cristobalite 3.059
kJ/mol (50.9 kJ/kg) at 1743.15 K, cristobalite to liquid 9.069 kJ/mol (150.9
kJ/kg) at 1996.00 K. Checked against the accepted order of magnitude; no error
computed.

**`physics/phases`**
`test_benchmark_inversion_temperature`: literature 573 degC (846.15 K), model
ds62 Landau Tc_0 847.00 K (573.85 degC), error +0.100 percent.
`test_benchmark_tridymite_boundary_from_gibbs_energies`: literature 870.0 degC,
model root of G_quartz = G_tridymite at 866.11 degC (1139.26 K), **error -0.45
percent**. The same dataset places the tridymite to cristobalite boundary above
2600 K against an accepted 1470 degC, which is documented as item 3 in
`thermal.LIMITATIONS` rather than hidden.
`test_benchmark_cumulative_strain_against_the_trade_figure`: trade figure 3.70
vol percent, model 4.2147, **error +13.91 percent**. The test notes that
Ringdalen 2015 measures the step at inversion as 0.4 vol percent, a different
quantity, so the two are not interchangeable.

**`physics/psd`**
`test_benchmark_geometric_ssa_against_ringdalen_bet` (`doi:10.1007/s11837-014-1149-y`): Ringdalen BET midpoint 0.420 m2/g (band 0.38 to 0.47) against a
geometric 0.002264 m2/g for a 1 mm lump, **error -99.46 percent, BET 185 times
larger**, and 0.000226 m2/g for a 10 mm lump, error -99.95 percent, 1855 times
larger. The equivalent Sauter mean that would reproduce the BET value is 5.39 um.
The docstring reports rather than asserts this error because it is not a model
defect: geometric surface area of a smooth sphere and BET surface area of a rough
porous lump are different quantities, and the factor is the roughness.

**`physics/packing`**
`test_benchmark_furnas_against_mcgeary_quaternary` (`doi:10.1111/j.1151-2916.1961.tb13716.x`,
McGeary 1961): literature phi_max 0.9510, model 0.98022, **error
+3.07 percent**; the finest-class composition error is **-44.9 percent**, the
model being short of fines. Size ratios 1:7:38:316, monomodal phi_1 = 0.625.

**`physics/diffusion`**
`test_benchmark_ti_barrier_against_liu_2026` (`doi:10.3390/min16080836`):
activation energy about 400 kJ/mol. The test prints a diffusion length deficit of
2.92 orders of magnitude at D0 = 1e-4 m2/s and 5.92 orders at D0 = 1e-10, a
prefactor spread of 3.00 orders. It states explicitly that no literature
diffusion length exists to difference against, because the paper reports Ea only.
`test_benchmark_xia_2024_residual_is_lattice`: a synthetic fixture floor of 31.50
ppm (Al 18.0 + Ti 9.0 + Li 4.0 + B 0.5) against the Xia residual of 24.23 ug/g, a
30.0 percent difference. The test states this is not a validation of the
fixture's synthetic lattice split, only an order-of-magnitude check.

**`physics/liberation`**
`test_benchmark_grind_implied_by_xia_removal`: the inversion reproduces 81.1966
percent exposure at d_p = 2.3414 x d_inc, inversion error 0.00e+00 percent. The
printed caveat: the paper's flowsheet includes calcination plus water quenching,
which opens fluid inclusions thermally, so the plant is not bound by this
geometric ratio. "This validates the inversion, not the physics of that campaign."
`test_benchmark_qu_2025_removal_requires_finer_grind`: required grinding-only
ratios HT 1.6313, PX 1.3988, Xia 2.3414, all labelled upper bounds rather than
predictions because all three flowsheets include calcination.

**`physics/reagents`**
`test_benchmark_impurity_load_against_xia_2024`: a 128.86 ug/g feed treated as Al
gives 4.775865 mol Al per tonne and an HCl demand of 14.327595 mol = 0.52236 kg
per tonne at the trivalent charge rule. **The paper reports no reagent
consumption, so no error against literature can be computed** and the test says
so.
`test_benchmark_yang_2020_iron_acid_demand`: 634 ppm Fe2O3 removed = 443.44 ppm
Fe, implying 7.94053 mol Fe and 23.82160 mol HCl = 0.86849 kg per tonne. Again no
published acid consumption exists to compare against.

**`physics/chlorination`**
`test_benchmark_chloride_boiling_points`: AlCl3 brief 180.0 degC against PubChem
180.00, error 0.000 percent; FeCl3 315.0 against 316.00, error 0.317 percent;
TiCl4 136.0 against 136.45, error 0.331 percent. Note what this actually checks:
the reference is the value stated in the project brief, and the module's values
come from PubChem property records, so it is a cross-check of two secondary
sources.
`test_benchmark_alkali_chlorides_are_not_volatile_at_roast`: NaCl 1465, KCl
sublimes 1500, LiCl 1360 degC, all 1 atm transition points above a 1200 degC
roast, with computed p_sat 16108, 12635 and 33381 Pa respectively. The dHvap
comes from Trouton's rule, that is an estimate, and the test labels it so. No
percent error.

**`physics/separation`**
`test_benchmark_flowsheet_grade_steps_against_lin_2020`
(`doi:10.1007/s42461-020-00247-0`, read from `src/ae/physics/separation.py:334`):
reject yields 1.980, 0.180 and
0.018 percent for the three grade steps. Table 3 of the paper gives grade bands in
a conceptual diagram and no yields, so these are the yields the mass balance
requires, not published values. Classified `self_consistency`.

**`econ/uncertainty`**
`test_sobol_reproduces_ishigami_analytic_indices`: Ishigami at N = 4096, 20480
evaluations, analytic variance 13.8446 against sampled 13.8151, error 0.21
percent. Analytic, not literature.

**`plant/scheduling`**
`test_simulation_reproduces_mm1_waiting_time`: error 0.30 percent.
`test_deterministic_service_matches_mm_d_1_bound`: error 1.68 percent. Both
analytic queueing results, not published measurements.

### What the golden vectors do and do not prove

115 of the 162 marked tests are `golden`: a hand-computed input and output pair
checked into the test, traceable by hand. They catch regressions and arithmetic
errors. They do not validate the model against reality, because the expected value
was computed by the same person who wrote the model. The seven modules whose
strongest evidence is a golden vector (`core/feedstock`, `econ/capex`,
`econ/unit_economics`, `econ/valuation`, `plant/capacity`, `plant/streams`,
`plant/yield_cascade`) should be read as arithmetic that has been checked, not as
physics or economics that has been validated.

## STUBBED

### The ML surrogate is trained on synthetic data

`src/ae/ml/surrogate.py` contains no `read_csv`, no data file load, and one
`np.random.default_rng(seed)` call at line 540. The training data in every test
comes from a `synthetic()` helper defined at line 14 of
`tests/test_surrogate.py`, with signature `synthetic(n_deposits=6, per_deposit=40,
seed=0, deposit_offset=0.0, feature_separation=...)`, which fabricates `al`, `ti`,
`fe` and `size` features and a target with a per-deposit offset.

The module's four benchmarks are all analytic: they check that grouped
leave-one-deposit-out validation is pessimistic relative to random k-fold, that
permutation importance recovers the known mechanism of the generator, that
baselines and per-fold spread are reported, and that out-of-distribution inputs
are refused. Those are correctness properties of the machinery. **None of them is
evidence that the surrogate predicts anything about real ore**, and the module's
own docstring states that predicting a lattice impurity ceiling for an
uncharacterized deposit is not a legitimate use.

What the module does do well, and what should be preserved by anyone retraining
it on real data: grouped validation by deposit (random k-fold leaks deposit
identity into the training set and inflates R2), mandatory mean-predictor and
ridge baselines (a model that cannot beat the mean is worthless and the harness
makes that visible), R2 computed against the global mean of the held-out fold's
parent dataset, and OOD refusal by Mahalanobis distance with a nearest-neighbour
fallback that announces when it has fallen back. Its stated limitations include
that with fewer than about five deposits the per-fold spread is wide and that no
calibrated predictive interval is provided.

### The agent decision layer has ASSUMED priors and no tests

`src/ae/agent/decisions.py` computes EVPI by nested Monte Carlo. Two facts about
its status:

1. **The priors are `ASSUMED`.** No ore is characterized, so the distributions
   that EVPI is computed against are engineering judgement. The module's own
   docstring says the output is "a RANKING to argue with, not a budget to
   execute". EVPI against a confident prior is small no matter how physically
   important the parameter is, because a confident prior says there is little left
   to learn; with `ASSUMED` priors the confidence itself is invented.
2. **There is no test file.** `tests/test_decisions.py` does not exist, the module
   contributes 0 of the 1,733 collected tests, and it has no doctest. I verified
   this session that its public API imports and that `evpi`, `rank_measurements`
   and `measurement_priority` have the signatures given in `PLAN.md`. Nothing
   beyond importability is verified.

It is also worth stating what the module is not, because the name invites
misreading: there is no LLM call in it and no autonomous loop. It is a Monte Carlo
estimator with a decision-theoretic wrapper.

### Modules with no benchmark and no golden vector

`core/units` (35 tests), `core/provenance` (19), `core/registry` (15),
`core/site` (13) and `agent/decisions` (0). The first four are infrastructure:
their tests check unit round trips, validator rejection, key resolution and FX
routing, which is behaviour rather than physics, so the absence of a benchmark is
appropriate rather than a gap. `agent/decisions` is a genuine gap.

### The workbook self-consistency check does not run here

`tests/test_workbook.py` calls `pytest.importorskip("formulas")` at line 38,
because recalculating the Excel mirror needs a formula engine. `formulas` is not
installed in the `ae-platform` environment (`python -c "import formulas"` raises
`ModuleNotFoundError`), so 2 tests skip at collection, including
`test_workbook_formulas_reproduce_the_python_model`, which is the only
self-consistency benchmark in `validation_record.csv` with no matching test case
in a junit run. **The claim that the Excel mirror reproduces the Python model is
therefore not verified in this environment.** Installing `formulas` would verify
it.

That file's docstring records one defect it caught while being written: the first
cost build omitted fixed costs entirely, summing to a cash cost of 157.77 USD per
tonne against the corrected 795.02, a factor of 5.0, and every scenario looked
profitable. The same docstring retracts a second defect an earlier version
claimed (stale row offsets in the downstream NPV and breakeven formulas), which
never occurred; see `docs/CORRECTIONS.md` entries C3 and C4.

### Provenance coverage, measured

From `python scripts/export_registry.py`: 203 provenance-tracked values, of which
`SOURCED` 134, `ASSUMED` 61, `DERIVED` 8. All 134 `SOURCED` values carry a DOI or
a URL (134 of 134). Zero point estimates lack a stated uncertainty. Tier
distribution: 133 rows at Tier 1, 1 row at Tier 2, 69 rows with no tier recorded,
and **zero rows at Tier 3**, which is the intended state (see
`CONTRIBUTING-provenance.md`).

`ASSUMED` values are not spread evenly, and the concentration tells you where the
platform is weakest:

| module | ASSUMED | SOURCED |
| --- | --- | --- |
| `impurity_location` | 18 | 3 |
| `diffusion` | 10 | 11 |
| `leaching` | 9 | 9 |
| `thermal` | 0 | 30 |
| `packing` | 0 | 4 |

`impurity_location` is the module that decides the ceiling grade of a deposit and
it is the module with the highest assumed-to-sourced ratio in the platform, 18 to
3. This is not an accident of bookkeeping: partition of impurities between lattice
sites, fluid inclusions and mineral inclusions is deposit-specific and is not
transferable from published work on other deposits. Its own benchmark measures
the cost of those assumptions at 14.67 percent relative error against Xia et al.
2024, in the direction of overstating lattice content.

## NEEDS DATA

This is the section that says what to go measure. The gate is
`src/ae/core/feedstock.py`, whose `permits_output` mapping refuses model outputs
that the current characterization tier does not support.

**The Vikarabad deposit is at tier `unmeasured`.** No assay exists in the citable
record. The source tree states this in the negative in two places rather than
implying otherwise: `physics/comminution.py:146` says "No work index in this
module is measured on Vikarabad ore", and `physics/impurity_location.py:115` says
"No partition data exists for the Vikarabad deposit." Every grade-conditional
result the platform can produce is a scenario gated on a future assay campaign,
never a property of the ore.

### Tier `unmeasured` to tier `screened`

**Measure:** whole-rock major and trace elements on 20 to 30 samples spanning the
deposit, sample ids following the `AE-Q-IN-VKB-001` convention (quartz, India,
Vikarabad, sample number). The required suite is Al, Ti, Li, Fe, Na, K, B.
**Method note that decides the instrument:** XRF will not do on its own. XRF
cannot measure Li or B at all, and has poor ppm detection limits for Ti and the
alkalis. The suite needs ICP-OES or ICP-MS on dissolved splits. A correction worth
carrying: bulk XRF's limitation is not that it "cannot resolve lattice
impurities", which is wrong because XRF measures total element content
irrespective of where the atoms sit; its limitations are Li and B blindness, ppm
detection limits, and no spatial information, which is true of every bulk method.
**Unlocks:** `screening_rank` and `mass_yield_estimate`. That is: you can rank
zones against each other and put a first number on how much mass survives
beneficiation.

### Tier `screened` to tier `bulk_quantified`

**Measure:** quantified bulk concentrations with uncertainties for the full
required suite, plus U and Th if the low-alpha filler route is in scope, on enough
samples to establish spatial variance rather than a single composite.
**Unlocks:** `reagent_demand` and `impurity_removal_estimate`. This is the tier
at which `physics/reagents` and `physics/leaching` produce numbers that mean
something: acid demand scales off the measured cation load, and the `reagents`
benchmarks show the calculation is implemented correctly (4.775865 mol Al per
tonne from a 128.86 ug/g feed) but has no published consumption to validate
against, so the input has to be your own assay.

### Tier `bulk_quantified` to tier `located`

**Measure:** the spatial distribution of each impurity. Concretely: LA-ICP-MS
spot and depth-profile analysis to separate lattice-bound from
inclusion-hosted content, cathodoluminescence to map growth zoning and Ti and Al
substitution, fluid inclusion density and microthermometry, and SEM or
petrographic identification of mineral inclusion phases.
**Unlocks:** `purification_ceiling`, `product_grade_claim` and
`qualification_dossier`. Nothing below this tier can support a grade claim,
because lattice-bound Al, Ti, Li and B cannot be removed by acid leaching at any
reagent loading and therefore set the ceiling grade.
**Why this tier is the one that matters most:** `impurity_location` runs on
`ASSUMED` lattice fractions today (Al 0.45, Ti 0.60, Li 0.75), and its own
benchmark measures those priors at 14.67 percent relative error against a
published campaign, overstating lattice content. The 18 `ASSUMED` values in that
module are exactly what an LA-ICP-MS campaign replaces. Until it is done, the
purification ceiling is refused by the tier gate rather than estimated, which is
the correct behaviour.

### Measurements that unlock specific models, keyed to the module

| measurement | module it feeds | what it replaces | what it unlocks |
| --- | --- | --- | --- |
| ICP-MS trace suite, 20 to 30 samples | `core/feedstock` | tier `unmeasured` | screening rank, mass yield estimate |
| spatial variance of the trace suite | `core/feedstock`, `plant/yield_cascade` | single-composite assumption | quantified feed distribution for Monte Carlo |
| LA-ICP-MS lattice versus inclusion split | `physics/impurity_location` | 18 `ASSUMED` values, Al 0.45, Ti 0.60, Li 0.75 priors | purification ceiling, product grade claim |
| fluid inclusion density and microthermometry | `physics/liberation`, `physics/phases` | assumed inclusion size and decrepitation behaviour | grind target that is not an upper bound |
| Bond ball mill work index on Vikarabad ore | `physics/comminution` | `ASSUMED` vein quartz prior 13.60 kWh/short ton, measured at 21.88 percent error against the one published quartz-rich ore | comminution energy and mill sizing |
| bench acid leach with staged sampling | `physics/leaching` | accounting identity only, no kinetic validation | conversion-time curve, stage count |
| BET surface area on the actual size fractions | `physics/psd` | geometric SSA, which is 185 to 1855 times low against published BET | reagent contact area, dissolution rate |
| measured WHIMS field exponent | `physics/separation` | `n_B` `ASSUMED` 2.0, bracket 1 to 3, which spreads recovery by 17.7 percent at 60 s and 89.4 percent at 10 s | magnetic separation sizing |
| Ti and Al diffusion prefactor D0 on this quartz | `physics/diffusion` | prefactor spread of 3.00 orders of magnitude | whether high-temperature treatment can move lattice impurities at all |
| real deposit dataset, five or more deposits | `ml/surrogate` | `synthetic()` fixture | a surrogate with any claim on reality |
| any of the above, as an EVPI input | `agent/decisions` | `ASSUMED` priors | a measurement ranking that is not judgement alone |

### The one measurement to buy first, and why this is not my call

The platform's own answer is whatever `agent/decisions.rank_measurements()`
returns, and that function runs on `ASSUMED` priors and has no test file, so its
ranking should be argued with rather than executed. What the measured evidence
supports saying: `impurity_location` carries the worst assumed-to-sourced ratio in
the platform (18 to 3), its priors are measured at 14.67 percent relative error in
a direction that overstates lattice content, and it gates the three outputs that
any customer conversation requires (`purification_ceiling`,
`product_grade_claim`, `qualification_dossier`). An LA-ICP-MS campaign is
therefore the measurement that unlocks the most refused outputs per rupee. That
is an argument from the tier gate and the error measurement, not a decision
analysis.

## A guard on this document

`tests/test_handoff_claims.py` re-derives the structural numbers in this file,
`PLAN.md`, `README.md` and `CONTRIBUTING-provenance.md` from the source tree and
the registry CSVs, and fails if a document has drifted. It checks the module and
import-edge counts, the zero-import claim between layers in both directions, the
fan-out figures, the evidence-class partition (13 / 2 / 7 / 5, which must sum to
27), the registry tag totals and the empty Tier 3 column, the assumed-to-sourced
ratio that the NEEDS DATA argument rests on, that the CI ceiling README.md quotes
equals the one CI enforces with a per-file split that sums to it, that no
document cites a DOI absent from `src/` or `tests/`, and that no document
contains an em dash or en dash.

The 9 guards were verified by control, each defect injected and then removed:

| injected defect | result |
| --- | --- |
| `from ae.plant.streams import Stream` added to `ae.physics.comminution` | edge-count guard failed at 58 against 57; the cross-layer guard did NOT fail, because it checked only one direction. Strengthened to check both, after which it failed with "physics now imports plant (1 edge(s))". Defect removed: all 9 pass. |
| a real DOI in this file replaced with an invented one (Elsevier prefix, nonexistent article suffix; not reproduced here, because the guard below correctly flags any DOI-shaped string in these documents that is absent from the code) | anti-fabrication guard failed, naming the string and that it appears nowhere in `src/` or `tests/`. Removed: passes. |
| em dash inserted into README.md, with the split drifted to 137 plus 10 plus 1 | dash guard failed naming line 159; ceiling guard failed on the split. Removed: both pass. |
| `13 of 27 modules` in this file changed to `16 of 27` | evidence-class guard failed. Removed: passes. |

The one-directional cross-layer guard passing against a real violation is
recorded here rather than quietly fixed, because it is the same failure mode as
`docs/CORRECTIONS.md` C6: a verification that ran and did not look at the case
that mattered.

What the guards deliberately do NOT check: the measured test counts. Re-running
the suite from inside the suite is not possible, and asserting a remembered total
is the error the whole file exists to prevent. Those counts are labelled with the
commit they were measured at, and re-measuring them is a manual step.

One further control result, found by running the full suite rather than the
guard file alone: the table above originally quoted the invented DOI verbatim,
and the anti-fabrication guard failed on this document, correctly, because a
DOI-shaped string appeared in a handoff file and nowhere in the code. The guard
cannot distinguish a citation from a worked example of a bad citation, and
making it try would weaken it. The example is described instead of quoted.

## Three findings from an independent review of this document

Recorded rather than quietly fixed, because each was a claim that read as
measured and was not.

**The figure's band arrow overstated what 39 edges means.** The arrow between the
input layer and physics was labelled "Feedstock and Site injected into every
physics model (39 import edges, measured)". The 39 is correct as a total of all
input-layer-to-physics edges, but the attribution was wrong: measured by AST
walk, `ae.core.site` is imported by only 4 of the 12 physics modules
(comminution, reagents, separation, thermal), while `ae.core.feedstock` reaches
all 12. The breakdown of the 39 is feedstock 12, units 12, provenance 11, site 4.
The label now states that breakdown instead, and the figure was re-rendered.

**The figure caption called a working-tree run a clean-clone run.** The
per-module test counts in the caption were parsed from a junit run in the working
tree I had been editing, not from the independent clone. The tree had no editable
install, which is the property the caption was reaching for, so the counts
themselves stand; the wording did not. The caption now says "in a clone with no
editable install", which is what was verified (`python -m pip show ae-platform`
reporting the package not found).

**The test-file count in README.md drifted by one.** It was corrected from 34 to
35 by measurement, and then adding `tests/test_handoff_claims.py` in the same
session made 35 wrong. No guard covered it, which is exactly the gap this guard
file exists to close, so a tenth guard now re-derives the count from the tree.
Verified by control: drifting the README back to 35 fails with "README.md says 35
test files, the tree has 36"; restoring it passes.

## The committed validation record, and what changes it

Measured at `fbefb47`. Every count in the VALIDATED section above is read from
the **committed** `data/registry/validation_record.csv`: 162 marked tests, 115
golden, 47 benchmark.

Regenerating that file moves it, for two separate reasons that must not be
confused. With `tests/test_handoff_claims.py` moved out of the tree, the
exporter produces 164 marked tests, 115 golden, 49 benchmark: two benchmark rows
added by another track's commits after the CSV was last written. With my guard
file present, its 11 `@pytest.mark.golden` guards are collected on top, raising
the golden count further. The first is upstream drift, the second is my own
additions being counted, and an earlier draft of this document attributed the
second to the first.

Neither difference touches the benchmark classification (literature 28, analytic
12, self-consistency 7), so no measured error in the VALIDATED section changes.
I have not committed a regenerated CSV. Regenerate it before quoting the
marked-test totals, and expect the golden count to rise by the number of guards
in that file, which is the expected behaviour of a marker-driven exporter rather
than drift.
