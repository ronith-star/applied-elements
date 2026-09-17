"""Falsification thresholds for the memo, solved rather than asserted.

WHY THIS EXISTS. A draft of section 7 of docs/decision-memo.md claimed that a
combined mass yield "below about 0.66 puts the project at the modelled P10,
where NPV is +1.31 MUSD". That conflated two different percentiles: 0.6604 is
the Monte Carlo P10 of OVERALL YIELD, and +1.31 MUSD is the P10 of NPV, and
they do not correspond to each other because yield explains only part of NPV
variance (combined total-order Sobol index 0.1142, correlation 0.359).

This script solves for the thresholds instead. Each one moves a single
parameter with all others held at their triangular mode, and reports the value
at which NPV crosses zero, which is the only threshold with a decision
meaning.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_integration_uncertainty import plant_npv  # noqa: E402

DOCS = ROOT / "docs"
FIELDS = ["section", "quantity", "parameter", "output", "value", "unit",
          "tag", "note"]
#: ASSUMED input bands (low, high), read here so that a note about whether a
#: solved threshold lies inside or outside its band is COMPUTED from the band
#: rather than asserted. A previous version hardcoded "below the ASSUMED band
#: low end of 0.70" for the flotation yield, which was false: the solved value
#: 0.7053 is INSIDE that band.
BANDS = {"mass_yield_leach": (0.80, 0.97), "mass_yield_flot": (0.70, 0.93),
         "price": (2800.0, 5200.0), "capex_musd": (25.0, 60.0),
         "discount_rate": (0.10, 0.20)}
NOMINAL = {"mass_yield_leach": 0.92, "mass_yield_flot": 0.85,
           "al_mean_ppm": 18.0, "al_sigma_ppm": 3.0, "price": 3500.0,
           "power_price": 0.07, "reagent_price": 1.8, "capex_musd": 38.0,
           "discount_rate": 0.14}


def npv_at(**kw: float) -> float:
    p = dict(NOMINAL)
    p.update(kw)
    return float(plant_npv(**p)["npv_musd"])


def overall_yield_npv(oy: float) -> float:
    """NPV when both mass yields are scaled to give a target overall yield."""
    s = (oy / (NOMINAL["mass_yield_leach"] * NOMINAL["mass_yield_flot"])) ** 0.5
    return npv_at(mass_yield_leach=NOMINAL["mass_yield_leach"] * s,
                  mass_yield_flot=NOMINAL["mass_yield_flot"] * s)


def main() -> None:
    res: dict[str, object] = {"nominal": NOMINAL,
                              "nominal_npv_musd": npv_at()}
    rows: list[dict[str, object]] = []

    oy0 = brentq(overall_yield_npv, 0.40, 0.70, xtol=1e-4)
    s = (oy0 / (NOMINAL["mass_yield_leach"] * NOMINAL["mass_yield_flot"])) ** 0.5
    res["overall_yield_at_zero_npv"] = {
        "overall_yield": oy0, "npv_at_threshold": overall_yield_npv(oy0),
        "implied_mass_yield_leach": NOMINAL["mass_yield_leach"] * s,
        "implied_mass_yield_flot": NOMINAL["mass_yield_flot"] * s}
    rows += [
        dict(section="falsification", quantity="overall_yield_at_zero_npv",
             parameter="overall_yield", output="npv_musd", value=oy0,
             unit="fraction", tag="COMPUTED",
             note="root of plant_npv in combined mass yield, all other "
                  "parameters at their triangular mode; both mass yields "
                  "scaled by a common factor"),
        dict(section="falsification",
             quantity="implied_mass_yield_leach_at_zero_npv",
             parameter="mass_yield_leach", output="npv_musd",
             value=NOMINAL["mass_yield_leach"] * s, unit="fraction",
             tag="COMPUTED",
             note=f"BELOW the ASSUMED band low end of {BANDS['mass_yield_leach'][0]}"
                  if NOMINAL["mass_yield_leach"] * s < BANDS["mass_yield_leach"][0]
                  else f"INSIDE the ASSUMED band, low end "
                       f"{BANDS['mass_yield_leach'][0]}"),
        dict(section="falsification",
             quantity="implied_mass_yield_flot_at_zero_npv",
             parameter="mass_yield_flot", output="npv_musd",
             value=NOMINAL["mass_yield_flot"] * s, unit="fraction",
             tag="COMPUTED",
             note=f"BELOW the ASSUMED band low end of {BANDS['mass_yield_flot'][0]}"
                  if NOMINAL["mass_yield_flot"] * s < BANDS["mass_yield_flot"][0]
                  else f"INSIDE the ASSUMED band, low end "
                       f"{BANDS['mass_yield_flot'][0]}; this yield alone does NOT "
                       f"require leaving its band"),
    ]

    # Calibration: what the MC yield P10 actually maps to, one at a time.
    mc = json.loads((DOCS / "memo_numbers.json").read_text())
    yp10 = mc["mc"]["summary"]["overall_yield"]["P10"]
    res["mc_yield_p10_maps_to_npv"] = {"overall_yield_p10": yp10,
                                       "npv_musd": overall_yield_npv(yp10)}
    rows.append(dict(section="falsification",
                     quantity="npv_at_mc_overall_yield_p10",
                     parameter="overall_yield", output="npv_musd",
                     value=overall_yield_npv(yp10), unit="MUSD",
                     tag="COMPUTED",
                     note=f"NPV when overall yield is set to its Monte Carlo "
                          f"P10 of {yp10:.4f} and all else sits at its mode; "
                          f"this is NOT the NPV P10, which is a joint outcome"))

    for name, lo, hi in (("price", 1500.0, 3500.0),
                         ("capex_musd", 38.0, 120.0),
                         ("discount_rate", 0.14, 0.60)):
        try:
            x0 = brentq(lambda x: npv_at(**{name: x}), lo, hi, xtol=1e-6)
        except ValueError:
            res[f"{name}_at_zero_npv"] = None
            continue
        res[f"{name}_at_zero_npv"] = x0
        unit = {"price": "USD/t", "capex_musd": "MUSD",
                "discount_rate": "fraction"}[name]
        blo, bhi = BANDS[name]
        if x0 < blo:
            where = f"BELOW the ASSUMED band ({blo} to {bhi}): this parameter " \
                    f"alone cannot drive NPV negative inside its own band"
        elif x0 > bhi:
            where = f"ABOVE the ASSUMED band ({blo} to {bhi}): this parameter " \
                    f"alone cannot drive NPV negative inside its own band"
        else:
            where = f"INSIDE the ASSUMED band ({blo} to {bhi}): this parameter " \
                    f"alone CAN drive NPV negative within its own band"
        rows.append(dict(section="falsification",
                         quantity=f"{name}_at_zero_npv", parameter=name,
                         output="npv_musd", value=x0, unit=unit,
                         tag="COMPUTED",
                         note="root of plant_npv in this parameter alone, all "
                              "others at their triangular mode; " + where))

    # Yield is a weak driver of NPV: record the evidence for that claim.
    sob = mc["sobol"]["npv_musd"]["total_order"]
    combined = sob["mass_yield_flot"] + sob["mass_yield_leach"]
    rows.append(dict(section="falsification",
                     quantity="combined_mass_yield_total_order_sobol",
                     parameter="mass_yield_flot+mass_yield_leach",
                     output="npv_musd", value=combined, unit="index",
                     tag="COMPUTED",
                     note="why a yield percentile does not map to an NPV "
                          "percentile: the two mass yields together carry only "
                          "this share of NPV variance"))
    res["combined_mass_yield_total_order_sobol"] = combined

    (DOCS / "memo_falsification.json").write_text(
        json.dumps(res, indent=1, default=str))

    path = DOCS / "memo_numbers.csv"
    existing = [r for r in csv.DictReader(path.open())
                if r["section"] != "falsification"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(existing)
        w.writerows(rows)

    print(f"[nominal] NPV {res['nominal_npv_musd']:+.4f} MUSD at all modes")
    print(f"[threshold] overall yield at zero NPV: {oy0:.4f} "
          f"(leach {NOMINAL['mass_yield_leach']*s:.4f}, "
          f"flot {NOMINAL['mass_yield_flot']*s:.4f})")
    print(f"[calibration] MC overall-yield P10 {yp10:.4f} maps to NPV "
          f"{overall_yield_npv(yp10):+.2f} MUSD, not to the NPV P10")
    for k in ("price_at_zero_npv", "capex_musd_at_zero_npv",
              "discount_rate_at_zero_npv"):
        print(f"[threshold] {k}: {res[k]}")
    print(f"[evidence] combined mass yield total-order Sobol {combined:.4f}")
    print(f"[csv] kept {len(existing)} rows, wrote {len(rows)} falsification "
          f"rows, total {len(existing)+len(rows)}")


if __name__ == "__main__":
    main()
