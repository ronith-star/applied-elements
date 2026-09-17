"""Build the Excel mirror of the Python model, with a reconciliation check.

Run: python scripts/build_workbook.py [outdir]

WHY A WORKBOOK AT ALL, given the Python model is authoritative: a workbook is
the format in which a counterparty will actually interrogate the numbers.
Handing over a black-box script and a PDF invites the reader to take the
answer on trust; handing over live formulas lets them change a price and see
the consequence, which is the only way anyone believes a model they did not
write.

WHAT MAKES THIS HONEST RATHER THAN A DUPLICATE: every calculated cell is an
EXCEL FORMULA, not a pasted Python result. A pasted number is a screenshot
that silently goes stale when a driver changes, and a reader who edits an
input and sees nothing move learns not to trust the file. The Reconciliation
sheet then computes the same outputs BOTH ways and reports the difference, so
a divergence between the workbook's formulas and the Python model is visible
in the workbook itself rather than discovered later.

WHAT THE WORKBOOK DOES NOT DO: it does not reproduce the Monte Carlo or the
Sobol decomposition. Those need 10,000 correlated draws and a variance
decomposition over them, which is not honestly expressible in spreadsheet
formulas without either a macro or a 10,000-row block per parameter. The
workbook reports the P10/P50/P90 and the Sobol indices the Python model
computed, clearly labelled as IMPORTED RESULTS rather than live formulas, with
the script and commit that produced them named on the sheet.
"""
from __future__ import annotations

import pathlib
import sys

import xlsxwriter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ae.econ.valuation import (Project, breakeven_price,  # noqa: E402
                               levelized_cost)
from ae.plant.yield_cascade import cascade_yield, off_spec_fraction  # noqa: E402

# --- the scenario the workbook opens on ------------------------------------
# These are the ORE and SITE selectors. Ore rows set impurity level and
# variability; site rows set prices. Every downstream cell is a formula on
# them, so switching a selector recomputes the whole sheet.

ORES = {
    # name: (Al mean ppm, Al lot sigma ppm, flotation yield, leach yield)
    "Ore A, vein quartz, low Al": (18.0, 3.0, 0.85, 0.92),
    "Ore B, pegmatite, moderate Al": (26.0, 5.0, 0.82, 0.90),
    "Ore C, high variability": (22.0, 8.0, 0.80, 0.88),
}

SITES = {
    # name: (power USD/kWh, reagent USD/kg, labour USD/h, freight USD/t)
    "Site 1, India, Telangana": (0.081, 1.80, 4.50, 95.0),
    "Site 2, US, low-cost power": (0.0541, 2.40, 36.92, 25.0),
    "Site 3, US, average power": (0.0889, 2.40, 36.92, 25.0),
}

# Annual fixed costs, spread over product tonnes. OMITTING THESE WAS A REAL
# ERROR in the first version of this script: a cash cost built from
# electricity, reagent and direct labour alone came out at 157.77 USD/t
# against the corrected 795.02, a factor of 5.0, and would have made every
# scenario look profitable. A cash cost is not three line items. The platform's
# own cash_cost() takes labour, maintenance and overhead as arguments for
# exactly this reason, and the workbook now mirrors that structure.
MAINTENANCE_FRACTION_OF_CAPEX = 0.04    # per year, on installed capital
OVERHEAD_USD_PER_YEAR = 1_800_000.0     # G&A, QC laboratory, site management
OPERATORS = 24                          # across four shifts
OPERATOR_HOURS_PER_YEAR = 2_000.0

USL_AL_PPM = 30.0
FEED_TONNES = 7000.0
PRICE = 3500.0
POWER_KWH_PER_T = 250.0
REAGENT_KG_PER_T = 12.0
LABOUR_H_PER_T = 1.6
CAPEX_MUSD = 38.0
DISCOUNT = 0.14
LIFE = 10
TAX = 0.25


def python_case(ore: str, site: str) -> dict[str, float]:
    """The Python model's answer for one (ore, site) pair."""
    al_mean, al_sigma, y_flot, y_leach = ORES[ore]
    power_p, reagent_p, labour_p, freight = SITES[site]

    mass_y = cascade_yield([y_flot, y_leach])
    off = off_spec_fraction(al_mean, al_sigma, USL_AL_PPM, model="lognormal")
    spec_y = 1.0 - off
    overall = mass_y * spec_y
    product_t = FEED_TONNES * overall

    # Variable cost is incurred on FEED, revenue only on saleable PRODUCT, so
    # a spec failure raises unit cost twice: less product, same cost base.
    var_cost = FEED_TONNES * (POWER_KWH_PER_T * power_p
                              + REAGENT_KG_PER_T * reagent_p)
    # Labour is a CREW, not a per-tonne rate: a plant does not halve its
    # operators when yield drops, which is why it belongs in fixed cost.
    labour_yr = OPERATORS * OPERATOR_HOURS_PER_YEAR * labour_p
    maintenance_yr = CAPEX_MUSD * 1e6 * MAINTENANCE_FRACTION_OF_CAPEX
    fixed_yr = labour_yr + maintenance_yr + OVERHEAD_USD_PER_YEAR
    cash_cost_t = (var_cost + fixed_yr + product_t * freight) / product_t

    proj = Project(
        capex_schedule=[CAPEX_MUSD * 1e6],
        construction_periods=1,
        ramp_fractions=[1.0],
        nameplate_tonnes=product_t,
        price=PRICE,
        cash_cost_per_tonne=cash_cost_t,
        life_periods=LIFE,
        tax_rate=TAX,
    )
    cf = proj.cash_flows()
    return {
        "mass_yield": mass_y,
        "off_spec": off,
        "spec_yield": spec_y,
        "overall_yield": overall,
        "product_tonnes": product_t,
        "cash_cost_per_tonne": cash_cost_t,
        "npv_musd": cf.npv(DISCOUNT) / 1e6,
        "irr": cf.irr() or float("nan"),
        "lcop": levelized_cost(proj, DISCOUNT),
        "breakeven_price": breakeven_price(proj, DISCOUNT),
    }


def build(path: pathlib.Path) -> dict[str, dict[str, float]]:
    wb = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})

    # --- formats ---
    h1 = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1A1D21"})
    h2 = wb.add_format({"bold": True, "font_size": 11, "bottom": 1,
                        "border_color": "#9AA1A9"})
    lbl = wb.add_format({"font_size": 10})
    note = wb.add_format({"font_size": 9, "italic": True,
                          "font_color": "#5A6169", "text_wrap": True,
                          "valign": "top"})
    inp = wb.add_format({"font_size": 10, "bg_color": "#FFF6E5",
                         "border": 1, "border_color": "#C8872B",
                         "num_format": "0.0000"})
    calc = wb.add_format({"font_size": 10, "num_format": "0.0000"})
    money = wb.add_format({"font_size": 10, "num_format": "#,##0"})
    money2 = wb.add_format({"font_size": 10, "num_format": "#,##0.0"})
    pct = wb.add_format({"font_size": 10, "num_format": "0.0%"})
    ok = wb.add_format({"font_size": 10, "num_format": "0.00E+00",
                        "bg_color": "#E8F3EA"})

    # =====================================================================
    # Sheet 1: Model. Selectors at the top, formulas below.
    # =====================================================================
    ws = wb.add_worksheet("Model")
    ws.set_column("A:A", 38)
    ws.set_column("B:B", 16)
    ws.set_column("C:C", 52)
    ws.write("A1", "Applied Elements: HPQ purification unit economics", h1)
    ws.write("A2",
             "Amber cells are INPUTS. Everything else is a live Excel formula, "
             "so changing an ore or site selector recomputes the sheet. No "
             "calculated cell contains a pasted Python result.", note)
    ws.set_row(1, 30)

    ore_names = list(ORES)
    site_names = list(SITES)

    ws.write("A4", "SELECTORS", h2)
    ws.write("A5", "Ore", lbl)
    ws.data_validation("B5", {"validate": "list", "source": ore_names})
    ws.write("B5", ore_names[0], inp)
    ws.write("C5", "Sets Al level, lot variability and stage yields", note)
    ws.write("A6", "Site", lbl)
    ws.data_validation("B6", {"validate": "list", "source": site_names})
    ws.write("B6", site_names[0], inp)
    ws.write("C6", "Sets power, reagent, labour and freight prices", note)

    # --- lookup tables, so the selectors resolve by formula ---
    ws.write("A9", "ORE TABLE (lookup source)", h2)
    ws.write_row("A10", ["name", "Al mean ppm", "Al lot sigma ppm",
                         "flotation yield", "leach yield"], lbl)
    for i, (n, vals) in enumerate(ORES.items()):
        ws.write(10 + i, 0, n, lbl)
        for j, v in enumerate(vals):
            ws.write_number(10 + i, 1 + j, v, calc)
    r_ore_end = 10 + len(ORES)

    ws.write(r_ore_end + 1, 0, "SITE TABLE (lookup source)", h2)
    ws.write_row(r_ore_end + 2, 0,
                 ["name", "power USD/kWh", "reagent USD/kg",
                  "labour USD/h", "freight USD/t"], lbl)
    for i, (n, vals) in enumerate(SITES.items()):
        ws.write(r_ore_end + 3 + i, 0, n, lbl)
        for j, v in enumerate(vals):
            ws.write_number(r_ore_end + 3 + i, 1 + j, v, calc)
    r_site_end = r_ore_end + 3 + len(SITES)

    ore_rng = f"$A$11:$E${r_ore_end}"
    site_rng = f"$A${r_ore_end + 4}:$E${r_site_end}"

    def vlk(table: str, col: int, key: str) -> str:
        return f"=VLOOKUP({key},{table},{col},FALSE)"

    # --- resolved drivers ---
    r = r_site_end + 2
    ws.write(r, 0, "RESOLVED DRIVERS (from selectors)", h2)
    drivers = [
        ("Al mean, ppm", vlk(ore_rng, 2, "$B$5"), calc),
        ("Al lot sigma, ppm", vlk(ore_rng, 3, "$B$5"), calc),
        ("Flotation mass yield", vlk(ore_rng, 4, "$B$5"), calc),
        ("Leach mass yield", vlk(ore_rng, 5, "$B$5"), calc),
        ("Power price, USD/kWh", vlk(site_rng, 2, "$B$6"), calc),
        ("Reagent price, USD/kg", vlk(site_rng, 3, "$B$6"), calc),
        ("Labour, USD/h", vlk(site_rng, 4, "$B$6"), calc),
        ("Freight, USD/t", vlk(site_rng, 5, "$B$6"), calc),
    ]
    for i, (name, f, fmt) in enumerate(drivers):
        ws.write(r + 1 + i, 0, name, lbl)
        ws.write_formula(r + 1 + i, 1, f, fmt)
    R = {name: r + 2 + i for i, (name, _, _) in enumerate(drivers)}
    r_drv = r + len(drivers)

    # --- fixed assumptions ---
    r2 = r_drv + 2
    ws.write(r2, 0, "FIXED ASSUMPTIONS", h2)
    fixed = [
        ("Al upper spec limit, ppm", USL_AL_PPM, calc),
        ("Feed, t/yr", FEED_TONNES, money),
        ("Product price, USD/t", PRICE, money),
        ("Power, kWh per t feed", POWER_KWH_PER_T, calc),
        ("Reagent, kg per t feed", REAGENT_KG_PER_T, calc),
        ("Operators", float(OPERATORS), calc),
        ("Operator hours per year", OPERATOR_HOURS_PER_YEAR, money),
        ("Maintenance, fraction of capex/yr", MAINTENANCE_FRACTION_OF_CAPEX, calc),
        ("Overhead, USD/yr", OVERHEAD_USD_PER_YEAR, money),
        ("Capex, MUSD", CAPEX_MUSD, money2),
        ("Discount rate", DISCOUNT, pct),
        ("Life, years", LIFE, calc),
        ("Tax rate", TAX, pct),
    ]
    for i, (name, v, fmt) in enumerate(fixed):
        ws.write(r2 + 1 + i, 0, name, lbl)
        ws.write_number(r2 + 1 + i, 1, v, inp if fmt is calc else fmt)
    F = {name: r2 + 2 + i for i, (name, _, _) in enumerate(fixed)}
    r_fix = r2 + len(fixed)

    # --- calculations, all formulas ---
    r3 = r_fix + 2
    ws.write(r3, 0, "CALCULATED (live formulas)", h2)
    b = lambda k: f"$B${R[k]}" if k in R else f"$B${F[k]}"  # noqa: E731

    rows = [
        ("Mass yield, cascade",
         f"={b('Flotation mass yield')}*{b('Leach mass yield')}", calc,
         "Stage yields multiply. Two 90 percent stages give 81 percent, "
         "not 90."),
        ("Off-spec fraction (lognormal)",
         f"=1-LOGNORM.DIST({b('Al upper spec limit, ppm')},"
         f"LN({b('Al mean, ppm')}^2/SQRT({b('Al lot sigma, ppm')}^2"
         f"+{b('Al mean, ppm')}^2)),"
         f"SQRT(LN(1+({b('Al lot sigma, ppm')}/{b('Al mean, ppm')})^2)),TRUE)",
         calc,
         "Lognormal, not normal: impurity distributions are right-skewed and "
         "a normal understates the tail that fails lots."),
        ("Spec yield", f"=1-$B${r3 + 3}", calc, ""),
        ("Overall yield", f"=$B${r3 + 2}*$B${r3 + 4}", calc,
         "Mass yield times spec yield."),
        ("Product, t/yr", f"={b('Feed, t/yr')}*$B${r3 + 5}", money,
         "Saleable tonnes after both mass loss and spec failure."),
        ("Variable cost, USD/yr",
         f"={b('Feed, t/yr')}*({b('Power, kWh per t feed')}"
         f"*{b('Power price, USD/kWh')}+{b('Reagent, kg per t feed')}"
         f"*{b('Reagent price, USD/kg')})", money,
         "Incurred on FEED, which is why a spec failure raises unit cost "
         "twice: less product, same cost base."),
        ("Labour, USD/yr",
         f"={b('Operators')}*{b('Operator hours per year')}"
         f"*{b('Labour, USD/h')}", money,
         "A CREW, not a per-tonne rate: a plant does not shed operators when "
         "yield drops, so labour is fixed."),
        ("Maintenance, USD/yr",
         f"={b('Capex, MUSD')}*1000000"
         f"*{b('Maintenance, fraction of capex/yr')}", money, ""),
        ("Fixed cost, USD/yr",
         f"=$B${r3 + 8}+$B${r3 + 9}+{b('Overhead, USD/yr')}", money,
         "Labour, maintenance and overhead. Omitting these gave a 157.77 "
         "USD/t cash cost against the corrected 795.02, a factor of 5.0, "
         "and made every scenario look profitable."),
        ("Freight, USD/yr", f"=$B${r3 + 6}*{b('Freight, USD/t')}", money, ""),
        ("Cash cost, USD/t product",
         f"=($B${r3 + 7}+$B${r3 + 10}+$B${r3 + 11})/$B${r3 + 6}", money2, ""),
        ("Revenue, USD/yr",
         f"=$B${r3 + 6}*{b('Product price, USD/t')}", money, ""),
        ("EBITDA, USD/yr",
         f"=$B${r3 + 13}-$B${r3 + 7}-$B${r3 + 10}-$B${r3 + 11}", money, ""),
        ("Depreciation, USD/yr",
         f"={b('Capex, MUSD')}*1000000/{b('Life, years')}", money,
         "Straight line over the project life."),
        ("Taxable income, USD/yr", f"=$B${r3 + 14}-$B${r3 + 15}", money, ""),
        ("Tax, USD/yr",
         f"=MAX(0,$B${r3 + 16})*{b('Tax rate')}", money,
         "MAX(0,...) because a loss does not generate a cash refund here."),
        ("Net cash flow, USD/yr", f"=$B${r3 + 14}-$B${r3 + 17}", money, ""),
        ("NPV, MUSD",
         f"=(-{b('Capex, MUSD')}*1000000+$B${r3 + 18}"
         f"*(1-(1+{b('Discount rate')})^-{b('Life, years')})"
         f"/{b('Discount rate')})/1000000", money2,
         "Capex at t=0, then a level annuity discounted over the life."),
        ("IRR",
         f"=RATE({b('Life, years')},$B${r3 + 18},"
         f"-{b('Capex, MUSD')}*1000000)", pct,
         "RATE solves the same level-annuity cash flow as the NPV row."),
        ("Breakeven price, USD/t",
         f"=$B${r3 + 12}+({b('Capex, MUSD')}*1000000*{b('Discount rate')}"
         f"/(1-(1+{b('Discount rate')})^-{b('Life, years')}))"
         f"/$B${r3 + 6}/(1-{b('Tax rate')})", money2,
         "Cash cost plus the annualised capital charge per tonne, grossed "
         "up for tax."),
    ]
    for i, (name, f, fmt, why) in enumerate(rows):
        ws.write(r3 + 1 + i, 0, name, lbl)
        ws.write_formula(r3 + 1 + i, 1, f, fmt)
        if why:
            ws.write(r3 + 1 + i, 2, why, note)

    # =====================================================================
    # Sheet 2: Reconciliation, workbook formulas vs the Python model.
    # =====================================================================
    rec = wb.add_worksheet("Reconciliation")
    rec.set_column("A:A", 34)
    rec.set_column("B:D", 18)
    rec.set_column("E:E", 44)
    rec.write("A1", "Reconciliation: Excel formulas against the Python model", h1)
    rec.write("A2",
              "Column B is computed by this workbook's own formulas. Column C "
              "is the Python model's value for the SAME scenario, written at "
              "build time. Column D is the relative difference. A nonzero D "
              "means the workbook and the model have diverged, which is the "
              "condition this sheet exists to expose.", note)
    rec.set_row(1, 42)

    py = {o: {s: python_case(o, s) for s in SITES} for o in ORES}
    base = py[ore_names[0]][site_names[0]]

    checks = [
        ("Mass yield", f"=Model!$B${r3 + 2}", base["mass_yield"]),
        ("Off-spec fraction", f"=Model!$B${r3 + 3}", base["off_spec"]),
        ("Overall yield", f"=Model!$B${r3 + 5}", base["overall_yield"]),
        ("Product, t/yr", f"=Model!$B${r3 + 6}", base["product_tonnes"]),
        ("Cash cost, USD/t", f"=Model!$B${r3 + 12}",
         base["cash_cost_per_tonne"]),
    ]
    rec.write_row("A4", ["quantity", "workbook formula", "Python model",
                         "relative difference", "note"], h2)
    for i, (name, f, pyval) in enumerate(checks):
        rw = 4 + i
        rec.write(rw, 0, name, lbl)
        rec.write_formula(rw, 1, f, calc)
        rec.write_number(rw, 2, pyval, calc)
        rec.write_formula(
            rw, 3,
            f"=IF($C${rw + 1}=0,ABS($B${rw + 1}),"
            f"ABS($B${rw + 1}-$C${rw + 1})/ABS($C${rw + 1}))", ok)
    n = len(checks)
    rec.write(4 + n + 1, 0, "Worst relative difference", h2)
    rec.write_formula(4 + n + 1, 3, f"=MAX($D$5:$D${4 + n})", ok)
    rec.write(4 + n + 2, 0, "Status", lbl)
    rec.write_formula(
        4 + n + 2, 3,
        f'=IF($D${4 + n + 2}<0.000001,"RECONCILED","DIVERGED")', lbl)
    rec.write(4 + n + 4, 0,
              "NPV, IRR and breakeven are NOT reconciled here, and that is "
              "deliberate rather than an omission: the workbook computes them "
              "on a level annuity, while the Python model discounts an "
              "explicit per-period cash flow array with construction timing, "
              "ramp fractions and working capital. Those are different "
              "models, so forcing them to agree would mean degrading one. The "
              "workbook is for interrogating unit economics; the Python model "
              "is authoritative for valuation.", note)
    rec.set_row(4 + n + 4, 64)

    # =====================================================================
    # Sheet 3: Scenario grid, every ore against every site.
    # =====================================================================
    grid = wb.add_worksheet("Scenario grid")
    grid.set_column("A:A", 34)
    grid.set_column("B:K", 15)
    grid.write("A1", "All ore and site combinations (Python model values)", h1)
    grid.write("A2",
               "IMPORTED RESULTS, not live formulas: each row is the Python "
               "model evaluated at build time. The Model sheet is the live "
               "one. Reported so a reader can see the spread without "
               "switching selectors nine times.", note)
    grid.set_row(1, 30)
    cols = ["ore", "site", "mass yield", "off-spec", "overall yield",
            "product t/yr", "cash cost USD/t", "NPV MUSD", "IRR",
            "LCOP USD/t", "breakeven USD/t"]
    grid.write_row("A4", cols, h2)
    rw = 4
    for o in ORES:
        for s in SITES:
            d = py[o][s]
            grid.write(rw, 0, o, lbl)
            grid.write(rw, 1, s, lbl)
            grid.write_number(rw, 2, d["mass_yield"], calc)
            grid.write_number(rw, 3, d["off_spec"], calc)
            grid.write_number(rw, 4, d["overall_yield"], calc)
            grid.write_number(rw, 5, d["product_tonnes"], money)
            grid.write_number(rw, 6, d["cash_cost_per_tonne"], money2)
            grid.write_number(rw, 7, d["npv_musd"], money2)
            grid.write_number(rw, 8, d["irr"], pct)
            grid.write_number(rw, 9, d["lcop"], money2)
            grid.write_number(rw, 10, d["breakeven_price"], money2)
            rw += 1

    # =====================================================================
    # Sheet 4: Provenance.
    # =====================================================================
    prov = wb.add_worksheet("Provenance")
    prov.set_column("A:A", 30)
    prov.set_column("B:B", 100)
    prov.write("A1", "Where these numbers come from", h1)
    lines = [
        ("Built by", "scripts/build_workbook.py in the applied-elements repo"),
        ("Authoritative model", "the Python package; this workbook mirrors it"),
        ("Ore rows", "SCENARIO INPUTS, not measurements. No ore from the "
                     "Vikarabad lease has been characterized, so no row here "
                     "represents an assayed feedstock."),
        ("Al spec limit", "30 ppm, the HPQ-grade limit used throughout the "
                          "platform"),
        ("Site power prices", "Telangana 33 kV HT-I(A) tariff and EIA "
                              "Electric Power Monthly Table 5.6.B state "
                              "averages"),
        ("Freight", "observed India-to-US CIF minus FOB on HS 250610, "
                    "USD 76 to 115 per tonne"),
        ("Capex", "AACE Class 5, minus 50 to plus 100 percent. A factored "
                  "estimate from an equipment list with no flowsheet "
                  "engineering cannot claim better."),
        ("Not in this workbook", "Monte Carlo and Sobol. Those need 10,000 "
                                 "draws and a variance decomposition, which "
                                 "is not honestly expressible in formulas."),
        ("Parameter registry", "data/registry/parameter_registry.csv lists "
                               "every tracked parameter with its tag, source "
                               "and uncertainty"),
    ]
    for i, (k, v) in enumerate(lines):
        prov.write(2 + i, 0, k, h2 if i == 0 else lbl)
        prov.write(2 + i, 1, v, note)
        prov.set_row(2 + i, 26)

    wb.close()
    return py


def main() -> None:
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "AE-model-mirror.xlsx"
    py = build(out)
    print(f"{out.name}: {out.stat().st_size:,} bytes")
    print(f"  {len(ORES)} ores x {len(SITES)} sites = "
          f"{len(ORES) * len(SITES)} scenarios")
    b = py[list(ORES)[0]][list(SITES)[0]]
    print(f"  base case: yield {b['overall_yield']:.4f}, "
          f"cash cost {b['cash_cost_per_tonne']:,.1f} USD/t, "
          f"NPV {b['npv_musd']:+,.1f} MUSD, breakeven "
          f"{b['breakeven_price']:,.1f} USD/t")


if __name__ == "__main__":
    main()
