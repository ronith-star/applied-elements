r"""Discrete-event simulation of batch processing with queues, QC holds and failures.

Why this exists alongside :mod:`ae.plant.capacity`: OEE gives a steady-state
average and cannot represent queueing. A plant whose average utilisation is 70
percent can still hold lots for days if arrivals are bursty and a downstream
unit is a single batch server, because waiting time diverges as utilisation
approaches unity even when the unit is nominally under-loaded.

The queueing result that motivates the simulation
------------------------------------------------
For a single server with utilisation :math:`\rho = \lambda / \mu` the expected
waiting time in an M/M/1 queue is

.. math::
   W_q = \frac{\rho}{\mu (1 - \rho)}

which diverges as :math:`\rho \to 1`. The Allen-Cunneen approximation
generalizes this to arbitrary arrival and service variability,

.. math::
   W_q \approx \frac{\rho}{1 - \rho} \cdot
               \frac{c_a^2 + c_s^2}{2} \cdot \frac{1}{\mu}

with :math:`c_a` and :math:`c_s` the coefficients of variation of interarrival
and service times (dimensionless, typically 0 to 2). Two consequences the model
must reproduce:

1. Waiting time is driven by VARIABILITY as much as by load. Halving
   :math:`c_s^2` halves the queue at fixed utilisation.
2. A batch process with a long fixed cycle (a calcination kiln, an acid leach
   autoclave) has low :math:`c_s`, which is protective, but its long service
   time raises :math:`\rho` for the same arrival rate.

:func:`mm1_waiting_time` and :func:`allen_cunneen_waiting_time` provide the
analytic forms, and the simulation is validated against them, which is the only
honest way to check a discrete-event model: an analytic limit it must reproduce.

Symbols
-------
:math:`\lambda` arrival rate (lots/h, positive), :math:`\mu` service rate
(lots/h, positive), :math:`\rho` utilisation (dimensionless, 0 to 1 strictly),
:math:`W_q` waiting time in queue (h), :math:`c_a, c_s` coefficients of
variation (dimensionless, non-negative).

LIMITATIONS
-----------
* Lot sizes are uniform. A plant with mixed lot sizes has a different queueing
  discipline and this model will understate delay.
* QC hold is modelled as a delay plus a pass/fail draw with a fixed failure
  probability. In reality failure probability is correlated with the process
  state that produced the lot, so failures cluster, which this understates.
* Rework is modelled as a single retry at most. Multi-pass rework loops exist in
  purification (a lot re-leached twice) and are not represented.
* No operator or reagent constraint. A unit is available whenever it is not
  busy or failed, which overstates availability in a plant that is
  labour-limited on nights or weekends.
* Failures are exponential (memoryless). Real equipment has wear-out, so this
  understates late-life downtime clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import simpy

__all__ = [
    "Station",
    "PlantSchedule",
    "ScheduleResult",
    "mm1_waiting_time",
    "allen_cunneen_waiting_time",
]


def mm1_waiting_time(arrival_rate: float, service_rate: float) -> float:
    """Expected queue waiting time for M/M/1, in hours.

    Examples
    --------
    At lambda = 0.8/h and mu = 1.0/h, rho = 0.8 and Wq = 0.8/(1.0 x 0.2) = 4.0 h.

    >>> round(mm1_waiting_time(0.8, 1.0), 10)
    4.0
    """
    if arrival_rate <= 0 or service_rate <= 0:
        raise ValueError("arrival and service rates must be positive")
    rho = arrival_rate / service_rate
    if rho >= 1.0:
        raise ValueError(
            f"utilisation is {rho:.4g} >= 1: the queue is unstable and has no "
            f"steady-state waiting time. Add capacity rather than simulating longer."
        )
    return rho / (service_rate * (1.0 - rho))


def allen_cunneen_waiting_time(
    arrival_rate: float, service_rate: float, cv_arrival: float, cv_service: float
) -> float:
    """Allen-Cunneen approximation for a G/G/1 queue, in hours.

    Reduces to M/M/1 when both coefficients of variation equal 1.

    Examples
    --------
    >>> round(allen_cunneen_waiting_time(0.8, 1.0, 1.0, 1.0), 6)
    4.0
    >>> round(allen_cunneen_waiting_time(0.8, 1.0, 1.0, 0.0), 6)
    2.0
    """
    if arrival_rate <= 0 or service_rate <= 0:
        raise ValueError("arrival and service rates must be positive")
    if cv_arrival < 0 or cv_service < 0:
        raise ValueError("coefficients of variation must be non-negative")
    rho = arrival_rate / service_rate
    if rho >= 1.0:
        raise ValueError(f"utilisation is {rho:.4g} >= 1: no steady state exists")
    return (rho / (1.0 - rho)) * ((cv_arrival**2 + cv_service**2) / 2.0) / service_rate


@dataclass
class Station:
    """One batch processing station.

    Parameters
    ----------
    name
        Label.
    service_hours
        Mean processing time per lot.
    capacity
        Number of parallel units (1 for a single kiln).
    cv_service
        Coefficient of variation of service time. 0.0 is a deterministic batch
        cycle, 1.0 is exponential.
    failure_mtbf_hours
        Mean time between failures. None means no failures.
    repair_hours
        Mean repair time.
    qc_hold_hours
        Fixed delay for assay turnaround after processing, 0.0 for none. This is
        the parameter that most often dominates cycle time in a purification
        plant, because a lot cannot ship until its assay returns.
    qc_fail_probability
        Probability a lot fails QC and is either reworked or scrapped.
    qc_disposition
        What happens on QC failure.
    """

    name: str
    service_hours: float
    capacity: int = 1
    cv_service: float = 0.0
    failure_mtbf_hours: float | None = None
    repair_hours: float = 0.0
    qc_hold_hours: float = 0.0
    qc_fail_probability: float = 0.0
    qc_disposition: Literal["rework", "scrap"] = "rework"

    def __post_init__(self) -> None:
        if self.service_hours <= 0:
            raise ValueError(f"{self.name}: service_hours must be positive")
        if self.capacity < 1:
            raise ValueError(f"{self.name}: capacity must be at least 1")
        if self.cv_service < 0:
            raise ValueError(f"{self.name}: cv_service must be non-negative")
        if not 0.0 <= self.qc_fail_probability <= 1.0:
            raise ValueError(f"{self.name}: qc_fail_probability must be a fraction")
        if self.failure_mtbf_hours is not None:
            if self.failure_mtbf_hours <= 0:
                raise ValueError(f"{self.name}: failure_mtbf_hours must be positive")
            if self.repair_hours <= 0:
                raise ValueError(
                    f"{self.name}: a station with an MTBF must have a positive repair time"
                )

    @property
    def service_rate(self) -> float:
        """Lots per hour per unit."""
        return 1.0 / self.service_hours

    @property
    def intrinsic_availability(self) -> float:
        """MTBF / (MTBF + MTTR), the availability failures alone imply."""
        if self.failure_mtbf_hours is None:
            return 1.0
        return self.failure_mtbf_hours / (self.failure_mtbf_hours + self.repair_hours)

    def draw_service(self, rng: np.random.Generator) -> float:
        """Sample a service time with the requested coefficient of variation.

        Uses a gamma distribution, which spans deterministic (cv 0) through
        exponential (cv 1) to more variable, and is strictly positive. A normal
        would admit negative service times.
        """
        if self.cv_service == 0.0:
            return self.service_hours
        shape = 1.0 / self.cv_service**2
        return float(rng.gamma(shape, self.service_hours / shape))


@dataclass
class ScheduleResult:
    """Outcome of a simulation run."""

    lots_started: int
    lots_shipped: int
    lots_scrapped: int
    lots_reworked: int
    cycle_times: np.ndarray
    queue_times: dict[str, np.ndarray]
    station_busy_fraction: dict[str, float]
    station_failed_fraction: dict[str, float]
    horizon_hours: float
    wip_timeline: list[tuple[float, int]] = field(default_factory=list)

    @property
    def throughput_per_hour(self) -> float:
        return self.lots_shipped / self.horizon_hours if self.horizon_hours else 0.0

    @property
    def mean_cycle_time(self) -> float:
        return float(np.mean(self.cycle_times)) if self.cycle_times.size else float("nan")

    def cycle_time_percentile(self, p: float) -> float:
        """A percentile of cycle time. p95 is the number a customer experiences."""
        if not 0 < p < 100:
            raise ValueError("percentile must lie in (0, 100)")
        if self.cycle_times.size == 0:
            return float("nan")
        return float(np.percentile(self.cycle_times, p))

    @property
    def mean_wip(self) -> float:
        """Time-averaged work in process, from the recorded timeline."""
        if len(self.wip_timeline) < 2:
            return float("nan")
        t = np.array([x[0] for x in self.wip_timeline])
        n = np.array([x[1] for x in self.wip_timeline])
        dt = np.diff(t)
        return float(np.sum(n[:-1] * dt) / np.sum(dt)) if np.sum(dt) > 0 else float("nan")

    def littles_law_check(self) -> dict[str, float]:
        """Verify L = lambda W against the simulation's own numbers.

        Little's law is an identity, not a model, so a discrepancy means the
        simulation's accounting is wrong. Returned rather than asserted so a
        caller can see the residual.
        """
        lam = self.throughput_per_hour
        w = self.mean_cycle_time
        return {"L_measured": self.mean_wip, "lambda_W": lam * w,
                "relative_error": (abs(self.mean_wip - lam * w) / (lam * w))
                if lam * w > 0 else float("nan")}


class PlantSchedule:
    """A serial line of batch stations, simulated with SimPy."""

    def __init__(self, stations: list[Station]) -> None:
        if not stations:
            raise ValueError("a schedule needs at least one station")
        names = [s.name for s in stations]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate station names: {names}")
        self.stations = stations

    def bottleneck_station(self) -> Station:
        """Station with the lowest capacity-adjusted service rate."""
        return min(self.stations, key=lambda s: s.service_rate * s.capacity
                   * s.intrinsic_availability)

    def max_arrival_rate(self) -> float:
        """Arrival rate at which the bottleneck saturates, lots/h."""
        b = self.bottleneck_station()
        return b.service_rate * b.capacity * b.intrinsic_availability

    def run(
        self,
        arrival_rate: float,
        horizon_hours: float,
        seed: int = 0,
        cv_arrival: float = 1.0,
        warmup_hours: float = 0.0,
    ) -> ScheduleResult:
        """Simulate lot arrivals through the line.

        Parameters
        ----------
        arrival_rate
            Mean lot arrivals per hour.
        horizon_hours
            Simulated duration.
        cv_arrival
            1.0 gives Poisson arrivals; 0.0 gives a perfectly regular schedule,
            which is what a well-run campaign plan looks like.
        warmup_hours
            Lots completing before this time are excluded from the statistics,
            so a cold start does not bias the cycle-time distribution.
        """
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if horizon_hours <= warmup_hours:
            raise ValueError("horizon must exceed the warmup period")
        rng = np.random.default_rng(seed)
        env = simpy.Environment()
        resources = {s.name: simpy.Resource(env, capacity=s.capacity) for s in self.stations}
        broken = {s.name: False for s in self.stations}

        cycles: list[float] = []
        queues: dict[str, list[float]] = {s.name: [] for s in self.stations}
        busy_time = {s.name: 0.0 for s in self.stations}
        failed_time = {s.name: 0.0 for s in self.stations}
        counters = {"started": 0, "shipped": 0, "scrapped": 0, "reworked": 0}
        wip = {"n": 0}
        timeline: list[tuple[float, int]] = [(0.0, 0)]

        def touch_wip(delta: int) -> None:
            wip["n"] += delta
            timeline.append((env.now, wip["n"]))

        def failure_process(s: Station) -> object:
            while True:
                yield env.timeout(rng.exponential(s.failure_mtbf_hours))
                broken[s.name] = True
                t0 = env.now
                yield env.timeout(rng.exponential(s.repair_hours))
                failed_time[s.name] += env.now - t0
                broken[s.name] = False

        def lot(idx: int) -> object:
            t_start = env.now
            touch_wip(+1)
            reworked_once = False
            for s in self.stations:
                res = resources[s.name]
                t_q = env.now
                with res.request() as req:
                    yield req
                    queues[s.name].append(env.now - t_q)
                    # Wait out any active failure before starting service.
                    while broken[s.name]:
                        yield env.timeout(0.25)
                    t_svc = env.now
                    yield env.timeout(s.draw_service(rng))
                    busy_time[s.name] += env.now - t_svc
                if s.qc_hold_hours > 0:
                    yield env.timeout(s.qc_hold_hours)
                if s.qc_fail_probability > 0 and rng.random() < s.qc_fail_probability:
                    if s.qc_disposition == "scrap":
                        counters["scrapped"] += 1
                        touch_wip(-1)
                        return
                    if not reworked_once:
                        reworked_once = True
                        counters["reworked"] += 1
                        with resources[s.name].request() as req2:
                            yield req2
                            t_svc = env.now
                            yield env.timeout(s.draw_service(rng))
                            busy_time[s.name] += env.now - t_svc
                    else:
                        counters["scrapped"] += 1
                        touch_wip(-1)
                        return
            counters["shipped"] += 1
            touch_wip(-1)
            if env.now >= warmup_hours:
                cycles.append(env.now - t_start)

        def source() -> object:
            i = 0
            while True:
                if cv_arrival == 0.0:
                    gap = 1.0 / arrival_rate
                elif cv_arrival == 1.0:
                    gap = rng.exponential(1.0 / arrival_rate)
                else:
                    shape = 1.0 / cv_arrival**2
                    gap = rng.gamma(shape, (1.0 / arrival_rate) / shape)
                yield env.timeout(gap)
                i += 1
                counters["started"] += 1
                env.process(lot(i))

        for s in self.stations:
            if s.failure_mtbf_hours is not None:
                env.process(failure_process(s))
        env.process(source())
        env.run(until=horizon_hours)
        timeline.append((horizon_hours, wip["n"]))

        eff = max(horizon_hours - warmup_hours, 1e-12)
        return ScheduleResult(
            lots_started=counters["started"],
            lots_shipped=counters["shipped"],
            lots_scrapped=counters["scrapped"],
            lots_reworked=counters["reworked"],
            cycle_times=np.asarray(cycles, dtype=float),
            queue_times={k: np.asarray(v, dtype=float) for k, v in queues.items()},
            station_busy_fraction={
                k: v / (horizon_hours * next(s.capacity for s in self.stations if s.name == k))
                for k, v in busy_time.items()
            },
            station_failed_fraction={k: v / horizon_hours for k, v in failed_time.items()},
            horizon_hours=eff,
            wip_timeline=timeline,
        )
