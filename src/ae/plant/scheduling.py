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

1. Waiting time is driven by VARIABILITY as much as by load, but only through
   the SUM :math:`c_a^2 + c_s^2`. Halving :math:`c_s^2` therefore cuts the queue
   by a factor :math:`(c_a^2 + c_s^2/2) / (c_a^2 + c_s^2)`, which depends on the
   arrival variability: from 1.0 to 0.5 at Poisson arrivals
   (:math:`c_a = 1`, the default in :meth:`PlantSchedule.run`) gives a 25
   percent reduction, not a halving. Only with perfectly regular arrivals
   (:math:`c_a = 0`) does halving :math:`c_s^2` halve the queue. Service-side
   variability reduction is worth less the burstier the arrivals, which is why
   scheduling discipline and process consistency have to be attacked together.
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
    "PlantSchedule",
    "ScheduleResult",
    "Station",
    "allen_cunneen_waiting_time",
    "mm1_waiting_time",
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
    #: Time in system for EVERY lot that departed inside the measurement
    #: window, shipped or scrapped. ``cycle_times`` covers shipped lots only,
    #: because that is the number a customer experiences, but Little's law is
    #: an identity about departures of either disposition: a scrapped lot
    #: occupied the line and contributed to the WIP integral exactly as a
    #: shipped one did. Omitting scrap understated lambda W by 41.5 percent at
    #: a 30 percent scrap rate.
    residence_times: np.ndarray
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

    #: Start of the measurement window, so the WIP integral uses the same
    #: interval as every other reported statistic.
    warmup_hours: float = 0.0

    @property
    def mean_wip(self) -> float:
        """Time-averaged work in process over the MEASUREMENT window.

        The integral starts at ``warmup_hours``, matching the basis of the
        counts and the busy fractions. Integrating the full timeline against a
        trimmed horizon was an accounting mismatch that made
        :meth:`littles_law_check` report a spurious residual.
        """
        if len(self.wip_timeline) < 2:
            return float("nan")
        t = np.array([x[0] for x in self.wip_timeline], dtype=float)
        n = np.array([x[1] for x in self.wip_timeline], dtype=float)
        w = self.warmup_hours
        end = t[-1]
        if end <= w:
            return float("nan")
        # Clip each segment to [w, end] and weight by the surviving duration.
        lo = np.clip(t[:-1], w, end)
        hi = np.clip(t[1:], w, end)
        dt = hi - lo
        total = float(np.sum(dt))
        return float(np.sum(n[:-1] * dt) / total) if total > 0 else float("nan")

    @property
    def departure_rate_per_hour(self) -> float:
        """Departures per hour of EITHER disposition, over the measured window.

        Distinct from :attr:`throughput_per_hour`, which counts shipped lots
        only and is the commercial number. Little's law needs this one.

        ``horizon_hours`` on this object is ALREADY the post-warmup window
        (``run`` stores ``eff``, not the wall-clock horizon), so subtracting
        ``warmup_hours`` again here overstated the rate: my first version did
        exactly that and reported 0.77636 departures per hour on a line offered
        0.7 per hour, which is impossible in steady state and is what caught
        the error.
        """
        if self.horizon_hours <= 0:
            return 0.0
        return (self.lots_shipped + self.lots_scrapped) / self.horizon_hours

    @property
    def mean_residence_time(self) -> float:
        """Mean time in system across all departures, shipped or scrapped."""
        return (float(np.mean(self.residence_times))
                if self.residence_times.size else float("nan"))

    def littles_law_check(self) -> dict[str, float]:
        """Verify L = lambda W against the simulation's own numbers.

        Little's law is an identity, not a model, so a discrepancy means the
        simulation's accounting is wrong. Returned rather than asserted so a
        caller can see the residual.

        The identity is about DEPARTURES, not sales. An earlier version used
        ``lots_shipped / horizon`` against the mean SHIPPED cycle time, so a
        line that scraps lots understated both factors: measured at a 30
        percent scrap rate (single station, arrival 0.7/h, horizon 20000 h,
        warmup 2000 h, seed 3), L_measured was 2.24732 against lambda_W
        1.58809, a residual of 0.41511, while the WIP integral itself was
        correct (2.3333 analytic for rho = 0.7). With no scrap the two bases
        coincide, which is why every existing check passed.
        """
        lam = self.departure_rate_per_hour
        w = self.mean_residence_time
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
            Transient-removal window. EVERY reported statistic is measured over
            the post-warmup interval only, on one consistent basis: lots
            completing before ``warmup_hours`` are excluded from the cycle-time
            distribution AND from the shipped, scrapped and reworked counts;
            station busy and failed fractions accumulate only after it; and the
            WIP integral is taken over the same interval. Mixing a trimmed
            denominator with untrimmed counts inflates throughput and makes
            :meth:`ScheduleResult.littles_law_check` report a residual that is
            an accounting artefact rather than a simulation error.
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
        residences: list[float] = []
        queues: dict[str, list[float]] = {s.name: [] for s in self.stations}
        busy_time = {s.name: 0.0 for s in self.stations}
        failed_time = {s.name: 0.0 for s in self.stations}
        counters = {"started": 0, "shipped": 0, "scrapped": 0, "reworked": 0}

        def post_warmup(t0: float, t1: float) -> float:
            """Overlap of the interval [t0, t1] with the measurement window.

            Busy and failed time are apportioned rather than counted whole, so a
            service or repair straddling the warmup boundary contributes only
            its post-warmup part. Counting it whole would attribute pre-warmup
            work to the measured window and push utilisation above the true
            value.
            """
            return max(0.0, min(t1, horizon_hours) - max(t0, warmup_hours))
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
                failed_time[s.name] += post_warmup(t0, env.now)
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
                    if env.now >= warmup_hours:
                        queues[s.name].append(env.now - t_q)
                    # Wait out any active failure before starting service.
                    while broken[s.name]:
                        yield env.timeout(0.25)
                    t_svc = env.now
                    yield env.timeout(s.draw_service(rng))
                    busy_time[s.name] += post_warmup(t_svc, env.now)
                if s.qc_hold_hours > 0:
                    yield env.timeout(s.qc_hold_hours)
                if s.qc_fail_probability > 0 and rng.random() < s.qc_fail_probability:
                    if s.qc_disposition == "scrap":
                        if env.now >= warmup_hours:
                            counters["scrapped"] += 1
                            residences.append(env.now - t_start)
                        touch_wip(-1)
                        return
                    if not reworked_once:
                        reworked_once = True
                        if env.now >= warmup_hours:
                            counters["reworked"] += 1
                        with resources[s.name].request() as req2:
                            yield req2
                            t_svc = env.now
                            yield env.timeout(s.draw_service(rng))
                            busy_time[s.name] += post_warmup(t_svc, env.now)
                    else:
                        if env.now >= warmup_hours:
                            counters["scrapped"] += 1
                            residences.append(env.now - t_start)
                        touch_wip(-1)
                        return
            touch_wip(-1)
            if env.now >= warmup_hours:
                counters["shipped"] += 1
                cycles.append(env.now - t_start)
                residences.append(env.now - t_start)

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
                if env.now >= warmup_hours:
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
            residence_times=np.asarray(residences, dtype=float),
            queue_times={k: np.asarray(v, dtype=float) for k, v in queues.items()},
            station_busy_fraction={
                k: v / (eff * next(s.capacity for s in self.stations if s.name == k))
                for k, v in busy_time.items()
            },
            station_failed_fraction={k: v / eff for k, v in failed_time.items()},
            horizon_hours=eff,
            wip_timeline=timeline,
            warmup_hours=warmup_hours,
        )
