#!/usr/bin/env python3
"""
Synthetic RTT pathway generator
===============================

Generates a synthetic NHS Referral-to-Treatment waiting list for development and
demonstration purposes.

DATA PROVENANCE
---------------
Every record produced by this tool is fabricated. Names, NHS numbers, dates and
events are generated from statistical distributions and from NHS England's
*publicly published* RTT rules and national statistics. Nothing here is derived
from, copied from, or informed by any NHS Trust's operational systems, and no
real patient data was accessed at any point in the construction of this tool.

RULES IMPLEMENTED (from the published national RTT rules suite)
---------------------------------------------------------------
  R1  A clock starts on the date the provider receives a referral for a
      consultant-led pathway.
  R2  The 18-week operational standard is measured in calendar days: 126 days
      from clock start.
  R3  52-week waits (364 days) are reported separately and are the headline
      national breach measure.
  R4  A clock stops on: first definitive treatment, a decision not to treat,
      the patient declining treatment, or the start of active monitoring.
  R5  A DNA against the FIRST care-professional appointment may nullify the
      clock (a new clock starts on rebooking) where the appointment was clearly
      communicated and nullification is in the patient's clinical interest.
      A DNA against any SUBSEQUENT appointment does not nullify.
  R6  A provider-initiated cancellation never affects the clock.
  R7  A patient-initiated cancellation does not stop or pause the clock.

DELIBERATE SIMPLIFICATIONS — declare these in your README
----------------------------------------------------------
  S1  R5 is applied probabilistically via --dna-nullify-rate rather than by
      modelling the clinical-interest decision. Real nullification is a
      judgement made by a person.
  S2  Bilateral procedures, planned/surveillance pathways and consultant-to-
      consultant referrals are not modelled.
  S3  Active monitoring is modelled as a terminal clock stop. In reality a new
      clock starts if a decision to treat follows.
  S4  Waiting-time distributions are plausible rather than fitted to any
      specific published dataset.

USAGE
-----
    python3 rtt_generator.py --patients 50000 --seed 42 --out ./data
    python3 rtt_generator.py --validate ./data

No third-party dependencies. Python 3.9+.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Iterator

# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #

WEEKS_18 = 126          # R2 — calendar days
WEEKS_52 = 364          # R3 — calendar days

SPECIALTIES = [
    ("100", "General Surgery", 0.14),
    ("101", "Urology", 0.10),
    ("110", "Trauma & Orthopaedics", 0.16),
    ("120", "ENT", 0.11),
    ("130", "Ophthalmology", 0.13),
    ("140", "Oral Surgery", 0.05),
    ("160", "Plastic Surgery", 0.04),
    ("170", "Cardiothoracic Surgery", 0.02),
    ("300", "General Medicine", 0.06),
    ("301", "Gastroenterology", 0.09),
    ("320", "Cardiology", 0.06),
    ("330", "Dermatology", 0.04),
]

PRIORITIES = [("ROUTINE", 0.78), ("URGENT", 0.15), ("TWO_WEEK_WAIT", 0.07)]

REFERRAL_SOURCES = [("E_RS", 0.72), ("PAPER", 0.14), ("CONSULTANT_TO_CONSULTANT", 0.09), ("OTHER", 0.05)]

# ODS-style provider/referrer codes. Format is real; the codes are invented.
REFERRER_CODES = [f"{p}{n:05d}" for p in ("G", "Y") for n in range(10001, 10041)]

EVENT_TYPES = (
    "REFERRAL_RECEIVED",
    "CLOCK_START",
    "APPOINTMENT_BOOKED",
    "APPOINTMENT_ATTENDED",
    "DNA",
    "CANCELLED_BY_PATIENT",
    "CANCELLED_BY_PROVIDER",
    "CLOCK_NULLIFIED",
    "ACTIVE_MONITORING_START",
    "DECISION_TO_TREAT",
    "TREATMENT_STARTED",
    "DECISION_NOT_TO_TREAT",
    "PATIENT_DECLINED_TREATMENT",
    "PATHWAY_CLOSED",
)

# Postcode sectors only — never full postcodes. Data minimisation by design.
POSTCODE_AREAS = ["W6", "W12", "W14", "SW6", "NW10", "HA0", "UB6", "TW8", "SW1", "N1"]


# --------------------------------------------------------------------------- #
# NHS Number — Modulus 11
# --------------------------------------------------------------------------- #

def nhs_check_digit(first_nine: str) -> int | None:
    """Modulus 11 check digit. Returns None where the number is unusable."""
    total = sum(int(d) * (10 - i) for i, d in enumerate(first_nine))
    remainder = total % 11
    check = 11 - remainder
    if check == 11:
        return 0
    if check == 10:
        return None          # invalid — caller regenerates
    return check


def valid_nhs_number(rng: random.Random) -> str:
    while True:
        first_nine = f"{rng.randint(400_000_000, 799_999_999)}"
        cd = nhs_check_digit(first_nine)
        if cd is not None:
            return first_nine + str(cd)


def is_valid_nhs_number(value: str) -> bool:
    if not value or len(value) != 10 or not value.isdigit():
        return False
    return nhs_check_digit(value[:9]) == int(value[9])


def corrupt_nhs_number(rng: random.Random, good: str) -> str:
    """Produce a plausible-looking but check-digit-invalid number."""
    digits = list(good)
    i = rng.randint(0, 8)
    digits[i] = str((int(digits[i]) + rng.randint(1, 9)) % 10)
    return "".join(digits) if not is_valid_nhs_number("".join(digits)) else corrupt_nhs_number(rng, good)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def weighted(rng: random.Random, options):
    """options: sequence of tuples whose last element is the weight."""
    weights = [o[-1] for o in options]
    return rng.choices(options, weights=weights, k=1)[0]


def wait_days(rng: random.Random) -> int:
    """
    Plausible RTT wait distribution: a large body inside 18 weeks, a meaningful
    tail past it, and a small but non-zero population past 52 weeks.
    """
    r = rng.random()
    if r < 0.50:
        return int(rng.triangular(1, 126, 68))             # comfortably inside 18w
    if r < 0.84:
        return int(rng.triangular(90, 210, 145))           # around and past 18w
    if r < 0.972:
        return int(rng.triangular(180, 364, 245))          # long waiters
    return int(rng.triangular(364, 640, 405))              # past 52w


@dataclass
class Event:
    pathway_id: str
    event_type: str
    effective_date: str
    recorded_date: str
    payload: str = "{}"


@dataclass
class Pathway:
    pathway_id: str
    patient_id: str
    referral_id: str
    specialty_code: str
    specialty_name: str
    priority: str
    referral_source: str
    referrer_ods_code: str
    referral_received_date: str
    clock_start_date: str | None
    clock_stop_date: str | None
    clock_status: str
    status: str
    events: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

class Generator:
    def __init__(self, seed: int, as_of: date, dna_nullify_rate: float,
                 mess_rate: float):
        self.rng = random.Random(seed)
        self.as_of = as_of
        self.dna_nullify_rate = dna_nullify_rate
        self.mess = mess_rate
        self.stats = {
            "invalid_nhs_number": 0,
            "missing_clock_start": 0,
            "duplicate_referral": 0,
            "future_dated_event": 0,
            "clock_nullified": 0,
        }

    # -- patients ---------------------------------------------------------- #
    def patient(self, idx: int) -> dict:
        rng = self.rng
        nhs = valid_nhs_number(rng)
        if rng.random() < self.mess * 0.4:
            nhs = corrupt_nhs_number(rng, nhs)
            self.stats["invalid_nhs_number"] += 1
        age = int(rng.triangular(18, 95, 62))
        dob = self.as_of - timedelta(days=age * 365 + rng.randint(0, 364))
        return {
            "patient_id": f"P{idx:07d}",
            "nhs_number": nhs,
            "date_of_birth": dob.isoformat(),
            "postcode_sector": f"{rng.choice(POSTCODE_AREAS)} {rng.randint(1, 9)}",
            "imd_decile": rng.randint(1, 10),
        }

    # -- pathways ---------------------------------------------------------- #
    def pathway(self, idx: int, patient_id: str) -> Pathway:
        rng = self.rng
        spec = weighted(rng, SPECIALTIES)
        priority = weighted(rng, PRIORITIES)[0]
        source = weighted(rng, REFERRAL_SOURCES)[0]

        waited = wait_days(rng)
        received = self.as_of - timedelta(days=waited)

        pid = f"PW{idx:07d}"
        rid = f"R{idx:07d}"

        p = Pathway(
            pathway_id=pid,
            patient_id=patient_id,
            referral_id=rid,
            specialty_code=spec[0],
            specialty_name=spec[1],
            priority=priority,
            referral_source=source,
            referrer_ods_code=rng.choice(REFERRER_CODES),
            referral_received_date=received.isoformat(),
            clock_start_date=received.isoformat(),
            clock_stop_date=None,
            clock_status="RUNNING",
            status="OPEN",
        )

        # R12 — a small share of referrals arrive with no determinable clock
        # start. These must be REJECTED at ingestion, never defaulted.
        if rng.random() < self.mess * 0.25:
            p.clock_start_date = None
            p.clock_status = "INDETERMINATE"
            self.stats["missing_clock_start"] += 1
            p.events.append(Event(pid, "REFERRAL_RECEIVED", received.isoformat(),
                                  received.isoformat()))
            return p

        self._build_events(p, received, waited)
        return p

    def _build_events(self, p: Pathway, received: date, waited: int) -> None:
        rng = self.rng
        pid = p.pathway_id
        ev = p.events
        clock_start = received

        ev.append(Event(pid, "REFERRAL_RECEIVED", received.isoformat(), received.isoformat()))
        ev.append(Event(pid, "CLOCK_START", received.isoformat(), received.isoformat()))

        cursor = received + timedelta(days=rng.randint(3, 28))
        appt_index = 0

        # Up to three appointment cycles before an outcome.
        for _ in range(rng.randint(1, 3)):
            if cursor >= self.as_of:
                break
            appt_index += 1
            booked = cursor
            appt_date = booked + timedelta(days=rng.randint(10, 70))
            ev.append(Event(pid, "APPOINTMENT_BOOKED", booked.isoformat(), booked.isoformat(),
                            json.dumps({"appointmentDate": appt_date.isoformat(),
                                        "sequence": appt_index})))
            if appt_date >= self.as_of:
                cursor = appt_date
                break

            roll = rng.random()
            if roll < 0.075:                                     # DNA
                ev.append(Event(pid, "DNA", appt_date.isoformat(), appt_date.isoformat()))
                # R5 — nullification only ever applies to the FIRST appointment
                if appt_index == 1 and rng.random() < self.dna_nullify_rate:
                    clock_start = appt_date + timedelta(days=rng.randint(1, 14))
                    ev.append(Event(pid, "CLOCK_NULLIFIED", appt_date.isoformat(),
                                    appt_date.isoformat(),
                                    json.dumps({"rule": "R5",
                                                "newClockStart": clock_start.isoformat()})))
                    ev.append(Event(pid, "CLOCK_START", clock_start.isoformat(),
                                    clock_start.isoformat()))
                    p.clock_start_date = clock_start.isoformat()
                    self.stats["clock_nullified"] += 1
                cursor = appt_date + timedelta(days=rng.randint(7, 35))
                continue

            if roll < 0.115:                                     # patient cancelled — R7
                ev.append(Event(pid, "CANCELLED_BY_PATIENT", appt_date.isoformat(),
                                appt_date.isoformat()))
                cursor = appt_date + timedelta(days=rng.randint(7, 42))
                continue

            if roll < 0.155:                                     # provider cancelled — R6
                ev.append(Event(pid, "CANCELLED_BY_PROVIDER", appt_date.isoformat(),
                                appt_date.isoformat()))
                cursor = appt_date + timedelta(days=rng.randint(7, 28))
                continue

            ev.append(Event(pid, "APPOINTMENT_ATTENDED", appt_date.isoformat(),
                            appt_date.isoformat()))
            cursor = appt_date
            break

        # Outcome. Roughly 40% of pathways in a live snapshot are still open.
        if cursor >= self.as_of or rng.random() < 0.40:
            p.clock_status = "RUNNING"
            p.status = "OPEN"
        else:
            stop = min(cursor + timedelta(days=rng.randint(1, 45)),
                       self.as_of - timedelta(days=1))
            if stop < clock_start:
                stop = clock_start + timedelta(days=1)
            outcome = rng.random()
            if outcome < 0.70:                                   # R4 — treatment
                ev.append(Event(pid, "DECISION_TO_TREAT",
                                (stop - timedelta(days=rng.randint(0, 20))).isoformat(),
                                stop.isoformat()))
                ev.append(Event(pid, "TREATMENT_STARTED", stop.isoformat(), stop.isoformat()))
            elif outcome < 0.86:                                 # R4 — active monitoring
                ev.append(Event(pid, "ACTIVE_MONITORING_START", stop.isoformat(),
                                stop.isoformat()))
            elif outcome < 0.95:                                 # R4 — decision not to treat
                ev.append(Event(pid, "DECISION_NOT_TO_TREAT", stop.isoformat(), stop.isoformat()))
            else:                                                # R4 — patient declined
                ev.append(Event(pid, "PATIENT_DECLINED_TREATMENT", stop.isoformat(),
                                stop.isoformat()))
            ev.append(Event(pid, "PATHWAY_CLOSED", stop.isoformat(), stop.isoformat()))
            p.clock_stop_date = stop.isoformat()
            p.clock_status = "STOPPED"
            p.status = "CLOSED"

        # Occasional event recorded before it happened — a real data-quality issue
        # that your ingestion layer should detect rather than accept silently.
        if ev and rng.random() < self.mess * 0.2:
            victim = rng.choice(ev)
            victim.recorded_date = (date.fromisoformat(victim.effective_date)
                                    - timedelta(days=rng.randint(1, 5))).isoformat()
            self.stats["future_dated_event"] += 1


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate(n_patients: int, seed: int, as_of: date, dna_nullify_rate: float,
             mess_rate: float):
    g = Generator(seed, as_of, dna_nullify_rate, mess_rate)
    patients, pathways, events = [], [], []

    pw_idx = 0
    for i in range(1, n_patients + 1):
        pat = g.patient(i)
        patients.append(pat)

        # Most patients have one pathway; some have two or three concurrently.
        n_pw = 1 if g.rng.random() < 0.86 else g.rng.choice([2, 2, 3])
        for _ in range(n_pw):
            pw_idx += 1
            p = g.pathway(pw_idx, pat["patient_id"])
            events.extend(p.events)
            pathways.append(p)

        # Duplicate referrals happen. Your ingestion should catch them.
        if g.rng.random() < g.mess * 0.3 and pathways:
            pw_idx += 1
            src = pathways[-1]
            dup = Pathway(**{**asdict(src), "pathway_id": f"PW{pw_idx:07d}",
                             "referral_id": f"R{pw_idx:07d}", "events": []})
            dup.events = [Event(dup.pathway_id, e.event_type, e.effective_date,
                                e.recorded_date, e.payload) for e in src.events]
            events.extend(dup.events)
            pathways.append(dup)
            g.stats["duplicate_referral"] += 1

    return patients, pathways, events, g.stats


def derive_snapshot(pathways, as_of: date):
    """
    Reference implementation of the breach calculation, for cross-checking your
    Spring Boot version. Your Java implementation should agree with this on
    every row — if it does not, one of you has a bug and finding out which is
    the point of having both.
    """
    rows = []
    for p in pathways:
        if p.clock_status != "RUNNING" or not p.clock_start_date:
            continue
        start = date.fromisoformat(p.clock_start_date)
        waiting = (as_of - start).days
        rows.append({
            "pathway_id": p.pathway_id,
            "patient_id": p.patient_id,
            "specialty_name": p.specialty_name,
            "priority": p.priority,
            "clock_start_date": p.clock_start_date,
            "days_waiting": waiting,
            "weeks_waiting": round(waiting / 7, 1),
            "breach_date_18w": (start + timedelta(days=WEEKS_18)).isoformat(),
            "breach_date_52w": (start + timedelta(days=WEEKS_52)).isoformat(),
            "days_to_18w_breach": WEEKS_18 - waiting,
            "days_to_52w_breach": WEEKS_52 - waiting,
            "breached_18w": waiting > WEEKS_18,
            "breached_52w": waiting > WEEKS_52,
        })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic RTT pathway generator")
    ap.add_argument("--patients", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--as-of", type=str, default=date.today().isoformat())
    ap.add_argument("--dna-nullify-rate", type=float, default=0.35,
                    help="Share of first-appointment DNAs that nullify the clock (rule R5)")
    ap.add_argument("--mess-rate", type=float, default=0.04,
                    help="Overall data-quality defect rate")
    ap.add_argument("--out", type=str, default="./data")
    ap.add_argument("--validate", type=str, default=None,
                    help="Validate a previously generated directory and exit")
    args = ap.parse_args()

    if args.validate:
        return validate(args.validate)

    as_of = date.fromisoformat(args.as_of)
    os.makedirs(args.out, exist_ok=True)

    print(f"Generating {args.patients:,} patients (seed={args.seed}, as_of={as_of})...")
    patients, pathways, events, stats = generate(
        args.patients, args.seed, as_of, args.dna_nullify_rate, args.mess_rate)

    write_csv(os.path.join(args.out, "patients.csv"), patients,
              ["patient_id", "nhs_number", "date_of_birth", "postcode_sector", "imd_decile"])

    pw_rows = [{k: v for k, v in asdict(p).items() if k != "events"} for p in pathways]
    write_csv(os.path.join(args.out, "pathways.csv"), pw_rows, list(pw_rows[0].keys()))

    write_csv(os.path.join(args.out, "clock_events.csv"), [asdict(e) for e in events],
              ["pathway_id", "event_type", "effective_date", "recorded_date", "payload"])

    snapshot = derive_snapshot(pathways, as_of)
    write_csv(os.path.join(args.out, "expected_snapshot.csv"), snapshot,
              list(snapshot[0].keys()))

    open_pw = [p for p in pathways if p.clock_status == "RUNNING" and p.clock_start_date]
    b18 = sum(1 for r in snapshot if r["breached_18w"])
    b52 = sum(1 for r in snapshot if r["breached_52w"])

    summary = {
        "generated_at": date.today().isoformat(),
        "as_of": as_of.isoformat(),
        "seed": args.seed,
        "patients": len(patients),
        "pathways": len(pathways),
        "open_pathways_with_running_clock": len(open_pw),
        "events": len(events),
        "breached_18w": b18,
        "breached_52w": b52,
        "pct_within_18w": round(100 * (1 - b18 / max(len(snapshot), 1)), 1),
        "injected_defects": stats,
        "provenance": "Fully synthetic. Generated from NHS England's published RTT "
                      "rules and national statistics. No Trust data of any kind was "
                      "accessed, referenced or derived from.",
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  patients                 {len(patients):>10,}")
    print(f"  pathways                 {len(pathways):>10,}")
    print(f"  events                   {len(events):>10,}")
    print(f"  open, clock running      {len(open_pw):>10,}")
    print(f"  breaching 18 weeks       {b18:>10,}  ({summary['pct_within_18w']}% within standard)")
    print(f"  breaching 52 weeks       {b52:>10,}")
    print(f"\n  injected defects (your ingestion layer should catch these):")
    for k, v in stats.items():
        print(f"    {k:<24} {v:>8,}")
    print(f"\nWritten to {args.out}/")
    return 0


def validate(path: str) -> int:
    """Re-check a generated dataset. Useful as a CI step once you have ingestion."""
    problems = 0
    with open(os.path.join(path, "patients.csv"), encoding="utf-8") as fh:
        bad = [r["patient_id"] for r in csv.DictReader(fh)
               if not is_valid_nhs_number(r["nhs_number"])]
    print(f"invalid NHS numbers      {len(bad):>8,}   (expected: non-zero, by design)")

    with open(os.path.join(path, "pathways.csv"), encoding="utf-8") as fh:
        pws = list(csv.DictReader(fh))
    missing = [p["pathway_id"] for p in pws if not p["clock_start_date"]]
    print(f"missing clock start      {len(missing):>8,}   (expected: non-zero, by design)")

    for p in pws:
        if p["clock_start_date"] and p["clock_stop_date"]:
            if p["clock_stop_date"] < p["clock_start_date"]:
                problems += 1
    print(f"stop before start        {problems:>8,}   (expected: 0)")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
