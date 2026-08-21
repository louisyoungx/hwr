"""Report-only contact force and impulse accounting for formal MuJoCo tasks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import mujoco
import numpy as np


ALLOWED_CONTACT_ROLES = (
    "floor_support",
    "manipulated_object",
    "target_container",
    "articulation",
)
CONTACT_CATEGORIES = (*ALLOWED_CONTACT_ROLES, "forbidden")
TIMESTEP_FIXTURE_ID = "hwr.mujoco-contact-ledger-timestep/v1"
_TIMESTEP_FIXTURE_XML = """
<mujoco model="contact_ledger_timestep">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="robot" pos="0 0 0.35">
      <freejoint/>
      <geom name="robot_collision" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


class ContactLedgerError(RuntimeError):
    """Raised when force evidence cannot satisfy the fail-closed contract."""


@dataclass(frozen=True)
class ContactPointObservation:
    geom1: int
    geom2: int
    normal_force: float | None = None


@dataclass
class _CategoryAccumulator:
    pair_peak_force: float = 0.0
    category_peak_force: float = 0.0
    impulse: float = 0.0
    contact_duration_seconds: float = 0.0
    contact_point_count: int = 0
    unique_pair_observation_count: int = 0

    def as_period_dict(self) -> dict[str, float | int]:
        return {
            "pair_peak_force": self.pair_peak_force,
            "category_peak_force": self.category_peak_force,
            "category_impulse": self.impulse,
            "contact_duration_seconds": self.contact_duration_seconds,
            "contact_point_count": self.contact_point_count,
            "unique_pair_observation_count": self.unique_pair_observation_count,
        }

    def as_episode_dict(self) -> dict[str, float | int]:
        value = self.as_period_dict()
        value["cumulative_impulse"] = value.pop("category_impulse")
        return value


class ContactLedger:
    """Aggregate contact points into unordered pairs and semantic categories."""

    def __init__(
        self,
        *,
        robot_geoms: Iterable[int],
        allowed_role_by_geom: Mapping[int, str],
        timestep: float,
        enabled: bool,
    ) -> None:
        self.robot_geoms = frozenset(int(value) for value in robot_geoms)
        self.allowed_role_by_geom = {
            int(geom): str(role) for geom, role in allowed_role_by_geom.items()
        }
        self.timestep = float(timestep)
        self.enabled = bool(enabled)
        if not self.robot_geoms:
            raise ValueError("contact ledger requires robot geometry")
        if not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise ValueError("contact ledger timestep must be finite and positive")
        unknown = set(self.allowed_role_by_geom.values()) - set(ALLOWED_CONTACT_ROLES)
        if unknown:
            raise ValueError(f"contact ledger roles are unknown: {sorted(unknown)}")
        if self.robot_geoms & self.allowed_role_by_geom.keys():
            raise ValueError("robot geometry cannot also be an allowed environment geom")
        self.reset()

    def reset(self) -> None:
        self._episode = _empty_categories()
        self._period: dict[str, _CategoryAccumulator] | None = None
        self._periods: list[dict[str, object]] = []
        self._physics_substep_count = 0
        self._contact_point_count = 0
        self._robot_environment_contact_point_count = 0
        self._ignored_robot_self_contact_point_count = 0
        self._ignored_world_world_contact_point_count = 0
        self._missing_normal_force_count = 0
        self._nonfinite_normal_force_count = 0
        self._invalid_negative_normal_force_count = 0

    def set_enabled(self, enabled: bool) -> None:
        if self._period is not None or self._periods:
            raise ContactLedgerError("contact ledger mode cannot change after measurement")
        self.enabled = bool(enabled)

    def begin_control_period(self) -> None:
        if self._period is not None:
            raise ContactLedgerError("contact ledger control period is already active")
        self._period = _empty_categories()
        self._period_substeps = 0
        self._period_contact_points = 0
        self._period_robot_environment_points = 0
        self._period_ignored_self = 0
        self._period_ignored_world = 0
        self._period_missing = 0
        self._period_nonfinite = 0
        self._period_negative = 0

    def sample_mujoco_substep(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.enabled:
            return
        if self._period is None:
            self.begin_control_period()
        points: list[ContactPointObservation] = []
        for index in range(data.ncon):
            contact = data.contact[index]
            first, second = int(contact.geom1), int(contact.geom2)
            normal: float | None = None
            if (first in self.robot_geoms) != (second in self.robot_geoms):
                force = np.empty(6, np.float64)
                try:
                    mujoco.mj_contactForce(model, data, index, force)
                    normal = abs(float(force[0]))
                except (ArithmeticError, RuntimeError, ValueError):
                    normal = None
            points.append(ContactPointObservation(first, second, normal))
        self.record_substep(points)

    def record_substep(self, points: Iterable[ContactPointObservation]) -> None:
        period = self._require_period()
        if not self.enabled:
            return
        pair_forces: dict[tuple[int, int], float] = {}
        pair_categories: dict[tuple[int, int], str] = {}
        point_counts = {category: 0 for category in CONTACT_CATEGORIES}
        invalid = False
        points = tuple(points)
        self._physics_substep_count += 1
        self._period_substeps += 1
        self._contact_point_count += len(points)
        self._period_contact_points += len(points)
        for point in points:
            first, second = int(point.geom1), int(point.geom2)
            robot_first = first in self.robot_geoms
            robot_second = second in self.robot_geoms
            if robot_first and robot_second:
                self._ignored_robot_self_contact_point_count += 1
                self._period_ignored_self += 1
                continue
            if not robot_first and not robot_second:
                self._ignored_world_world_contact_point_count += 1
                self._period_ignored_world += 1
                continue
            self._robot_environment_contact_point_count += 1
            self._period_robot_environment_points += 1
            other = second if robot_first else first
            category = self.allowed_role_by_geom.get(other, "forbidden")
            point_counts[category] += 1
            normal = point.normal_force
            if normal is None:
                self._missing_normal_force_count += 1
                self._period_missing += 1
                invalid = True
                continue
            normal = float(normal)
            if not math.isfinite(normal):
                self._nonfinite_normal_force_count += 1
                self._period_nonfinite += 1
                invalid = True
                continue
            if normal < 0.0:
                self._invalid_negative_normal_force_count += 1
                self._period_negative += 1
                invalid = True
                continue
            pair = tuple(sorted((first, second)))
            pair_forces[pair] = pair_forces.get(pair, 0.0) + normal
            pair_categories[pair] = category
        category_totals = {category: 0.0 for category in CONTACT_CATEGORIES}
        pair_counts = {category: 0 for category in CONTACT_CATEGORIES}
        for pair, force in pair_forces.items():
            category = pair_categories[pair]
            category_totals[category] += force
            pair_counts[category] += 1
            period[category].pair_peak_force = max(
                period[category].pair_peak_force, force
            )
        for category in CONTACT_CATEGORIES:
            total = category_totals[category]
            value = period[category]
            value.category_peak_force = max(value.category_peak_force, total)
            value.impulse += total * self.timestep
            value.contact_duration_seconds += self.timestep * (total > 0.0)
            value.contact_point_count += point_counts[category]
            value.unique_pair_observation_count += pair_counts[category]
        if invalid:
            raise ContactLedgerError(
                "contact ledger observed missing, nonfinite, or negative normal force"
            )

    def end_control_period(self) -> dict[str, object]:
        if not self.enabled:
            return self.report()
        if self._period is None:
            self.begin_control_period()
        period = self._require_period()
        report = {
            "period_index": len(self._periods),
            "physics_substep_count": self._period_substeps,
            "contact_point_count": self._period_contact_points,
            "robot_environment_contact_point_count": (
                self._period_robot_environment_points
            ),
            "ignored_robot_self_contact_point_count": self._period_ignored_self,
            "ignored_world_world_contact_point_count": self._period_ignored_world,
            "missing_normal_force_count": self._period_missing,
            "nonfinite_normal_force_count": self._period_nonfinite,
            "invalid_negative_normal_force_count": self._period_negative,
            "categories": {
                category: period[category].as_period_dict()
                for category in CONTACT_CATEGORIES
            },
        }
        for category in CONTACT_CATEGORIES:
            source = period[category]
            target = self._episode[category]
            target.pair_peak_force = max(
                target.pair_peak_force, source.pair_peak_force
            )
            target.category_peak_force = max(
                target.category_peak_force, source.category_peak_force
            )
            target.impulse += source.impulse
            target.contact_duration_seconds += source.contact_duration_seconds
            target.contact_point_count += source.contact_point_count
            target.unique_pair_observation_count += (
                source.unique_pair_observation_count
            )
        self._period = None
        self._periods.append(report)
        return report

    def report(self) -> dict[str, object]:
        return {
            "schema_version": "hwr.mujoco-contact-ledger/v1",
            "enabled": self.enabled,
            "measurement_only": True,
            "timestep": self.timestep,
            "control_period_count": len(self._periods),
            "physics_substep_count": self._physics_substep_count,
            "contact_point_count": self._contact_point_count,
            "robot_environment_contact_point_count": (
                self._robot_environment_contact_point_count
            ),
            "ignored_robot_self_contact_point_count": (
                self._ignored_robot_self_contact_point_count
            ),
            "ignored_world_world_contact_point_count": (
                self._ignored_world_world_contact_point_count
            ),
            "missing_normal_force_count": self._missing_normal_force_count,
            "nonfinite_normal_force_count": self._nonfinite_normal_force_count,
            "invalid_negative_normal_force_count": (
                self._invalid_negative_normal_force_count
            ),
            "contract_valid": not any(
                (
                    self._missing_normal_force_count,
                    self._nonfinite_normal_force_count,
                    self._invalid_negative_normal_force_count,
                )
            ),
            "categories": {
                category: self._episode[category].as_episode_dict()
                for category in CONTACT_CATEGORIES
            },
            "periods": list(self._periods),
        }

    def _require_period(self) -> dict[str, _CategoryAccumulator]:
        if self._period is None:
            raise ContactLedgerError("contact ledger control period is not active")
        return self._period


def run_timestep_stability_fixture() -> dict[str, object]:
    """Run the frozen dt/dt/2 contact fixture and compare category impulses."""
    duration_seconds = 1.0
    timesteps = (0.002, 0.001)
    reports = [_run_timestep_fixture(value, duration_seconds) for value in timesteps]
    relative_differences: dict[str, float] = {}
    for category in CONTACT_CATEGORIES:
        impulses = [
            float(report["categories"][category]["cumulative_impulse"])
            for report in reports
        ]
        if max(impulses) > 0.0:
            relative_differences[category] = (
                abs(impulses[0] - impulses[1]) / max(impulses)
            )
    return {
        "schema_version": TIMESTEP_FIXTURE_ID,
        "fixture_xml_sha256": hashlib.sha256(
            _TIMESTEP_FIXTURE_XML.encode("utf-8")
        ).hexdigest(),
        "duration_seconds": duration_seconds,
        "timesteps": list(timesteps),
        "reports": reports,
        "relative_impulse_differences": relative_differences,
        "maximum_relative_impulse_difference": max(
            relative_differences.values(), default=0.0
        ),
        "passed": bool(relative_differences)
        and max(relative_differences.values()) <= 0.10,
    }


def _run_timestep_fixture(timestep: float, duration: float) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_string(_TIMESTEP_FIXTURE_XML)
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    robot = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_collision"))
    floor = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    ledger = ContactLedger(
        robot_geoms=(robot,),
        allowed_role_by_geom={floor: "floor_support"},
        timestep=timestep,
        enabled=True,
    )
    ledger.begin_control_period()
    for _ in range(round(duration / timestep)):
        mujoco.mj_step(model, data)
        ledger.sample_mujoco_substep(model, data)
    ledger.end_control_period()
    return ledger.report()


def resolve_allowed_contact_role_ids(
    model: mujoco.MjModel,
    allowed_names: Iterable[str],
    role_names: Mapping[str, Iterable[str]],
) -> tuple[frozenset[int], dict[int, str]]:
    """Resolve explicit roles and recheck their union at the loaded model boundary."""
    if set(role_names) != set(ALLOWED_CONTACT_ROLES):
        raise ValueError("allowed robot contact role keys differ from the contract")
    allowed = frozenset(_geom_id(model, name) for name in allowed_names)
    role_by_geom: dict[int, str] = {}
    for role in ALLOWED_CONTACT_ROLES:
        for name in role_names[role]:
            geom = _geom_id(model, name)
            if geom in role_by_geom:
                raise ValueError("resolved allowed contact roles overlap")
            role_by_geom[geom] = role
    if frozenset(role_by_geom) != allowed:
        raise ValueError("resolved allowed contact role union differs from allow-list")
    return allowed, role_by_geom


def contact_ledger_from_binding(
    model: mujoco.MjModel,
    robot_geoms: Iterable[int],
    binding: object,
) -> ContactLedger:
    _, role_by_geom = resolve_allowed_contact_role_ids(
        model,
        getattr(binding, "allowed_robot_contact_geoms"),
        getattr(binding, "allowed_robot_contact_roles"),
    )
    return ContactLedger(
        robot_geoms=robot_geoms,
        allowed_role_by_geom=role_by_geom,
        timestep=float(model.opt.timestep),
        enabled=False,
    )


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    value = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
    if value < 0:
        raise ValueError(f"formal dual-arm scene is missing {name}")
    return value


def _empty_categories() -> dict[str, _CategoryAccumulator]:
    return {category: _CategoryAccumulator() for category in CONTACT_CATEGORIES}
