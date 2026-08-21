"""MuJoCo implementation of the project-owned three-dimensional runtime."""

from hwr.adapters.mujoco.backend import Mujoco3DBackend, Mujoco3DConfig
from hwr.adapters.mujoco.bimanual_backend import MujocoBimanualTaskBackend
from hwr.adapters.mujoco.bimanual_bindings import (
    BimanualMujocoBinding,
    load_bimanual_mujoco_bindings,
)
from hwr.adapters.mujoco.bindings import MujocoTaskBinding, load_mujoco_task_bindings
from hwr.adapters.mujoco.contact import GraspContactMonitor, GraspContactSample
from hwr.adapters.mujoco.contact_ledger import (
    ALLOWED_CONTACT_ROLES,
    CONTACT_CATEGORIES,
    ContactLedger,
    ContactLedgerError,
    ContactPointObservation,
    resolve_allowed_contact_role_ids,
    run_timestep_stability_fixture,
)
from hwr.adapters.mujoco.dual_arm_backend import MujocoDualArmBackend, MujocoDualArmConfig
from hwr.adapters.mujoco.entity_contact_graph import (
    ENTITY_ROLES,
    MEASUREMENT_SCHEMA,
    P40_FIELDS,
    ROBOT_BODY_ROOT_NAMES,
    ROBOT_PARTS,
    EntityContactGraph,
    EntityContactGraphError,
    EntityContactPointObservation,
    EntityMotionSource,
    p40_conservation_differences,
    resolve_robot_part_by_geom,
)
from hwr.adapters.mujoco.evidence import (
    BIMANUAL_EVIDENCE_VIEWS,
    MujocoBimanualEvidenceSource,
)
from hwr.adapters.mujoco.expert import PrivilegedCartesianExpert
from hwr.adapters.mujoco.formal_expert import FormalExpertOutput, PrivilegedHouseholdExpert
from hwr.adapters.mujoco.formal_household_backend import (
    MujocoFormalHouseholdDualArmBackend,
)
from hwr.adapters.mujoco.household_backend import MujocoHouseholdBackend
from hwr.adapters.mujoco.inspection import RobotModelReport, inspect_robot_model
from hwr.adapters.mujoco.scene_preview import ScenePreview, render_scene_preview
from hwr.adapters.mujoco.smoke_trial import ContactGraspReport, run_contact_grasp_trial
from hwr.adapters.mujoco.training_catalog import (
    MujocoBimanualBackendFactory,
    MujocoFormalHouseholdBackendFactory,
    load_default_bimanual_training_catalogs,
    load_default_formal_household_catalogs,
)

__all__ = [
    "Mujoco3DBackend",
    "Mujoco3DConfig",
    "MujocoBimanualTaskBackend",
    "MujocoBimanualBackendFactory",
    "MujocoDualArmBackend",
    "MujocoDualArmConfig",
    "MujocoFormalHouseholdDualArmBackend",
    "MujocoFormalHouseholdBackendFactory",
    "MujocoBimanualEvidenceSource",
    "MujocoHouseholdBackend",
    "MujocoTaskBinding",
    "BimanualMujocoBinding",
    "GraspContactMonitor",
    "GraspContactSample",
    "ContactLedger",
    "ContactLedgerError",
    "ContactPointObservation",
    "EntityContactGraph",
    "EntityContactGraphError",
    "EntityContactPointObservation",
    "EntityMotionSource",
    "ContactGraspReport",
    "FormalExpertOutput",
    "PrivilegedCartesianExpert",
    "PrivilegedHouseholdExpert",
    "RobotModelReport",
    "BIMANUAL_EVIDENCE_VIEWS",
    "ALLOWED_CONTACT_ROLES",
    "CONTACT_CATEGORIES",
    "ENTITY_ROLES",
    "MEASUREMENT_SCHEMA",
    "P40_FIELDS",
    "ROBOT_BODY_ROOT_NAMES",
    "ROBOT_PARTS",
    "ScenePreview",
    "inspect_robot_model",
    "load_mujoco_task_bindings",
    "load_bimanual_mujoco_bindings",
    "load_default_bimanual_training_catalogs",
    "load_default_formal_household_catalogs",
    "render_scene_preview",
    "p40_conservation_differences",
    "resolve_robot_part_by_geom",
    "resolve_allowed_contact_role_ids",
    "run_timestep_stability_fixture",
    "run_contact_grasp_trial",
]
