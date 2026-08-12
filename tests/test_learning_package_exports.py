from __future__ import annotations

import subprocess
import sys


def test_policy_and_train_packages_do_not_eagerly_import_legacy_models() -> None:
    command = (
        "import sys; import hwr.policy; import hwr.train; "
        "print(int('hwr.policy.vla_model' in sys.modules), "
        "int('hwr.train.asymmetric_rl' in sys.modules), "
        "int('pyarrow' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], check=True, capture_output=True, text=True
    )

    assert result.stdout.strip() == "0 0 0"


def test_new_lazy_learning_exports_remain_available() -> None:
    from hwr.policy import LatentActor, LatentValueModel
    from hwr.train import ImaginationActorCritic, ImaginationRLConfig

    assert LatentActor.__module__ == "hwr.policy.latent_actor"
    assert LatentValueModel.__module__ == "hwr.policy.latent_value"
    assert ImaginationActorCritic.__module__ == "hwr.train.imagination_rl"
    assert ImaginationRLConfig.__module__ == "hwr.train.imagination_rl"
