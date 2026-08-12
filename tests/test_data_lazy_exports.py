from __future__ import annotations

import subprocess
import sys


def test_data_package_does_not_eagerly_import_training_or_pyarrow() -> None:
    command = (
        "import sys; import hwr.data; "
        "import hwr.data.foundation_cache; "
        "print(int('torch' in sys.modules), int('pyarrow' in sys.modules), "
        "int('hwr.train' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "0 0 0"


def test_data_lazy_exports_preserve_public_imports() -> None:
    from hwr.data import FoundationCacheKey, FoundationFeatureCache

    assert FoundationCacheKey.__module__ == "hwr.data.foundation_cache"
    assert FoundationFeatureCache.__module__ == "hwr.data.foundation_cache"
