from hwr.apps.evaluate_foundation_world_model import ABLATIONS, build_parser


def test_foundation_evaluation_defaults_match_fixed_acceptance_protocol() -> None:
    arguments = build_parser().parse_args(["runs/example"])

    assert arguments.seed_count == 20
    assert arguments.video_seed_count == 1
    assert ABLATIONS == ("none", "lock_left", "lock_right")


def test_foundation_evaluation_has_no_exploration_or_training_switch() -> None:
    destinations = {action.dest for action in build_parser()._actions}

    assert "exploration" not in destinations
    assert "train" not in destinations
    assert "expert" not in destinations
