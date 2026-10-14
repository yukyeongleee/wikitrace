"""make_split: positional + stratified + explicit-role splits and the sizing policies."""

from __future__ import annotations

import pytest

from wikitrace.config import make_split, split_sizes
from wikitrace.config import Sample
from wikitrace.config import DatasetConfig


def _samples(n: int, strata: list[str] | None = None) -> list[Sample]:
    return [
        Sample(id=f"s{i:03d}", db_id="db", question=f"q{i}", stratum=(strata[i] if strata else ""))
        for i in range(n)
    ]


def _cfg(**kw) -> DatasetConfig:
    base = dict(test_size=100, train_size=50)
    base.update(kw)
    return DatasetConfig(**base)


# ---------------------------------------------------------------------------
# split_sizes
# ---------------------------------------------------------------------------


def test_split_sizes_fixed_test():
    assert split_sizes(300, _cfg(test_size=100, train_size=50), "db") == (100, 50)


def test_split_sizes_raises_when_pool_too_small_for_fixed_test():
    with pytest.raises(ValueError, match="short of test_size"):
        split_sizes(120, _cfg(test_size=100, train_size=50), "db")


def test_split_sizes_null_test_takes_remainder():
    assert split_sizes(140, _cfg(test_size=None, train_size=50), "db") == (90, 50)


def test_split_sizes_null_raises_when_no_room_for_test():
    with pytest.raises(ValueError, match="cannot fund train_size"):
        split_sizes(50, _cfg(test_size=None, train_size=50), "db")


# ---------------------------------------------------------------------------
# Positional split (no stratum)
# ---------------------------------------------------------------------------


def test_positional_split_head_test_tail_train_reversed():
    s = _samples(200)
    sp = make_split(s, _cfg(test_size=100, train_size=50), "db")
    assert sp.test == s[:100]
    assert sp.train[0].id == s[-1].id          # tail reversed
    assert sp.train[-1].id == s[150].id
    assert {x.id for x in sp.test}.isdisjoint({x.id for x in sp.train})


# ---------------------------------------------------------------------------
# Stratified split — both sides must span the strata
# ---------------------------------------------------------------------------


def test_stratified_split_spans_strata_on_both_sides():
    # 4 strata x 30 = 120 samples, grouped (all of A, then B, ...).
    strata = [c for c in "ABCD" for _ in range(30)]
    s = _samples(120, strata)
    sp = make_split(s, _cfg(test_size=None, train_size=50), "db")
    test_strata = {x.stratum for x in sp.test}
    train_strata = {x.stratum for x in sp.train}
    # A positional cut would give test={A,B,...} / train={D} with no overlap;
    # round-robin makes both sides cover all four.
    assert test_strata == set("ABCD")
    assert train_strata == set("ABCD")
    assert {x.id for x in sp.test}.isdisjoint({x.id for x in sp.train})
    assert len(sp.train) == 50 and len(sp.test) == 70


def test_stratified_train_is_broad_not_deep_when_strata_are_many():
    # 50 strata x 3 = 150; train_size 50 -> one per stratum, no repeats.
    strata = [f"t{i}" for i in range(50) for _ in range(3)]
    s = _samples(150, strata)
    sp = make_split(s, _cfg(test_size=100, train_size=50), "db")
    assert len({x.stratum for x in sp.train}) == 50  # 50 distinct templates


def test_stratified_is_deterministic():
    strata = [c for c in "ABC" for _ in range(20)]
    s = _samples(60, strata)
    cfg = _cfg(test_size=None, train_size=30)
    a = make_split(s, cfg, "db")
    b = make_split(_samples(60, strata), cfg, "db")
    assert [x.id for x in a.test] == [x.id for x in b.test]
    assert [x.id for x in a.train] == [x.id for x in b.train]


def _role_samples(roles: list[str]) -> list[Sample]:
    return [
        Sample(id=f"r{i:03d}", db_id="db", question=f"q{i}", split_role=role)
        for i, role in enumerate(roles)
    ]


def test_explicit_role_split_bypasses_config_sizing():
    # A benchmark that pins each sample's role OWNS its
    # membership: config sizes must not move it — not even sizes this n cannot
    # fund, which the sizing path would reject outright.
    sp = make_split(_role_samples(["train", "train", "test"]),
                    _cfg(test_size=100, train_size=50), "db")
    assert [x.id for x in sp.train] == ["r000", "r001"]
    assert [x.id for x in sp.test] == ["r002"]


def test_explicit_role_limits_are_opt_in_and_preserve_frozen_order():
    samples = _role_samples(["train"] * 50 + ["test"] * 100)
    cfg = _cfg(train_size=4, test_size=2)
    original = make_split(samples, cfg, "db")
    limited = make_split(samples, cfg.model_copy(update={"limit_explicit_split": True}), "db")
    assert (len(original.train), len(original.test)) == (50, 100)
    assert limited.train == original.train[:4]
    assert limited.test == original.test[:2]


@pytest.mark.parametrize("field", ["train_size", "test_size"])
@pytest.mark.parametrize("value", [True, False, 0, -1, None, 1.5, "4"])
def test_explicit_role_limits_require_positive_integer_sizes(field, value):
    # The check is scoped to the explicit-role branch of `make_split`; the config model
    # must stay silent, or `test_size: null` becomes unloadable. `model_copy` is what
    # puts an unparsed value in front of that branch.
    samples = _role_samples(["train"] * 4 + ["test"] * 2)
    cfg = _cfg(limit_explicit_split=True, train_size=4, test_size=2)
    with pytest.raises(ValueError, match="positive integer"):
        make_split(samples, cfg.model_copy(update={field: value}), "db")


def test_null_test_size_validates_without_explicit_split_limits():
    assert _cfg(test_size=None, limit_explicit_split=False).test_size is None


@pytest.mark.parametrize("field", ["train_size", "test_size"])
def test_explicit_role_limits_refuse_underfilled_roles(field):
    samples = _role_samples(["train"] * 4 + ["test"] * 2)
    sizes = {"train_size": 4, "test_size": 2, field: 5}
    with pytest.raises(ValueError, match="exceeds frozen role size"):
        make_split(samples, _cfg(limit_explicit_split=True, **sizes), "db")


def test_explicit_role_limits_validate_copied_config_and_leave_other_splits_unchanged():
    samples = _role_samples(["train"] * 4 + ["test"] * 2)
    cfg = _cfg(limit_explicit_split=True, train_size=4, test_size=2)
    with pytest.raises(ValueError, match="positive integer"):
        make_split(samples, cfg.model_copy(update={"test_size": True}), "db")
    ordinary = _samples(8, list("AABBCCDD"))
    assert make_split(ordinary, cfg, "db") == make_split(
        ordinary, cfg.model_copy(update={"limit_explicit_split": False}), "db"
    )


def test_explicit_role_limit_is_recorded_in_config_fingerprint():
    from wikitrace.config import config_hash, load_config, resolved_dict

    cfg = load_config("configs/base.yaml")
    original = config_hash(cfg)
    cfg.dataset.limit_explicit_split = True
    assert resolved_dict(cfg)["dataset"]["limit_explicit_split"] is True
    assert config_hash(cfg) != original


def test_explicit_role_split_requires_both_sides():
    # An all-train role set is a vacuous suite -> fail loud here, rather than run
    # with an empty test and report meaningless metrics.
    with pytest.raises(ValueError, match="needs both train and test"):
        make_split(_role_samples(["train", "train"]), _cfg(), "db")


def test_explicit_role_split_rejects_a_sample_on_both_sides():
    s = _role_samples(["train", "test"])
    s[1] = s[1].model_copy(update={"id": s[0].id})
    with pytest.raises(ValueError, match="marked both train and test"):
        make_split(s, _cfg(), "db")


def test_stratified_raises_if_train_underfilled():
    # Too few samples to fund a 50-train after the test draw.
    strata = [c for c in "AB" for _ in range(20)]  # 40 samples
    with pytest.raises(ValueError):
        make_split(_samples(40, strata), _cfg(test_size=None, train_size=50), "db")
