from nind_denoise.common.libs import utilities


def test_avg_listofdicts_empty():
    assert utilities.avg_listofdicts([]) == {}


def test_avg_listofdicts_union_of_keys():
    data = [
        {"a": 1, "b": 3},
        {"a": 3},
        {"b": 5, "c": 9},
    ]
    out = utilities.avg_listofdicts(data)
    assert out["a"] == 2
    assert out["b"] == 4
    assert out["c"] == 9
