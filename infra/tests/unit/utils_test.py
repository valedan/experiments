from infra import utils


def test_flatten():
    nested_dict = {"str": "foo", "dict": {"a": 1, "b": 2}, "nested": {"dict": {"a": 1}}}
    flat_dict = utils.flatten(nested_dict)
    assert flat_dict == {"str": "foo", "dict.a": 1, "dict.b": 2, "nested.dict.a": 1}


def test_unflatten():
    flat_dict = {"str": "foo", "dict.a": 1, "dict.b": 2, "nested.dict.a": 1}
    nested_dict = utils.unflatten(flat_dict)
    assert nested_dict == {"str": "foo", "dict": {"a": 1, "b": 2}, "nested": {"dict": {"a": 1}}}
