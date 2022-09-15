import dataclasses


def override_params(dc, exp):
    for k, v in exp.items():
        if hasattr(dc, k):
            if dataclasses.is_dataclass(getattr(dc, k)):
                setattr(dc, k, override_params(getattr(dc, k), v))
            else:
                setattr(dc, k, v)
    return dc
