__GUIDED_SAMPLERS__ = {}


def register_guided_sampler(name: str):
    def wrapper(cls):
        if __GUIDED_SAMPLERS__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __GUIDED_SAMPLERS__[name] = cls
        return cls

    return wrapper


def get_guided_sampler(
    name: str,
    model,
    sde,
    shape,
    sampling_eps,
    inverse_scaler,
    H_func,
    noiser,
    device,
    **kwargs,
):
    if __GUIDED_SAMPLERS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __GUIDED_SAMPLERS__[name](
        model_fn=model,
        sde=sde,
        shape=shape,
        sampling_eps=sampling_eps,
        inverse_scaler=inverse_scaler,
        H_func=H_func,
        noiser=noiser,
        device=device,
        **kwargs,
    )