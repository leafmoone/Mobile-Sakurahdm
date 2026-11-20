from importlib import import_module

_EXPORTS = {
    "mobileoFastForCausalLM": ".language_model.mobileo",
    "mobileoFastConfig": ".language_model.mobileo",
    "mobileoForInferenceLM": ".language_model.mobileo_inference",
    "mobileoConfig": ".language_model.mobileo_inference",
    "mobileoFastSFTForCausalLM": ".language_model.mobileo_sft",
    "mobileoFastSFTConfig": ".language_model.mobileo_sft",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
