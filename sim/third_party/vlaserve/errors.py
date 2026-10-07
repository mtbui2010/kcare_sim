"""Typed errors. Caller mistakes map to 4xx in the server; everything else is a 500."""


class VLAServeError(Exception):
    """Base class."""


class ModelNotFound(VLAServeError):
    """The name is neither a registry alias, a local checkpoint directory nor a cached HF repo."""


class ModelNotLoaded(VLAServeError):
    """The server knows the model but has not loaded it."""


class ObservationError(VLAServeError):
    """The observation does not match the policy's input spec (missing key, wrong shape...)."""


class ProfileMismatch(VLAServeError):
    """A robot profile and a policy spec disagree (camera names, state/action dimension)."""


class UnsupportedCheckpoint(VLAServeError):
    """The checkpoint is not in a format any backend can load."""
