"""Load checkpoints saved by older LeRobot versions.

LeRobot renames processor steps between releases, and a pipeline JSON that names a step the
installed version does not know fails to load. This module registers the old names as thin
subclasses of their replacements, so the checkpoint on disk is never modified.

A subclass, not a second registration of the same class: `ProcessorStepRegistry.register` stores
the name on the class (`_registry_name`), so registering the new class under an old name would
make it serialise under the old name too. A subclass also keeps `isinstance` checks working,
which `make_pre_post_processors` relies on to pair a relative-actions step with its
absolute-actions inverse.

Add a rename here only after confirming the config fields are identical across the two versions.
"""
from __future__ import annotations

_installed = False

# old name -> (module, new class name). Each verified by reading both versions' source.
LEGACY_STEPS = {
    # LeRobot 0.5.x -> 0.6: DeltaActions renamed to RelativeActions, same fields
    # (enabled, exclude_joints, action_names). Seen on KETI-IRRC/pi05_openarm_* checkpoints.
    "delta_actions_processor": ("lerobot.processor.relative_action_processor",
                                "RelativeActionsProcessorStep"),
}


def install() -> list[str]:
    """Register every legacy name the installed LeRobot no longer knows. Idempotent."""
    global _installed
    if _installed:
        return []
    import importlib

    from lerobot.processor.pipeline import ProcessorStepRegistry

    added = []
    for old, (module, cls_name) in LEGACY_STEPS.items():
        if old in ProcessorStepRegistry._registry:
            continue  # this LeRobot still knows it
        new_cls = getattr(importlib.import_module(module), cls_name)
        legacy = type(f"Legacy_{old}", (new_cls,), {"__doc__": f"Alias of {cls_name} for '{old}'."})
        ProcessorStepRegistry.register(old)(legacy)
        added.append(old)
    _installed = True
    return added
