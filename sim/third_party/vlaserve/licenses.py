"""Licence bookkeeping. Policy: record and warn, never block.

Every served model reports a licence, and anything that is not plainly permissive is printed as a
warning at load time and returned in `/v1/models`. Nothing is refused: this tool serves a lab's own
checkpoints, and the point is that whoever ships a robot knows what they are shipping.

A checkpoint's own config rarely says anything (`"license": null` on every LeRobot checkpoint seen
so far), so the licence comes from, in order: the registry entry, the checkpoint config, then an
inference from the backbone. Only inferences we have verified are encoded here.
"""
from __future__ import annotations

from dataclasses import dataclass

PERMISSIVE = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "bsd"}


@dataclass
class LicenseInfo:
    license: str          # SPDX-ish id, or a named custom licence, or "unknown"
    source: str           # "registry" | "checkpoint" | "inferred:<why>" | "none"
    note: str = ""

    @property
    def permissive(self) -> bool:
        return self.license.lower() in PERMISSIVE

    @property
    def warning(self) -> str | None:
        if self.permissive:
            return self.note or None
        if self.license == "unknown":
            return ("licence unknown: no registry entry, nothing in config.json, and no known "
                    "backbone. Check the model card before redistributing.")
        return f"licence '{self.license}' is not permissive. {self.note}".strip()


def _backbone_inference(cfg: dict) -> LicenseInfo | None:
    # pi0 / pi05 / pi0_fast are built on PaliGemma. The Gemma Terms treat fine-tunes and
    # distillations as Model Derivatives and require the use restrictions to flow down on
    # redistribution (ai.google.dev/gemma/terms; openpi ships a separate LICENSE_GEMMA.txt).
    if "paligemma_variant" in cfg:
        return LicenseInfo("gemma-terms", "inferred:paligemma backbone",
                           "PaliGemma derivative: redistribution must carry the Gemma use "
                           "restrictions and notice.")
    return None


def resolve(cfg: dict, registry_license: str | None) -> LicenseInfo:
    if registry_license:
        info = LicenseInfo(registry_license, "registry")
        inferred = _backbone_inference(cfg)
        if inferred and info.permissive:
            # A registry that says "apache-2.0" for a PaliGemma fine-tune is describing the code,
            # not the weights. Say so rather than silently trusting it.
            info.note = f"registry says {registry_license} but backbone implies {inferred.license}"
        return info
    if cfg.get("license"):
        return LicenseInfo(str(cfg["license"]), "checkpoint")
    return _backbone_inference(cfg) or LicenseInfo("unknown", "none")
