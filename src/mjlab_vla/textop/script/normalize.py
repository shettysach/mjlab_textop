from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MOTION_REL = (
    "TextOpTracker/artifacts/Data10k-open/"
    "homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/"
    "motion.npz"
)


@dataclass(kw_only=True)
class NormalizeCommand:
    normalized_motion_file: str = "/tmp/textop_walk_mjlab.npz"
    motion_rel: str = DEFAULT_MOTION_REL
    data_dir: str = "/tmp/textop-data"
    device: str = "cuda:0"
