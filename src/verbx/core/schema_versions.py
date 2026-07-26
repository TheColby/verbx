"""Shared schema/version constants for JSON payloads and metadata.

Using centralized constants prevents silent drift between payload producers,
docs, and tests as features evolve.
"""

from __future__ import annotations

# Batch and checkpoint payloads
BATCH_MANIFEST_VERSION = "0.5"
BATCH_CHECKPOINT_VERSION = "0.5"

# IR morph batch outputs
IR_MORPH_SWEEP_VERSION = "0.7"

# Augmentation outputs
AUGMENT_MANIFEST_VERSION = "0.7"
AUGMENT_SUMMARY_VERSION = "0.7"
AUGMENT_QA_BUNDLE_VERSION = "augmentation-qa-v1"

# Immersive deliverables
IMMERSIVE_DELIVERABLE_VERSION = "0.7"
IMMERSIVE_QUEUE_VERSION = "0.7"
IMMERSIVE_ADM_SIDECAR_SCHEMA = "verbx.adm-bwf.sidecar.v0.7"

# Calibration and analysis sidecars
TRACK_C_CALIBRATION_VERSION = "track-c-cal-v1"

# IR synthesis metadata
IR_GENERATOR_METADATA_VERSION = "0.6.0"
