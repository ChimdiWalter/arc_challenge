# Shared-source release

Prepared 2026-09-27T17:55:27.879031+00:00. Repository role: arc_challenge.

This is a focused release view of the existing single InvariantCell implementation. All four repositories carry the same small shared Python core, identified by SHA256 `8c35f6006c8a48ebe02d7ba4870c8fd1414470f158b3d85c0fd81dcd8b41071c`, plus their relevant entry scripts and portable engineering tests. They are not four independently trained systems. Shared changes should be developed once and exported consistently.

Every included original source file is listed with its hash in release_manifest.json. This export intentionally omits local configuration, authentication state, private notes, datasets, model binaries, full research runs and quarantined outcomes. Evidence is an aggregate snapshot; source-run hashes establish local provenance, not unrestricted data redistribution or clinical validity.

Some archived research entry scripts require the original frozen asset/configuration manifests, which are not part of this portable export. The portable tests and check/replay example work without those research assets. Reproducing full empirical runs requires separately obtaining permitted assets and recreating the documented freezes; no missing outcome should be guessed.
