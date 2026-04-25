PROJECT_DIRECTION_LOCK:
Build a positive, rigorous, reproducible NeurIPS-main-track method for black-box / strict-limited-output LLM model extraction or functional cloning from API-observable signals, with clear mechanism, fair baselines, and strong ablations.

Allowed changes:
- mechanism;
- objective/loss;
- query selection;
- teacher-student setup, if it remains a legitimate black-box model-extraction threat model;
- logging and verification;
- ablations and baselines.

Forbidden pivots:
- negative-result-only paper;
- workshop-level failure analysis as main target;
- dataset-specific preprocessing trick;
- benchmark-only optimization;
- weakening CE-only/top-K baselines;
- copying existing methods;
- paper claim before gates pass.

Scope Compliance Status:
PASS for T-DART if implemented as black-box teacher-specific residual extraction under strict API.
