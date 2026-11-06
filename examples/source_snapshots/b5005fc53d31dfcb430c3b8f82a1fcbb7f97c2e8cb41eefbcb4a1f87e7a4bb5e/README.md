# Historical example runtime

This directory preserves the exact runtime bytes that produced the bundled synthetic examples. Its full runtime fingerprint is the directory name. The examples remain unchanged.

This historical implementation includes the independent-graph execution limitation corrected in the current runtime. Use the repository root for new experiments. These old examples support historical software replay, not the revised independent-graph admission claim.

For trusted historical checkpoint reexecution, run the `scripts/evaluate_checkpoints.py` inside this snapshot against the chosen complete example directory. It verifies the original source identity before loading checkpoints. Use the ordinary repository environment.
