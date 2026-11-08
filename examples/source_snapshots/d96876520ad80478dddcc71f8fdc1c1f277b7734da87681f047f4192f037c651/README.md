# Preserved runtime source

This is the exact CROSS-MORSE runtime from repository revision `89dedcbe1432b4e9580fa4cabcd0923f9f28f6c6`, with source identity `d96876520ad80478dddcc71f8fdc1c1f277b7734da87681f047f4192f037c651`.

It preserves the code whose selection took the studentized maximum over every executed registered configuration, so that artifacts declaring this identity can be verified and replayed with their producing code. The current runtime takes that maximum over the feasible family Q_feas of main Eq. (18), as Supplementary Algorithm S4 specifies, and records the family in every selection decision; records without it replay with every executed configuration. New experiments use the current runtime and record its distinct identity.

`SOURCE_ORIGIN.json` records the originating revision and every runtime file hash. The source resolver verifies the complete runtime fingerprint before use.
