# Contributing

For a bug report, include the source commit, Python/PyTorch versions, configuration,
command, and the relevant error. Use a small synthetic example when possible.

## Development

Create a branch, make a focused change, and run the checks relevant to it:

```bash
make verify
make test
```

The [method guide](docs/METHOD.md) maps the major components to their source files.
[Validation](docs/VALIDATION.md) explains software checks and scientific evidence.

Keep predictor freezing, label-role separation, confirmation-before-admission,
and policy lock intact. Changes to those contracts need focused regression tests.
Run outputs record the source fingerprint, configuration, split identities, and
artifact hashes. Preserve sealed historical examples; a new implementation should
produce a separately identified run.

## Results and documentation

Experimental claims need the actual predictions or per-unit observations and
measurement records. Store official archives through the
[archive workflow](docs/RUN_ARCHIVES.md). Synthetic examples and transcribed
manuscript values retain their explicit labels.

Update the relevant guide rather than creating a second explanation of the same
workflow. Regenerate committed views after changing their source data:

```bash
make registry-views
make manifests
make availability
make manifest
make verify
```

`make manifest` is the final packaging step after intentional changes; it should
not be used to conceal missing or altered files inside a sealed example.

Submit a focused pull request describing the problem, resulting behavior, tests,
and any changed assumptions. Do not include credentials, local datasets, generated
training directories, or private allocation identifiers.
