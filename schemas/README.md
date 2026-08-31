# Public schemas

`hmx-1.7.schema.json` is the normative, versioned interoperability contract for
Hexis Memory Exchange. It is MIT-licensed with the repository, packaged with the
Python distribution, and consumed directly by `core.memory_exchange` for export
and import validation.

Other memory and agent projects are welcome to import or emit HMX. Preserve
unknown minor fields, honor the declared export intent, remap export-scoped
references, retain provenance, and never treat transported embeddings as local
truth. See `docs/reference/hmx.md` for the human-readable contract and
`docs/hmx-acceptance.md` for executable compatibility evidence.

