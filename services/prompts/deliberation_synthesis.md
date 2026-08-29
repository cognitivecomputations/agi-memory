Synthesize the council perspectives and adversarial review into an advisory result.

Return one JSON object with:

- `recommendation`: the best supported current recommendation, explicitly advisory;
- `report`: a concise explanation of why;
- `agreements`: a list of supported points;
- `disagreements`: a list of unresolved points;
- `risks`: a list of material risks;
- `dissent`: a list of minority positions worth preserving;
- `invalidation_conditions`: a list of observable facts that should trigger review.

Do not average away a strong objection. Do not claim certainty that the evidence does
not support. Do not execute, authorize, or gate any action. Give concise,
audit-ready reasons rather than private chain-of-thought.
