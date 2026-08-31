# Proxy replication decision implementation v1

Status: frozen on 2026-08-31 after the DocVQA full answer-likelihood job began,
but before its first complete report or any full-bank endpoint was available.
Only the already frozen five-condition rule in
`proxy-to-outcome-cross-domain-protocol-v1.md` is implemented here.

## Bound code

- Protocol SHA-256:
  `106879da7d15db351a4145e5a06c43fc3f33803182d1ca4e6f08362b076f8cbe`.
- Code revision:
  `d1301e02b143d68565789f47c9b7ca4fe4e4fc73`.
- Decision module SHA-256:
  `f800edfdb516caf128e0036d824130dc7849e13a599a484d715513d2aa294613`.
- CLI SHA-256:
  `61bbcd5392eceb65837d95ffc25c23f8b4e29690eb4683037afe5dc176232204`.
- Test SHA-256:
  `1065f0a85b9fcc576c03c9a8b47a14cb940699d7c569c4eb4a1a7267256a97bf`.
- Targeted tests: `2 passed` before this contract was written.

## Input guards

The evaluator requires:

- the frozen proxy-audit schema and study label
  `DocVQA ranker development`;
- report and protocol hashes supplied by the caller when the result is run;
- the report-bound protocol hash to equal the independently hashed protocol;
- exactly 2,000 bootstrap resamples, seed `20260901`, and confidence `0.95`;
- 2,000 valid resamples for every consumed metric;
- all candidate-search and protected-outcome-use flags to be false;
- all five fixed sparse call-rate rows to be present exactly once.

## Mechanical decision

The evaluator implements the protocol without an extra threshold:

1. answer-loss Spearman 95% lower endpoint greater than zero;
2. answer-loss top-one task-gain 95% lower endpoint greater than zero;
3. answer-loss top-one task-gain point estimate greater than both entropy and
   exact-uniform-random estimates;
4. at least one fixed rate in `[0.005, 0.01, 0.02, 0.05, 0.10]` with a positive
   answer-loss policy-utility lower endpoint;
5. answer-loss top-one harm point estimate below both entropy and random.

All five passing yields `replicated_alignment`. Conditions 1 and 2 passing
while any later condition fails yields `partial_alignment`. Any failure of
condition 1 or 2 yields `non_replication`.

The evaluator writes JSON, Markdown, and a completion record binding their
hashes. It explicitly records that no score threshold or call rate was selected
and that no protected outcome was used. Existing outputs are never overwritten.
