# GapSight external-comparator feasibility audit

Date checked: 2026-09-01

Status: prospective planning note written while InfographicVQA full-generation
job `200130` was running, before DECAR train OOF outcomes existed.  It does not
change the frozen DECAR experiment, authorize a held-out evaluation, or supply
an external result.

## Decision

An author-code-faithful GapSight comparison is **not currently executable from
the discoverable public materials**.  A targeted search of the paper title,
method name, and arXiv identifier found the arXiv abstract and full text, but no
official repository, checkpoint, project page, or configuration link.  This is
a dated search result, not proof that no release exists; the search must be
repeated before any external-comparator freeze.

The current frozen `loss_only` variant is therefore only a **GapSight-style
fixed-bank loss-gap ablation**.  It is not GapSight, not an author-faithful
reproduction, and cannot support a claim that DECAR beats GapSight.

Primary paper audited:

- Learning to Look Again / GapSight: https://arxiv.org/abs/2608.21762
- arXiv HTML full text: https://arxiv.org/html/2608.21762

## What the paper specifies well enough to identify the method

The public text specifies the method at a conceptual level:

- execute a diverse crop candidate bank offline against a frozen target VLM;
- use teacher-forced answer-NLL reduction, or correct-option margin increase,
  as target-model-specific crop utility;
- choose the maximum-utility candidate, map low or negative utility to
  `preserve`, and filter or down-weight ambiguous supervision;
- train a router with preserve/review, scalar-utility, and continuous-box
  outputs; and
- evaluate across six benchmarks and three backbones, including 1,000-example
  subsets for several VQA datasets.

These facts are enough to establish a direct conceptual collision and to
design a clearly labelled paper-described reimplementation.  They are not
enough to claim reproduction of the reported system.

## Reproduction-critical fields not discoverable in the audited release

The paper materials available in this audit do not expose an executable
configuration containing all of the following:

- the exact candidate-bank size and complete crop geometries;
- preserve/review and ambiguity thresholds;
- optimizer, learning-rate schedule, epoch count, batch size, and seed policy;
- numerical weights for the classification, utility, and box losses;
- the exact identities or sampling seeds of reported 1,000-example subsets;
- the precise image-token accounting and preprocessing configuration for every
  backbone; and
- a versioned implementation, environment, checkpoint, and inference command.

These omissions prevent an author-code-faithful same-population comparator.
They do not invalidate the paper's published results.

## Evidence taxonomy

Any future table must label GapSight evidence using exactly one of these
levels:

1. **Official executable release.** Versioned author code and configuration,
   with checkpoint if required.  This is the preferred external comparator.
2. **Author-clarified reproduction.** Our implementation after the missing
   configuration has been supplied and frozen.  Any author contact requires
   the user's explicit authorization; no message is sent by this audit.
3. **Paper-described reimplementation.** Our best implementation from public
   text, with every inferred choice listed.  It must not be called
   author-faithful.
4. **GapSight-style fixed-bank ablation.** The existing DECAR `loss_only`
   variant.  This isolates loss-gap supervision inside our bank and execution
   protocol; it is not an external comparison.
5. **Published-number context.** Numbers copied from the paper with its
   backbone, population, action rate, and metric.  These are contextual only,
   never a head-to-head ranking.

No result at levels 3--5 licenses the phrase "DECAR beats GapSight."

## Conditional execution plan

Do not spend a large compute budget reproducing GapSight before the registered
DECAR train gate is known.  If DECAR does not advance, the external comparison
cannot rescue the frozen positive method claim and should not be run merely to
search for a favorable table.

If DECAR advances:

1. repeat the public-release search and record the exact date and source
   versions;
2. prefer an official release, or request the user's authorization before any
   author contact;
3. freeze implementation level, dataset identifiers, source splits, backbone,
   preprocessing, execution/token cost, seeds, and pass/fail rule before
   opening held-out outcomes;
4. run the comparator on the same population and backbone as DECAR, with both
   matched-call and matched-compute views where continuous crops make cost
   non-equivalent; and
5. report failures and implementation deviations unchanged.

If no official or clarified implementation becomes available, a
paper-described reimplementation may be useful, but the top-tier claim must be
phrased as comparison to that reimplementation plus the published GapSight
context, not to the original system itself.

## Consequence for the current paper route

The train decision remains determined only by the registered DECAR OOF gate.
The closest in-protocol test is whether DECAR beats `loss_only` at the identical
call count while satisfying utility, harm, source-coverage, and audit clauses.
A train pass would establish that realized signed task-effect and explicit harm
modeling add value over loss-gap-only routing within this fixed bank.  It would
not yet establish superiority to GapSight's continuous crop router.
