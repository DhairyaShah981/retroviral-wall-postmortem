# Tanay Call Prep — Likely Questions + Answers

> Based on review-agent risk register. Memorize the bolded one-liner per question; the supporting detail is backup.

---

## 1. "What's your Stage 2 number?"

**One-liner:** *"I don't have one — Stage 2 access is yours. But my Stage-1 model is the same one that would go to Stage 2. My bigger claim is the audit harness: without it, you can't trust your Stage 2 winner either, because the Stage 1 leaderboard is six teams tied at 1.0."*

The pitch this opens: Stage 2 is your real evaluation precisely because Stage 1's metric is gameable. My bench is the dry-side tool that would have sorted the 1.0 cohort *before* the wet lab tells you who actually won. That's the same job, done one step earlier and cheaper.

---

## 2. "Why didn't you just use ProteinMPNN like David Liu's group?"

(They're referring to *"AI-guided redesign of laboratory-evolved reverse transcriptases enhances prime editing"* — Nature Biotechnology 2026, s41587-026-03149-6, from the same group that produced the dataset.)

**One-liner:** *"Inverse folding redesigns an active RT — it's an engineering tool, not an activity classifier. The Retroviral Wall asks the inverse question: given a candidate, will it work. ProteinMPNN log-likelihoods over the YXDD catalytic motif could be a strong feature for classification too — that's on my day-30 roadmap if you want me building this layer."*

Don't get caught not knowing the paper. They redesigned PE8/PEmax with 30-163 AA substitutions, got 2× protein levels, up to 2.9× efficiency in mice. The point is *engineering* (redesign), not *prediction* (classification). Different problem, complementary tool.

---

## 3. "What's the per-family confusion?"

**One-liner:** *"Retroviral isn't the wall. LTR_Retrotransposon is. Retroviral held-out PR-AUC is 0.83. LTR's is 0.27, and that's consistent across every model I tried — classification, regression, active-learning capture rate (-28pp vs random). The handcrafted features don't carry LTR-activity signal in any regime."*

Have the table memorized:

```
Retroviral       PR-AUC 0.83  (held out, RF + isotonic stack)
Retron           PR-AUC 0.61
Group_II_Intron  PR-AUC 1.00  (n=5, 2 active, easy)
LTR              PR-AUC 0.27  ← the wall
CRISPR/Other/Unclassified — 0 active, N/A
```

---

## 4. "Why RF over a fine-tuned ESM head?"

**One-liner:** *"Hsu et al. 2022 — at N≤100 with assay labels, ridge or RF on one-hot site features + one density feature beats deep models. My ablations confirm it: LightGBM at 200 trees gets CLS 0.30, vanilla RF gets 0.52. The dataset is in the regime where simple+supervised dominates self-supervised."*

Reference: Hsu et al., *"Learning protein fitness models from evolutionary and assay-labeled data,"* PMID 35039677. This is the small-N protein ML standard citation. Use it once and they'll trust you read.

---

## 5. "What would you do with [compute / months]?"

**One-liner:** *"Four things, in order: (1) supervised contrastive ESM fine-tune with family-balanced batches — CLEAN-style debiasing; (2) ProteinMPNN per-residue log-likelihoods over the YXDD motif as features; (3) sequence-identity proximity probe baked into the audit harness; (4) wet-lab Stage-2 feedback wired into the candidate selector so the bench learns from your bench."*

Reference for (1): CLEAN (Yu et al., Science 2023, doi:10.1126/science.adf2465) — supervised contrastive learning on EC numbers; positives = same function, negatives = different function. The principled version of the ESM-PCA-orthogonal hack I tried. With family as the negative dimension, it would directly attack the wall.

---

## 6. "Is your bench actually reusable?"

**One-liner:** *"It's a 7-day prototype with the right shape, not a packaged release. Week 2 work: harden the API surface, add tests against the official evaluator on adversarial inputs, add the sequence-identity proximity probe, write the per-problem dataset-card auto-generator. That turns each future Open Problem into a one-week setup instead of multi-week."*

Don't overclaim. The repo is real (`pip install -e .` works), but it's a prototype.

---

## 7. "What's your Kaggle rank?"

**One-liner if didn't submit:** *"I didn't submit. Top of the public leaderboard is 1.00000 × 6, and my own audit harness would flag at least the binary-classifier and family-constant attacks before they hit Stage 2. I'm not optimising for that leaderboard — I'm optimising for what the leaderboard fails to capture."*

**If pressed:** *"I can submit the isotonic stack tonight — but the relevant comparison is which of the 1.0 cohort survives the audit, not which model lands on top of a metric that's already saturated."*

---

## Sharpest single risk (per reviewer): the harness pitch only lands if Tanay believes Mandrake **needs** the harness.

If Tanay says "we already have an internal harness," counter with:
- **The ε = 0.01 vs 0.1 discrepancy** between the page and the official evaluator. This is concrete evidence that whatever internal harness exists is not catching everything.
- **Six teams tied at 1.0.** Whatever internal harness sorted these, it sorted them wrong (or didn't sort them at all — that's why you bolted Stage 2 on).
- **The framing isn't "your engineers built a bad harness."** The framing is: *participants need a harness too, and right now they're scoring against an inconsistent spec, which is why you get tied-at-1.0.* This positions the bench as the missing participant-facing surface, not a replacement for Mandrake's internal one.

---

## What NOT to say

- Don't pretend to be a senior protein-ML researcher. Tanay's JD says 5-7 years pre-LLM; you're not. Own it: *"I'm a founding engineer who turns ML into shipped tools. I'm not the modeller you'd hire for the foundation-model lead — I'm the engineer who builds the surface around your modeller's models."*
- Don't lead with the 0.544 CLS score. It's real, but it doesn't differentiate from the 1.0 cohort visually.
- Don't overclaim "my structural features didn't work means custom structural features can't work." Say: *"my 12 family-agnostic features are subsumed by the existing handcrafted set; a richer featurization — ProteinMPNN motif log-likelihoods, Foldseek 3Di tokens — might yet help."*
- Don't oversell the "active learning works on 3 of 4 families." It's a cross-family transfer triage demo, not an acquire-fit-acquire loop. Be precise.

---

## Five things to bring up unprompted

1. **The ε discrepancy** between page and code. Concrete, technical, immediately useful.
2. **The LTR wall finding** — your challenge name foregrounds Retroviral, but LTR is where every model breaks. That's a dataset-level finding, not a model-level one.
3. **The audit harness vs the 1.0-tied cohort.** Lead the demo with the synthetic-submission audit: A (oracle binary) → CLS 0.24, flagged classifier-only. That's the slide.
4. **The "what I'd build day-30" roadmap.** Have a one-sentence answer ready (see Q5 above).
5. **Honest negative: I couldn't crack LTR.** Don't hide it. *"The biophysics features in your dataset don't carry LTR-activity signal in any regime I tried. Real signal must be sequence-context-dependent or in the Cas9-fusion interface. That's an open question I'd want to work on with you."*
