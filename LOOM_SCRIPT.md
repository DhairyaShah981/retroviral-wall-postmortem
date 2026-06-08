# Loom Walkthrough Script (8 minutes)

Target audience: Tanay Lohia, Mandrake Bio founder. He's an operator-founder, not a research-PhD. Lead with the audit-harness story, then back into the model.

**Pre-record setup:**
- Browser tab 1: https://github.com/DhairyaShah981/retroviral-wall-postmortem (README)
- Browser tab 2: results/05_audit_synthetic_summary.csv open in viewer
- Terminal: `cd ~/Documents/retroviral-wall-postmortem && source .venv/bin/activate`
- VS Code with `WRITEUP.md`, `mandrake_bench/audit.py`, `notebooks/05_audit_synthetic_perfect.py` open

**Suggested record style:** screen + small face cam. Tanay will be watching the screen, not you.

---

## 0:00 - 0:45 — Hook (the audit angle, not the score)

> "Hey Tanay — quick walkthrough. I spent a week on your Retroviral Wall challenge as a post-mortem, since Stage 1 closed in April. What I want to show you isn't really a model — it's the methodology-audit harness you'd want before sending Stage 2 candidates to your wet lab. The motivating observation is on screen: your Kaggle public leaderboard has six teams tied at exactly 1.00000 on a 57-sample LOFO problem. That's a tell, not a winning signal — and the Stage 2 wet-lab validation you bolted on exists precisely to sort that mess. I built the dry-side tool that would have done the sorting before wet-lab cost."

**Show:** browser tab on the Kaggle leaderboard, scroll to the 6 tied at 1.00000.

---

## 0:45 - 2:30 — The audit harness, demoed on synthetic submissions

> "Let me show you what I mean. The harness has four detectors. Here's the demo:"

**Show:** open `notebooks/05_audit_synthetic_perfect.py`, walk through the three synthetic submissions:

> "A — predicts the binary ground truth exactly. PR-AUC 1.0. Looks like a winner. But CLS collapses to 0.24 because Weighted Spearman dies — it can't rank within the active set, which is what your wet lab actually needs. The `class_rank_consistency` flag catches it.
>
> B — predicts 1 for every Retroviral RT, 0.5 for Retron and G2I, 0 for others. Family-memorisation attack. The `family_leakage` flag catches the zero within-family std.
>
> C — predicts pe_efficiency_pct exactly. The legitimate oracle. CLS 1.0, shuffle-null p < 2e-4 SIGNIFICANT, no false positives. The harness doesn't false-positive on real signal."

**Show:** terminal — `python notebooks/05_audit_synthetic_perfect.py`. Let the output run live, ~5 seconds.

> "If you ran this against the six teams tied at 1.0, my hypothesis is two are attack A, two are attack B, and maybe one is real. That sorting is what your team is doing manually right now with Stage 2 wet-lab results."

---

## 2:30 - 4:00 — The model + Mandrake reference comparison

**Show:** `results/04_summary.csv` and the WRITEUP.md headline numbers table.

> "On the actual model — vanilla isotonic-calibrated RF stack on your handcrafted features. CLS 0.544. Your published reference is 0.318. That's a 71% relative improvement using a 60-line Python file — no fancy architecture. The reason it beats the reference: CLS is harmonic-meaned across PR-AUC and Weighted Spearman, so you need both. Most participants seem to train one model end-to-end and miss that. My stack has two RFs — one classifier head, one regression head on pe_efficiency — isotonic-calibrated so their outputs are on the same scale, then geometric-meaned."

**Show:** terminal — `python mandrake-repo/evaluation/evaluate.py --predictions results/04_isotonic_stack_handonly_predictions.csv`. Run it live. The output will print:
```
PR-AUC:             0.6003
Weighted Spearman:  0.4993
CLS:                0.5452
```

> "That's your official evaluator, on my predictions. 0.545. I tried 25+ more variants — LightGBM, XGBoost, GP regression, family-balanced sampling, ESM-2 fine-tuned heads, multi-seed ensembles. Nothing beat the simple isotonic stack. That's the Hsu et al. 2022 result — at N=57, supervised + handcrafted dominates self-supervised + deep. I cite it in the writeup."

---

## 4:00 - 5:00 — The real wall: LTR_Retrotransposon, not Retroviral

**Show:** the per-family PR-AUC table in WRITEUP.md.

> "Two findings worth flagging at the dataset level. First: your challenge name is 'The Retroviral Wall', but in LOFO the Retroviral fold gives 0.83 PR-AUC across reasonable models — not the wall. LTR_Retrotransposon is the wall. PR-AUC 0.27 consistently, across classification, regression, and cross-family transfer triage. Both LTR actives have efficiencies up at 9-34% — they're not low-signal — but no model I tried generalizes to them."

**Show:** the sequence-identity probe results from notebook 06.

> "And here's why structurally — LTR_Retrotransposon has the lowest mean cross-family identity at 34.7%. Twilight-zone homology. When LTR is held out, training RTs in other families share only ~35% identity on average. Biophysical features just don't carry through that distance. So either the LTR wall isn't crackable from these features at all, or the signal lives in the Cas9-fusion interface, or RNA-template binding kinetics, somewhere outside the 66-D feature space."

---

## 5:00 - 6:00 — The other findings (ε bug + foldseek + active learning)

> "Three more findings the audit surfaced.
>
> One: your challenge page says ε = 0.1 for the Weighted Spearman weights. Your official evaluator code uses ε = 0.01. 10x discrepancy. The harness scores both so participants don't get bitten.
>
> Two: dropping the 10 foldseek_TM_* columns from the feature set — those are the explicit family-similarity features — actually hurts pooled CLS by 0.07. They carry real signal mixed with leakage. The right move isn't removal; it's a leakage-aware regularizer that uses them only insofar as they predict activity *orthogonal* to family. That's on the day-30 roadmap.
>
> Three: I built a cross-family transfer-triage selector — `(predicted_efficiency**2 × prediction_uncertainty)` for top-K candidate selection. Works for 3 of 4 active-containing families: Group_II_Intron +73pp capture vs random, Retron +26pp, Retroviral +7pp. LTR fails — minus 28pp — same wall. Not a selector bug; same feature-space failure."

---

## 6:00 - 7:00 — What I'd build at Mandrake

**Show:** the "What I'd build at Mandrake" section of WRITEUP.md.

> "The package I shipped is participant-facing. The internal version is what I'd want to build at Mandrake — four things:
>
> One, a Kaggle leaderboard auto-auditor that pulls submissions and ranks them by methodological soundness, not just by metric.
>
> Two, a dataset-card auto-generator that turns each new Open Problem dataset into a participant-facing README + leaky-feature list + suggested LOFO splits.
>
> Three, a Stage 2 regret-curve analyser — given participant Stage 1 predictions and Stage 2 wet-lab results, produce a capture-rate vs k curve per participant. The team you hire is the one with the steepest curve, not the highest pooled CLS.
>
> Four, a 'future problems' template that swaps the dataset and ships a new challenge in a week, not a month.
>
> Three weekends of work. Each future Open Problem becomes a one-week setup."

---

## 7:00 - 7:45 — Honest "what I won't claim"

> "Two things I'll be straight about. I'm not a senior protein-ML researcher — I joined ML LLM-first, don't have the pre-LLM loss-landscape intuition your Sr. AI Engineer JD asks for. What I bring is the engineer who builds the surface around the modeller's models. Voice-eval-harness for LLMs at Trifetch is the same shape. healthcare-skills npm package is the same shape. This bench is the same shape applied to your domain.
>
> Second: I couldn't crack LTR. I don't know if it's crackable from the features you provided. That's a real open question, not a marketing position."

---

## 7:45 - 8:00 — Close + call to action

> "All on GitHub — github.com/DhairyaShah981/retroviral-wall-postmortem. MIT licensed, pip-installable, ten unit tests passing against your own evaluator. Loom plus repo together is what I want to lead with on our call next week. Excited to talk."

---

## Recording tips

- **Pace:** read each section once, then record. Don't ad-lib — the script is tight on time.
- **Don't apologize for the score.** 0.544 above their reference 0.318 is real. Don't say "only 0.544."
- **Show the terminal running live** at least once — proof it works.
- **End the Loom on the GitHub URL** so the thumbnail captures it.
- **Aim for 7:30, not 8:00** — Tanay's watching at 1.5x, will appreciate brevity.

## After recording

1. Upload to Loom, get the share link
2. DM Tanay on X (`@tanaylohia` per the Mandrake research) — short, no signature, just hook + link:
   > "Hey Tanay — worked your Retroviral Wall challenge as a 7-day post-mortem. Built an audit harness that flags the 6 tied at 1.0 on Kaggle. CLS 0.545 above your reference 0.318. 8-min Loom + GitHub: [loom] / [github]. Looking forward to our call."
3. Mirror on LinkedIn DM in case X is buried.
4. Bring the WRITEUP.md printed (or PDF on tablet) to the call as backup.
