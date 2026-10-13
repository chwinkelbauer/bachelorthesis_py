# CLAUDE.md: Money from Scratch (Bachelor thesis, WU Wien)

Operating manual for every session on this project, including unattended overnight runs.
**The model, its mathematics, the metrics, parameters, presets and results live in `docs/FRAMEWORK.md`** (the living
specification). Read it before changing anything. This file covers how to work, the current state, and what to do next.

---

## 1. Project in brief
**Research question:** under which minimal conditions do learning agents with agent-specific (sparse Dirichlet log, Stone–Geary)
preferences *invent* money, and which condition causes it?

**Framework:**
- A MARL economy: GATv2 + goods-transformer policy, MAPPO, centralised critic, pure PyTorch on CPU.
- No good is labelled or rewarded as money. Money is identified functionally, symmetrically over goods, and against **two
  paired null models** (random and ZI-C) on identical economies.
- **Stages** (each is an economy type and a curriculum step):
  - **A:** pure exchange
  - **B:** lumpy raw-goods income + processing + consumption + subsistence/death
  - **C:** harvesting as a labour choice (only raw goods are harvested)
  - **D:** public trade ledger (F6, "money is memory")
- **Design principles:**
  - a frictionless baseline where money must not emerge
  - knock-in/knock-out of frictions F1–F6
  - change points to locate *when* money emerges
  - a modern framing (New Monetarist essentiality, Kocherlakota, Menger vs. Graeber)

## 2. Working rules (the user's explicit preferences: follow them)
1. **Act as a senior researcher in monetary agentic economics.**
   - Challenge proposals scientifically, including the user's own ideas and your own earlier fixes.
   - Example: the "costly only" indirect-exchange criterion was dropped because it would exclude Kiyotaki–Wright indirect exchange.
   - Say plainly when something does not work.
2. **Design before build:** present a concept, challenge it, and get explicit approval (AskUserQuestion) before larger changes:
   - economics, protocol, metrics/thresholds, experimental design, new stages
   - Small bug fixes and tooling do not need approval, but must be reported.
3. **`docs/FRAMEWORK.md` is updated in the same change as every code/model/parameter/preset/metric/design change:**
   - affected sections, the status table (§2), the results log (§12), and a dated changelog line (§13)
   - If code and document disagree, that is a bug.
4. **Organisation:**
   - Rewrite in place: no new parallel folders, and no copies of modules.
   - Keep `configs/ economy/ agents/ learning/ analysis/ scripts/ tests/ docs/`.
5. **Tests:** `python -m unittest discover -s tests -t .` must pass after every change (41 tests at the time of writing). Add tests for new behaviour.
6. **Scientific integrity:**
   - No good may be hard-coded or rewarded as money, and no BC/shaping toward trade or money.
   - Metrics stay symmetric over goods and null-referenced.
   - New parameters default to "off" so that old run configurations stay reproducible.
   - Report failures and confounds honestly (e.g. the stage-B learning trap).
7. **Communication:**
   - The user writes English (sometimes informal). Answer in English, concisely, with plain explanations.
   - For decisions, use AskUserQuestion with a recommended option first.
8. **Git:**
   - Interactive sessions: commit only when the user asks.
   - **Unattended sessions:** commit at milestones (a finished run with its FRAMEWORK update, a bug fix with tests) to a **work branch** `night/<date>` (create it from main at the start of the night). Never commit to main and never push. The user reviews and merges.

## 3. Environment and pitfalls (Windows 11, Git Bash + PowerShell)
- **Software:**
  - Python 3.14, torch 2.14 (CPU), numpy, networkx, pandas, matplotlib.
  - **Not installed:** scipy, pytest, yaml, ruptures. Use unittest, dataclass configs and the own implementations in `analysis/`.
- **Inline Python heredocs:** they sometimes fail in the Bash tool with `unexpected EOF while looking for matching '`, especially with triple quotes.
  - Write patch/diagnostic scripts with the Write tool into the scratchpad, then run `python <script> .`.
- **Background tasks of a session die when the session/editor restarts.** Start long runs **detached**:
  ```powershell
  Start-Process -FilePath python -ArgumentList "-u","-m","scripts.train","--preset","B","--seed","0" `
    -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "results\B_s0.log" `
    -RedirectStandardError "results\B_s0.err" -WindowStyle Hidden -PassThru
  ```
- **The PC can suspend during unattended nights** even with sleep set to "never". This laptop uses **Modern Standby** (S0 low power idle), which starts when the display turns off and pauses desktop processes. On 2026-10-12/13, it suspended the runs repeatedly between 23:02 and 02:37 (Kernel-Power events 506/507). `ES_SYSTEM_REQUIRED` alone did not prevent this, so the helper now also requests `ES_DISPLAY_REQUIRED` and the display stays on.
  - **Even this was not enough on 2026-10-13.** The PC had woken itself at 02:37 without user input. At 03:26, two seconds after Windows' automatic maintenance (drive optimisation) finished, it went back to standby until 08:59.
  - Note that Kernel-Power event 566 logs "Reason SleepButton" for *every* Modern Standby entry on this laptop, so it does **not** mean that the lid or a button was used.
  - The power plan turns the screen off after 30 min on AC (sleep: never).
  - **Reliable fix (user setting, do not change it yourself):** set "Turn off my screen after" (plugged in) to *Never* before nights. Remind the user if a night is planned.
  - At the start of every night, run the keep-awake helper (no power settings are changed; it ends at -Hours or when `results/.stop_keep_awake` exists):
    `Start-Process powershell -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts\keep_awake.ps1","-Hours","10" -WindowStyle Hidden`
  - A suspension shows up as one huge `seconds` jump in metrics.csv. It does not affect results.
- **Watching runs:** use Monitor (`tail -F log | grep --line-buffered`, which expires after at most 30 min; re-arm) or a background `until` loop. Check for `Traceback` in the `.err` file. Python processes: `tasklist //FI "IMAGENAME eq python.exe"`.
- **A running process keeps the code it loaded at start.** Editing files does not affect it. After code changes to metrics, re-run `scripts.evaluate` and `scripts.report` on finished runs.
- **Speed (12 threads):**
  - A ≈ 8 s/iteration alone (12–17 s when sharing the CPU); 300 iterations ≈ 40 min.
  - B/C ≈ 20–35 s/iteration (2–3 h for 300 iterations).
  - Run **at most 2 trainings concurrently**.
- **Resume:**
  - `python -m scripts.train --resume --run-dir results/<run>` continues from the last checkpoint (every 50 iterations + final), bit-identically, using the run's own `config.json`. `--iterations N` can extend a run.
  - Checkpoints are written atomically.
  - Old-format checkpoints (no RNG state) resume with a deterministic reseed.
- **Checkpoint compatibility:** the observation layout is unified across stages (public 8 / private 15 / decision token 21 / context 16), so warm starts A→B→C→D work. Checkpoints from before the unification (`results/pilot_A_s0_old_features`) do **not** load.
- **`results/` is git-ignored.** Naming:
  - `results/<preset>_s<seed>`
  - warm starts: `results/<preset>_from<Source>_s<seed>`
  - archived pilots: `results/pilot_*`
  - Logs: `results/<run>.log` / `.err` while running; move them into the run folder (`train.log` / `train.err`) when archiving.

## 4. Standard commands
```bash
python -m unittest discover -s tests -t .                                   # tests
python -m scripts.train --preset A --seed 0                                  # train (+ automatic evaluation at the end)
python -m scripts.train --preset B --seed 0 --init results/A_s0/checkpoint.pt --run-dir results/B_fromA_s0   # warm start
python -m scripts.train --preset C --seed 0 --set production.skill_sigma=1.0 # any parameter override
python -m scripts.train --resume --run-dir results/B_fromA_s0               # continue an interrupted run
python -m scripts.evaluate --run-dir results/A_s0                           # held-out eval vs. nulls + change points
python -m scripts.evaluate --run-dir results/D_s0 --set ledger.switch_round=12 --out evaluation_switch_off.json
python -m scripts.report --run-dir results/A_s0                             # report.png + JSON summary (look at the PNG!)
```
**Presets:** `A`, `A_complete`, `A_full_info`, `B`, `B_constant_income`, `C`, `C_no_specialization`, `D`, and `quick`, `quick_B`, `quick_C`, `quick_D` (smoke tests).

## 5. Current state (2026-10-12, 15:45)
- **Implemented:** stages A–D, stronger-null classification, descriptive costly-indirect-exchange metrics, the subsistence reserve (presets B–D: 10 rounds), resumable training, the report script. 41 tests pass.
- **Committed:** the rewrite (`5ee669d`, by the user) and the reserve / stronger-null / report / CLAUDE.md changes (committed to main on 2026-10-12, 15:50).
- **Pilot A** (`results/pilot_A_s0_old_features`, old features, finished):
  - efficient barter (efficiency vs. Walras 0.54; ZI-C 0.23) plus purposeful indirect exchange (resale 0.40 vs. stronger null 0.21)
  - **no money** (HHI 0.36 vs. 0.245 + 0.15 required)
  - change point at iteration ≈ 50
- **Pilot B** (`results/pilot_B_s0_no_reserve`, stopped at iteration 86): collapsed into autarky.
  - 89% of agents needed others' goods to eat, held nothing at the start, and died by round ≈ 4 before trade could be learned.
  - Fix: the starting reserve plus the A→B warm start (approved).
- **A_s0** (unified features, the curriculum base) is **finished and evaluated**: efficiency 0.55, resale 0.43 vs. stronger null 0.21, costly resale 0.44, HHI 0.30 vs. 0.245, giving **indirect exchange, no money** (FRAMEWORK §12). Its training log uses the random-only null.
- **Night 1 (2026-10-12/13, branch `night/2026-10-12`, not merged):** summary in FRAMEWORK §12.
  - A_s1 done (indirect exchange, near the money threshold).
  - B_s0 collapsed (autarky).
  - B_fromA_s0 failed the gate (survival ≈ 0.2) and is being resumed to 300 as a trend check.
  - A_s2 running.
  - **Blocked on the user's decision in §8** (stage-B survival / intermediation).
- **Next:** finish and evaluate A_s2 and B_fromA_s0; A_s3/A_s4; extend A_s1 to 600 (item 10). Do not start C/D before the §8 decision.

## 6. Unattended / overnight protocol
**Pre-approved without asking:**
- running the experiment queue (§7) in order
- evaluating and reporting finished runs
- updating the FRAMEWORK results log/changelog
- resuming interrupted runs
- stopping a run that has clearly failed a gate (document why)
- fixing bugs that do not change economics, protocol or metric definitions (with tests and changelog)

**Not pre-approved:** any change to economics, protocol, metric definitions/thresholds, presets' experimental content, or new conditions. Write these as proposals into §8 "Pending decisions" (diagnosis, evidence, options, recommendation) and continue with the next queue item that does not depend on them.

**Loop:**
1. Keep up to 2 detached runs going.
2. Check every 20–30 min (ScheduleWakeup / Monitor). Never sit idle while the queue has runnable items.
3. On completion:
   - `scripts.evaluate` (current code) and `scripts.report`
   - **look at the PNG**
   - check the gates
   - add a results-log entry to FRAMEWORK §12
   - tick the queue item
   - launch the next item

**Gates** (checked at iteration ≈ 150 and at the end of B/C runs):
- survival_rate < 0.5, or trade_rate < 0.05 (autarky collapse), or NaN/exploding losses ⇒ stop dependent follow-ups (e.g. C from that B).
- Then diagnose (survival by agent group, consumption vs. γ, labour shares; see the pilot-B diagnostic idea in FRAMEWORK §11) and record it in §8.

**End of the night:** write a concise summary (what ran, key numbers vs. nulls, stages, change points, failures, pending decisions) as the final message and into FRAMEWORK §12.

## 7. Experiment queue (ordered; tick items and add the result path)
- [x] 1. A_s0 finished at 16:22 and was re-evaluated against the stronger null: indirect exchange, no money (FRAMEWORK §12). Logs are in `results/A_s0/train.log`; the old evaluation is kept as `evaluation_random_null_only.json`.
- [x] 2. **B warm from A**: `results/B_fromA_s0`. Gate failed at iteration 150 (survival 0.23). Evaluated: barter; +23% welfare from trade vs. no trade; survival still rising. It is resumed to 300 as a trend check for option (c) in §8.
- [x] 3. **B from scratch**: `results/B_s0`. Stopped at iteration 69 after failing the gate (autarky collapse, survival 10–15%). Diagnosis in §8 / FRAMEWORK §11.
- [x] 4. Gate check: **both failed**. The B/C/D chain (items 5, 6, 8 and the B/C controls in 7) is **blocked until the user decides on the stage-B proposal in §8**.
- [ ] 5. **C warm from B:** `--preset C --seed 0 --init results/B_fromA_s0/checkpoint.pt --run-dir results/C_fromB_s0`
- [ ] 6. **C from scratch:** `--preset C --seed 0` → `results/C_s0`
- [ ] 7. **Controls:**
  - `B_constant_income` warm from A → `results/B_constant_income_fromA_s0` (F5b)
  - `C_no_specialization` warm from B → `results/C_no_specialization_fromB_s0` (F2)
- [ ] 8. **D warm from C:** `--preset D --seed 0 --init results/C_fromB_s0/checkpoint.pt --run-dir results/D_fromC_s0`. Then the disturbance evaluations:
  - D with `--set ledger.switch_round=12 --out evaluation_switch_off.json`
  - C_fromB with `--set ledger.enabled=true ledger.switch_round=12 --out evaluation_switch_on.json`
- [ ] 9. **More seeds for statistics** (A seeds 1–4, then B_fromA seeds 1–2), as time allows. *Progress:* A_s1 and A_s2 done (indirect exchange; transient money episodes; cross-seed table in FRAMEWORK §12); A_s3 running (started 09:37); A_s4 next. The B_fromA seeds are blocked (§8).
- [ ] 10. *(added in night 1, run-length check, no design change)* Extend A_s1 to 600 iterations (`--resume --iterations 600 --run-dir results/A_s1`), because its HHI was still rising at iteration 300. If it crosses the threshold, also extend A_s0. Aggregate with `analysis/stats.py` (IQM + bootstrap CI).

## 8. Pending decisions (for the user; add proposals here)
- *(decided 2026-10-12)* Commit policy for unattended work: work branch `night/<date>`, no push (§2.8).
- **[NEW 2026-10-13, needs decision] Stage-B survival requires intermediation; learning cannot discover it before agents die.**
  - **Evidence:**
    - B runs at iteration 50: survival 14% (warm-started from A) and 10% (scratch). The scratch run collapsed into autarky by iteration 50–60. The 89% of agents that depend on others' goods survive 3.5% / 0%.
    - 76–85% of deprived rounds happen because the agent holds less than γ, i.e. it has run out.
    - A hand-coded sensible barter policy also only reaches 22% survival (12% for dependents). The results do *not* improve with a 4× supply surplus, 4 trade sessions per round, a death threshold of 10, or a money convention (the most durable good accepted by all). They do improve with raw-only subsistence goods (55%) and a complete network (63%).
    - Flow analysis (heuristic, raw-only variant): dependents receive more than they need *in aggregate* (2817 vs. 1797 units), but only 56% ever receive any. Many have no neighbour who earns their subsistence good, so survival requires **intermediation along network chains**.
  - **Interpretation:** survival is gated by exactly the coordination problem (indirect exchange through intermediaries) that money solves. That is scientifically ideal, but a learning agent dies before it can explore it, so the gradient pushes toward autarky. Not a bug.
  - **Options:**
    - (a) **Mortality curriculum:** death off (subsistence only as a utility kink) for the first ~100–150 iterations, then on. Agents can then learn trade and intermediation first.
    - (b) **Subsistence only from raw goods** (`preferences.subsistence_max_tier=0`, `goods.raw_consumable_prob=1.0`): shorter supply chains; heuristic survival 22% → 55%.
    - (c) Longer training (e.g. 1000 iterations). Uncertain; the scratch run already collapsed.
    - (d) Guarantee every agent a neighbour who earns its subsistence good. **Not recommended:** it removes the intermediation need that makes the economy interesting.
  - **Recommendation:** (a) + (b). Both are economics/curriculum changes and need approval. Diagnostic scripts used (scratchpad, not in the repo): survival by agent group, deprivation causes, heuristic barter/money policies, calibration variants.
- **Confirmatory design:** number of seeds (≥ 5 planned), and which knock-in/knock-out conditions and dose–response sweeps to run (FRAMEWORK §6.2/6.3). Needs a proposal and approval.
- **Final stage thresholds** (`MetricsConfig`) must be fixed before confirmatory runs. They are currently provisional.
- **Planned extensions (need approval):**
  - GRU agent memory, needed to separate private memory from the public ledger in F6
  - a production-economy benchmark (dynamic planner)
  - CES robustness
  - a multi-seed sweep runner script

## 9. Key literature (for correct framing)
- Kiyotaki & Wright 1989
- Marimon, McGrattan & Sargent 1990 (AI agents in a KW economy)
- Duffy & Ochs 1999
- Kocherlakota 1998 ("money is memory")
- Lagos & Wright 2005; Wallace (essentiality)
- Menger 1892 (saleability)
- Jevons 1875 (double coincidence)
- Feldman 1973; Ostroy & Starr 1974/1990 (transaction role of money in bilateral trade)
- Graeber 2011 (credit before money)
- Gode & Sunder 1993 (ZI-C)
- Yu et al. 2022 (MAPPO)
- Brody et al. 2022 (GATv2)
- Agarwal et al. 2021 (IQM / bootstrap)
- Zheng et al. 2022 (AI Economist, built-in money)
- Johanson et al. 2022 (emergent bartering in MARL)
- Pardo et al. 2018 (time-limit truncation)
