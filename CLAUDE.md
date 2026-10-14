# CLAUDE.md: Money from Scratch (Bachelor thesis, WU Wien)

Operating manual for every session on this project, including unattended overnight runs.
**The model, its mathematics, the metrics, parameters, presets and results live in `docs/FRAMEWORK.md`** (the living
specification). Read it before changing anything. **`docs/HOW_IT_WORKS.md`** is the user's detailed plain-language
explanation of every mechanic and of the training; keep it in sync (§2.3). This file covers how to work, the current state, and what to do next.

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
3. **`docs/HOW_IT_WORKS.md` (the user's detailed explanation of everything) is updated in the same change as every change of the mechanics or of the optimisation/training settings** (user request 2026-10-14): the affected sections plus a line in its change log (§20). It explains in plain language, FRAMEWORK specifies; both must match the code.
4. **`docs/FRAMEWORK.md` is updated in the same change as every code/model/parameter/preset/metric/design change:**
   - affected sections, the status table (§2), the results log (§12), and a dated changelog line (§13)
   - If code and document disagree, that is a bug.
5. **Organisation:**
   - Rewrite in place: no new parallel folders, and no copies of modules.
   - Keep `configs/ economy/ agents/ learning/ analysis/ scripts/ tests/ docs/`.
6. **Tests:** `python -m unittest discover -s tests -t .` must pass after every change (65 tests at the time of writing). Add tests for new behaviour.
7. **Scientific integrity:**
   - No good may be hard-coded or rewarded as money, and no BC/shaping toward trade or money.
   - Metrics stay symmetric over goods and null-referenced.
   - New parameters default to "off" so that old run configurations stay reproducible.
   - Report failures and confounds honestly (e.g. the stage-B learning trap).
8. **Communication:**
   - The user writes English (sometimes informal). Answer in English, concisely, with plain explanations.
   - For decisions, use AskUserQuestion with a recommended option first.
9. **Git:**
   - Interactive sessions: commit only when the user asks.
   - **Unattended sessions:** commit at milestones (a finished run with its FRAMEWORK update, a bug fix with tests) to a **work branch** `night/<date>` (create it from main at the start of the night). Never commit to main and never push. The user reviews and merges.

## 3. Environment and pitfalls (Windows 11, Git Bash + PowerShell)
- **Software:**
  - Python 3.14, torch 2.14 (CPU), numpy, networkx, pandas, matplotlib.
  - **Not installed:** scipy, pytest, yaml, ruptures. Use unittest, dataclass configs and the own implementations in `analysis/`.
- **PowerShell aliases win over functions:** never name a helper function like a built-in alias (`ac` = Add-Content, `sc`, `cat`, `ls`, `cd`, `diff`, `sort`, …). On 2026-10-13 a helper `ac` wrote the stray files `SUB_SLEEP` / `SUB_VIDEO` into the repo (removed 2026-10-14).
- **Queue files use forward slashes:** `scripts/run_queue.py` splits commands like a POSIX shell and rejects backslashes (a backslash path once became a drive-relative file in the project root).
- **Inline Python heredocs:** they sometimes fail in the Bash tool with `unexpected EOF while looking for matching '`, especially with triple quotes.
  - Write patch/diagnostic scripts with the Write tool into the scratchpad, then run `python <script> .`.
- **Background tasks of a session die when the session/editor restarts.** Start long runs **detached**:
  ```powershell
  Start-Process -FilePath python -ArgumentList "-u","-m","scripts.train","--preset","B","--seed","0" `
    -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "results\B_s0.log" `
    -RedirectStandardError "results\B_s0.err" -WindowStyle Hidden -PassThru
  ```
- **The PC can suspend during unattended nights** even with sleep set to "never". This laptop uses **Modern Standby** (S0 low power idle), which starts when the display turns off and pauses desktop processes. On 2026-10-12/13, it suspended the runs repeatedly between 23:02 and 02:37 (Kernel-Power events 506/507). A keep-awake request with `ES_SYSTEM_REQUIRED` alone did not prevent this; adding `ES_DISPLAY_REQUIRED` (display stays on) helped.
  - **Even this was not enough on 2026-10-13.** The PC had woken itself at 02:37 without user input. At 03:26, two seconds after Windows' automatic maintenance (drive optimisation) finished, it went back to standby until 08:59.
  - Note that Kernel-Power event 566 logs "Reason SleepButton" for *every* Modern Standby entry on this laptop, so it does **not** mean that the lid or a button was used.
  - The power plan turns the screen off after 30 min on AC (sleep: never).
  - **The user wants a dark screen at night**, so the screen timeout stays as it is. Instead, use the system tool **Dark Awake** (outside the repo): `C:\Users\cwink\Tools\dark-awake.ps1`, started with the hotkey **Ctrl+Alt+D** (Start-menu shortcut "Dark Awake"). It blacks out the screen (brightness 0 + black overlay) while the display stays logically on, so Modern Standby cannot suspend the runs. It runs until any key or click, then restores the brightness. Only one instance runs at a time.
    - At the start of a night, ask or remind the user to press **Ctrl+Alt+D** before leaving.
    - If they did not, the session may start it itself (this blacks out the screen): `Start-Process powershell -ArgumentList "-NoProfile","-WindowStyle","Hidden","-ExecutionPolicy","Bypass","-File","C:\Users\cwink\Tools\dark-awake.ps1"`
  - At every wake-up, check `Get-WinEvent` (System log, Kernel-Power events 506/507) since the last check, so a standby is noticed immediately.
- **Watching runs:** use Monitor (`tail -F log | grep --line-buffered`, which expires after at most 30 min; re-arm) or a background `until` loop. Check for `Traceback` in the `.err` file. Python processes: `tasklist //FI "IMAGENAME eq python.exe"`.
- **A running process keeps the code it loaded at start.** Editing files does not affect it. After code changes to metrics, re-run `scripts.evaluate` and `scripts.report` on finished runs.
- **Speed (12 threads):**
  - A ≈ 8 s/iteration alone (12–17 s when sharing the CPU); 300 iterations ≈ 40 min.
  - B/C without newcomers ≈ 20–35 s/iteration alone. Two B runs with learning sessions in parallel (B2, measured): 62 / 49 / 35 s per iteration in sessions 1 / 2 / 3, i.e. 3.8 h per 300 iterations including evaluation.
  - **With newcomers (respawn, presets B–D since 2026-10-14) the full economy is ≈ 2× slower** (B: 16 → 32 s alone; B_rare_perishable: 23 → 42 s), because the market stays full. Estimate for two parallel runs: B ≈ 4.8 h, B_rare_perishable ≈ 6.2 h per 300 iterations.
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
python -m scripts.summarize --out results/summary.md                        # all runs in one table + IQM/CI across seeds
python -m scripts.train --preset B --seed 0 --set curriculum.sessions=[]    # control: no learning sessions
python -m scripts.run_queue configs/night_queue.txt --parallel 2            # unattended job queue (start detached, see §6)
```
**Presets:** `A`, `A_complete`, `A_full_info`, `B`, `B_constant_income`, `B_perishable`, `B_rare`, `B_rare_perishable`, `C`, `C_no_specialization`, `D`, and `quick`, `quick_B`, `quick_C`, `quick_D` (smoke tests).

## 5. Current state (2026-10-14, 12:30): nothing is running
- **Implemented:** stages A–D (code and tests; 65 tests pass), stronger-null classification, costly-indirect-exchange indicators, the subsistence reserve (presets B–D: 10 rounds), resumable training, `scripts/report.py`, `scripts/summarize.py`.
- **New (2026-10-13, approved stage-B redesign, not yet trained):**
  - **Learning sessions** (`curriculum.sessions`, presets B–D): S1 iterations 1–75 raw goods only and death after 20; S2 76–150 tiers ≤ 1 and death after 10; S3 the full economy. Only the full economy is evaluated.
  - **Basic needs vs. higher wants:** subsistence only from raw goods (all raw goods consumable); 2–5 valued goods per agent; random log-normal tier multipliers (means 0.2/1/2, log-spreads 0.25/0.75/1.0). Calibration check in FRAMEWORK §3.2. `preferences.min_processed_valued` (≥ k processed valued goods) stays in the code but is **off** in the presets (user decision).
  - **Death penalty** `market.death_penalty` (one-off −p at death, default 0). **Presets B–D: p = 0.5** (user: "always have a penalty for now"), although in the B2 A/B test it led to late hoarding.
  - **Entropy coefficient 0.03** in presets B–D (was 0.01; entropy collapse accompanied the trade drop).
  - **Treatment preset `B_perishable`** (spoilage U[0, 0.25], F4).
  - **`scripts/run_queue.py` + `configs/night_queue.txt`** for unattended nights.
  - All new options default to off; old configurations sample bit-identically.
- **Git:** `main` up to `ef6a91a` (night branch merged). **Do not commit:** the user commits everything to main themselves (said 2026-10-13). Today's changes are uncommitted.
- **Results** (details in FRAMEWORK §12 "Current status"; table: `python -m scripts.summarize`):
  - **Stage A** (seeds 0–2 finished, seed 3 stopped at checkpoint 100):
    - efficient barter + purposeful indirect exchange: efficiency 0.54 vs. ZI-C 0.23; resale 0.39 vs. stronger null 0.21
    - **no stable money:** HHI excess +0.06 to +0.14, below +0.15; money conventions appear transiently late in training
  - **Stage B:**
    - from scratch, it collapses into autarky
    - warm-started from A: barter with +32% welfare from trade, but survival is only 0.25 (gate 0.5), rising slowly
    - diagnosis: survival requires intermediation (FRAMEWORK §11)
  - **C and D:** not run.
- **B2 results (redesign, 2026-10-13):** without penalty survival 0.46 (greedy 0.50) vs. 0.25 old design, still rising; barter, no money. Death penalty 0.5: late hoarding, survival 0.38, so it is not adopted (FRAMEWORK §12).
- **B2 diagnosis** (FRAMEWORK §11): exploration collapse, eating mistakes (31–42% of deprived rounds while holding enough food), agents accept unusable goods only 3% of the time.
- **Night 2 results (B3 / B3_perishable, seeds 0–1, FRAMEWORK §12):** barter everywhere, no money. B3 survival 0.51 / 0.59 passes the gate only nominally: agents **game the death rule** (feast-and-fast: eat in lumps, fast up to 4 rounds). Entropy 0.03 fixed the exploration collapse. Perishability did not create store-of-value holdings (survival 0.33 / 0.35).
- **Decided 2026-10-14:** continuous subsistence utility (implemented, presets B–D), night 3 = 2×2 screening F1 × F4 with F1 = 15 goods / 2–3 valued (presets `B_rare`, `B_rare_perishable`). **Agent memory implemented** (`--set model.memory=true`; per-good GRU + own trade experience; FRAMEWORK §4.2), off by default; planned as the night-4 treatment on the best night-3 condition.
- **Decided 2026-10-14 (afternoon):** death mechanics revised: **hunger debt** (cumulative missing food, partial meals count partially) and **newcomers** refill slots empty for 2 rounds (same position/income, new preferences, no history). Both on in presets B–D (FRAMEWORK §3.4). Survival now means: share of the round-0 agents alive at the end; new metrics births, death_rate, population_share.
- **Next: night 3** = `configs/night_queue.txt`: only the two corners `B4_fromA_s0` and `B4_rare_perishable_fromA_s0` (~6.5 h; newcomers make iterations ≈ 2× slower); `B4_rare` and `B4_perishable` are commented out for night 4. **Start only on the user's command** (start procedure §6).
- **Night runs:** the user starts the system tool **Dark Awake** (Ctrl+Alt+D) before leaving (§3).

## 6. Unattended / overnight protocol
**Pre-approved without asking:**
- running the experiment queue (§7) in order
- evaluating and reporting finished runs
- updating the FRAMEWORK results log/changelog
- resuming interrupted runs
- stopping a run that has clearly failed a gate (document why)
- fixing bugs that do not change economics, protocol or metric definitions (with tests and changelog)

**Not pre-approved:** any change to economics, protocol, metric definitions/thresholds, presets' experimental content, or new conditions. Write these as proposals into §8 "Pending decisions" (diagnosis, evidence, options, recommendation) and continue with the next queue item that does not depend on them.

**Start procedure (when the user gives the command):**
1. `git status` / `tasklist` check: no python training running; `results/queue.stop` absent; `configs/night_queue.txt` up to date.
2. Launch the queue runner detached:
   ```powershell
   Start-Process -FilePath python -ArgumentList "-u","-m","scripts.run_queue","configs/night_queue.txt","--parallel","2" `
     -WorkingDirectory (Get-Location).Path -RedirectStandardOutput "results\queue.out" -RedirectStandardError "results\queue.err" -WindowStyle Hidden -PassThru
   ```
3. Confirm both first jobs started (`results/queue.log`, `results/<job>.log`), then remind the user of Ctrl+Alt+D (Dark Awake).
4. Monitoring (ScheduleWakeup ~30 min) is optional: the queue runs on its own; to stop it gracefully create `results/queue.stop` (running jobs finish).

**Loop:**
1. Keep up to 2 detached runs going (the queue runner does this).
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
- [x] 2. **B warm from A**: `results/B_fromA_s0`. Gate failed at iteration 150 (survival 0.23). Resumed and finished at 300: barter; +32% flow utility vs. no trade; survival 0.25, rising only ~+0.05 per 100 iterations.
- [x] 3. **B from scratch**: `results/B_s0`. Stopped at iteration 69 after failing the gate (autarky collapse, survival 10–15%). Diagnosis in §8 / FRAMEWORK §11.
- [x] 4. Gate check: **both failed**. The B/C/D chain (items 5, 6, 8 and the B/C controls in 7) is **blocked until the user decides on the stage-B proposal in §8**.
- [x] 4b. *(new B design, 2026-10-13)* **B with learning sessions, warm from A**, two runs in parallel. **Done:** no penalty survival 0.46 (better), penalty 0.5 survival 0.38 (hoarding). Gate 0.5 not reached yet (greedy 0.50). Logs in the run folders.
  - `--preset B --seed 0 --init results/A_s0/checkpoint.pt --run-dir results/B2_fromA_s0` (no death penalty)
  - the same with `--set market.death_penalty=0.5 --run-dir results/B2_penalty_fromA_s0`
  - Then the control without sessions (`--set curriculum.sessions=[]`, no penalty): `results/B2_nosessions_fromA_s0`. Not started.
  - Proposed: extend `B2_fromA_s0` to 450 iterations (`--resume --iterations 450 --run-dir results/B2_fromA_s0`; move its train.log/err aside first). Gates at iteration 150 (end of session 2) and at the end; judge survival in session 3 only (sessions 1–2 have a softer death rule). Old B runs (`B_fromA_s0`, `B_s0`) used the old design. Items 5–8 use the new B checkpoint (`results/B2_fromA_s0`).
- [x] 4c. **Night 2** (`configs/night_queue.txt`, 2026-10-13 21:01 – 2026-10-14 03:32, no failures). Results: FRAMEWORK §12 (barter, no money; feast-and-fast gaming of the death rule). Jobs: `B3_fromA_s0`, `B3_perishable_fromA_s0`, then `B3_fromA_s1`, `B3_perishable_fromA_s1` (warm from A_s1), then `summary`. Gates as in §6, judged on session 3. Afterwards: compare B3 vs. B3_perishable (pass-through and HHI excess over the null, survival, trade-rate trend) and B3 vs. B2 (entropy + penalty effect).
- [ ] 4d. **Night 3** (`configs/night_queue.txt`, prepared 2026-10-14, not started): corners of the 2×2 screening, `B4_fromA_s0` and `B4_rare_perishable_fromA_s0`, then `summary`. Night 4: uncomment `B4_rare_fromA_s0` and `B4_perishable_fromA_s0` (the other two arms). Afterwards: money metrics per condition (pass-through and HHI excess over the null, acceptance of unusable goods, durability of held no-use goods), survival in session 3, feast-and-fast check (should be gone); replicate the best condition with seeds 1–2.
- [ ] 5. **C warm from B:** `--preset C --seed 0 --init results/B2_fromA_s0/checkpoint.pt --run-dir results/C_fromB_s0`
- [ ] 6. **C from scratch:** `--preset C --seed 0` → `results/C_s0`
- [ ] 7. **Controls:**
  - `B_constant_income` warm from A → `results/B_constant_income_fromA_s0` (F5b)
  - `C_no_specialization` warm from B → `results/C_no_specialization_fromB_s0` (F2)
- [ ] 8. **D warm from C:** `--preset D --seed 0 --init results/C_fromB_s0/checkpoint.pt --run-dir results/D_fromC_s0`. Then the disturbance evaluations:
  - D with `--set ledger.switch_round=12 --out evaluation_switch_off.json`
  - C_fromB with `--set ledger.enabled=true ledger.switch_round=12 --out evaluation_switch_on.json`
- [ ] 9. **More seeds for statistics** (A seeds 1–4, then B_fromA seeds 1–2), as time allows. *Progress:* A_s1 and A_s2 done (cross-seed table in FRAMEWORK §12). **A_s3 stopped at its iteration-100 checkpoint:** continue it with `--resume --run-dir results/A_s3` (move its partial logs aside first; they are kept as `train_part1.*`). A_s4 not started. The B_fromA seeds are blocked (§8).
- [ ] 10. *(added in night 1, run-length check, no design change)* Extend A_s1 to 600 iterations (`--resume --iterations 600 --run-dir results/A_s1`), because its HHI was still rising at iteration 300. If it crosses the threshold, also extend A_s0. Aggregate with `analysis/stats.py` (IQM + bootstrap CI).

## 8. Pending decisions (for the user; add proposals here)
- *(decided 2026-10-12)* Commit policy for unattended work: work branch `night/<date>`, no push (§2.9).
- **[DECIDED 2026-10-13] Stage-B survival requires intermediation; learning cannot discover it before agents die.** The user chose learning sessions (technology unlock + death ramp) and raw-only subsistence with low-utility basic goods. Agents also value 2–5 goods (variable), with random high-variance weights on processed goods (FRAMEWORK §3.2, §6.1). The original analysis is kept below for the record.
  - **Evidence:**
    - B runs at iteration 50: survival 14% (warm-started from A) and 10% (scratch). The scratch run collapsed into autarky by iteration 50–60. The 89% of agents that depend on others' goods survive 3.5% / 0%.
    - 76–85% of deprived rounds happen because the agent holds less than γ, i.e. it has run out.
    - A hand-coded sensible barter policy also only reaches 22% survival (12% for dependents). The results do *not* improve with a 4× supply surplus, 4 trade sessions per round, a death threshold of 10, or a money convention (the most durable good accepted by all). They do improve with raw-only subsistence goods (55%) and a complete network (63%).
    - Flow analysis (heuristic, raw-only variant): dependents receive more than they need *in aggregate* (2817 vs. 1797 units), but only 56% ever receive any. Many have no neighbour who earns their subsistence good, so survival requires **intermediation along network chains**.
  - **Interpretation:** survival is gated by exactly the coordination problem (indirect exchange through intermediaries) that money solves. That is scientifically ideal, but a learning agent dies before it can explore it, so the gradient pushes toward autarky. Not a bug.
  - **Options:**
    - (a) **Mortality curriculum:** death off (subsistence only as a utility kink) for the first ~100–150 iterations, then on. Agents can then learn trade and intermediation first.
    - (b) **Subsistence only from raw goods** (`preferences.subsistence_max_tier=0`, `goods.raw_consumable_prob=1.0`): shorter supply chains; heuristic survival 22% → 55%.
    - (c) Longer training. **Tested on B_fromA_s0 to 300 iterations:** survival rises steadily but slowly (0.10 → 0.26, ~+0.05 per 100 iterations). Reaching 0.5 would take roughly 800 iterations if the trend held (≈ 8 h per seed). Possible, but impractical as the main fix; it is useful as a complement.
    - (d) Guarantee every agent a neighbour who earns its subsistence good. **Not recommended:** it removes the intermediation need that makes the economy interesting.
  - **Recommendation (then):** (a) + (b), optionally with longer training. Implemented as decided above. Death is ramped (20 → 10 → 5) rather than switched off, because below subsistence utility is 0, so without death there is no survival incentive at all. Diagnostic scripts used (scratchpad, not in the repo): survival by agent group, deprivation causes, heuristic barter/money policies, calibration variants.
- **[PROPOSED 2026-10-13, not adopted] Further fixes/levers** (FRAMEWORK §11): automatic subsistence consumption (removes eating-mistake deaths; consumption above need stays learned), softer death ramp (20 → 10 → 7 → 5), rarer double coincidence (12 goods, 2–4 valued; F1), lumpier income (P 1–8; F5b), 24 agents. The user chose only entropy 0.03 + perishability for night 2.
- **Confirmatory design:** number of seeds (≥ 5 planned), and which knock-in/knock-out conditions and dose–response sweeps to run (FRAMEWORK §6.2/6.3). Needs a proposal and approval.
- **Final stage thresholds** (`MetricsConfig`) must be fixed before confirmatory runs. They are currently provisional.
- **Planned extensions (need approval):**
  - *(done 2026-10-14)* agent memory (`model.memory`)
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
