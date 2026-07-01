# Paper Replication Framework

This document is a guide for replicating quantitative finance research papers. It is a living document — fill it in as you work through each stage. Nothing is final until the entire project is complete.

The goal is not to prove the paper right. The goal is to systematically determine whether its claims hold up, and to what degree biases, overfitting, or data issues limit how far the findings generalise.

!!! note "Write in your own words"
    Paraphrasing forces deeper understanding than copying. If you must quote directly, cite it precisely. Unintentional plagiarism still has consequences — including in internal research that later gets published or shared.

---

## Process Overview

Most paper replications take roughly one week of focused work. The literature review can extend significantly if the territory is unfamiliar. Time budgeting matters — you can chase ideas indefinitely, so be deliberate about how much time you allocate to each stage before moving on.

| Stage | Action | Deliverable |
|-------|--------|-------------|
| 1 | Summarise the paper | 2–3 page précis document |
| 2 | Describe the hypotheses | Structured hypothesis/test pairs |
| 3 | Literature review | Annotated bibliography or topic-organised review |
| 4 | Locate and validate data | Data log + quality checks + loading code |
| 5 | Build and validate the model | Working code + results comparison + documented divergences |
| 6 | Extend the analysis | Out-of-sample tests, assumption challenges, similar techniques |

---

## Stage 1 — Summarise the Paper

Write a formal 2–3 page summary before doing anything else. The goal is to confirm you understand the paper and its research context — and to surface logical errors or gaps early, when they are cheapest to find.

A concise summary acts as a road map for every subsequent stage. It will help uncover logical errors or omissions in both the paper and your own reading, and identify early where you will likely have difficulty.

### Introductory Paragraph

Write 4–6 declarative sentences structured as follows:

- **One sentence** — title, authors, venue, and the thesis of the paper. Use a precise verb: *assert, argue, demonstrate, refute, examine, disprove*.
- **One sentence per major point** (no more than five) — state the technique used and its claimed result.
- **One sentence** — what the reader should take away from the paper overall.

### One Paragraph Per Major Point

For each major point identified in the introduction, write one paragraph structured as:

- A declarative sentence restating and paraphrasing the technique or claim
- 2–4 sentences covering: the main methods or tests used, key formulas with citation where they add precision, and results from this portion of the paper
- One concluding sentence: the takeaway from this specific point

Candidates for main points include key assertions or findings, major contributions, and a summary of the main techniques used.

### Conclusion Paragraph

Tie together all main points. Describe the relevance of the work to your specific research interests.

---

## Stage 2 — Describe the Hypotheses

Extract the hypotheses from the paper explicitly. Do not leave them implicit. Each hypothesis needs to be enumerated, described, and paired with a test before you begin the literature review or model-building.

For each hypothesis, document:

- **Subject** — what is being analysed
- **Dependent variable(s)** — the output, result, or prediction
- **Independent variables** — the inputs into the model
- **Anticipated possible outcomes** — including direction or comparison
- **Validation method** — how you will confirm or refute this hypothesis

!!! warning "Hypotheses may be revised"
    The literature review in Stage 3 may refute one or more of your initial hypotheses. If it does, document the refutation carefully and revise. Do not silently discard the original hypothesis — the revision itself is part of the record.

---

## Stage 3 — Literature Review

The literature review confirms you understand the framework your paper sits within, identifies what tools and knowledge you need, and surfaces existing work that supports or challenges the paper's claims.

### Format for Each Reference

Follow this structure for every paper you review:

1. Full bibliographic entry
2. One sentence: the thesis of the work
3. 2–4 sentences: main points or findings
4. One sentence: relevance and relation to this replication project

The goal is not deep understanding of every paper. The goal is to understand the framework your research fits within and gather the tools needed to proceed — like a chef measuring out ingredients before starting a recipe.

### What to Include

**Key references from the source paper** — papers that provided starting material or key techniques. Foundational papers cited by many others should be prioritised.

**Similar work** — use Google Scholar's recommendations and keyword searches from the source paper. Review at least a few papers at the top of a literature search on the key topics.

**Implementation references** — hands-on references covering the key analytical techniques to be used in the replication. These should be as practical as possible (e.g. a specific chapter of a textbook that covers the relevant method).

**Refuting evidence** — if the literature search uncovers papers that challenge or contradict the source paper's methods, document these carefully. They are potentially the most valuable part of the review.

### Organisation Models

Choose one before you start:

**Annotated bibliography** — organised alphabetically by author. Easier to build from reference manager software. Harder to navigate by topic during model-building.

**Topic-organised** — grouped by the 2–4 main topics of the paper. Faster to locate relevant references during active work. More project-specific, less reusable for future projects.

!!! note "Always cite everything"
    Every paper you look up, every formula you borrow, every blog post you consult. You are your own most important future reader. Every properly recorded citation shortens the time to come back up to speed when you return to this project months or years later.

---

## Stage 4 — Locate and Validate Data

Summarise the data described in the source paper first, then document your process for locating equivalent data. The original source data is almost never publicly available.

### Data Described in the Source Paper

Record the following from the paper:

- Data sources named (vendors, databases)
- Time frame used
- Specific instruments or assets
- Any data cleaning or filtering described

If the data is not published, consider emailing the primary author politely to request it. If they provide it, cite this in your report.

### Data Acquisition Order

Work through this sequence:

1. Check what data vendors your organisation has access to
2. Check if any vendors named in the paper are accessible to you
3. If not, list possible alternative sources from what you have available
4. Determine which symbols or instruments to request from each source
5. If full coverage is not possible, determine if representative data is sufficient to proceed

Keep notes on every step: the vendor used, symbols requested, download date, and any quirks encountered. This will save significant time if you return to this project later.

### Data Quality Checklist

- [ ] Is the data complete? Are there gaps in the time series?
- [ ] Do you have all the instruments described in the source paper?
- [ ] Do the time frames match (or reasonably approximate) those in the paper?
- [ ] Can you quickly replicate a chart or table from the paper to sanity-check the data?
- [ ] Are there any obvious outliers that need cleaning?
- [ ] Does the paper describe a specific cleaning mechanism? Have you replicated it?
- [ ] Have you reserved more recent data (beyond the paper's time frame) as a held-out validation set?

!!! tip "Reserve a validation set"
    Always request data through to today's date, not just the paper's time frame. The newer data is your out-of-sample validation set. Do not use it until the replication is complete.

---

## Stage 5 — Build and Validate the Model

This is the hardest stage and the one most likely to contain errors. Work incrementally. Document as you go, including failed attempts — they are part of the record and often valuable.

### Strategy Model Type

Identify which model type the paper presents before you start building:

**Signal-based strategy** — implements one or more indicators, signals, and rules to generate trades.

**Portfolio strategy** — constructs and rebalances a portfolio of instruments using an optimisation or weighting method.

**Pricing strategy** — may or may not implement a trading strategy; may use a strategy only as an example of the pricing methodology.

!!! note "You may choose a more realistic method than the paper used"
    Many papers ignore transaction costs, execution timing, and look-ahead bias. Replicating with a complete indicator-signal-rules model may produce a more useful result even if it diverges from the paper's exact method. Document the choice either way.

### Validation at Each Stage

At each step of the model build, check the following:

- [ ] Does your output match (or come close to) every number published in the paper?
- [ ] Have you documented where results diverge and investigated why?
- [ ] Have you noted any key assumptions or guesses made to fill in missing implementation detail?
- [ ] Are failed attempts left in the document rather than deleted?
- [ ] Have you checked code carefully for bugs, reversed signs, or off-by-one errors?
- [ ] If results contradict the paper: can you replicate from a different source using the same technique?
- [ ] Can you independently validate the maths?
- [ ] Can data differences explain any divergence?
- [ ] Have you run the hypothesis tests identified in Stage 2?
- [ ] Have you included model fit or calibration statistics where applicable?

### When Results Contradict the Paper

If you correctly implement a technique but cannot match the paper's results after careful checking, work through this sequence:

1. Check your work carefully for bugs, reversed signs, and indexing errors — this is the most likely cause
2. Attempt to replicate the same technique from a different source paper
3. Independently validate the maths
4. Determine whether data differences or data cleaning choices can explain the divergence

If after all of this you are confident in your implementation but the results still do not match, document it thoroughly with code and conclusions. A contradictory result is sometimes the most important finding of a replication.

---

## Stage 6 — Extend the Analysis

Replication confirms what the paper claims. Extension tests whether those claims generalise — to new data, different instruments, and more realistic assumptions.

### Summary Statistics

Present standard summary statistics for your replication. Some will match what the paper reported, providing further confirmation. Others will be new and may warrant commentary.

### More Data

Extend the analysis using more data in the following order:

**More recent data, same instruments** — the clearest and simplest test of out-of-sample deterioration and overfitting. This should always be part of a replication report.

**Similar instruments** — once the code is working, extension to similar instruments is straightforward. This tests for selection bias and begins to address generalisability.

**Different asset classes** — the furthest extension from the original paper. Usually saved for follow-on work rather than the initial replication.

### Challenging Simplifying Assumptions

Almost all published papers use simplifying assumptions. Many exist to fit in with prior literature or to make the analysis tractable. In practice they often make a technique unsuitable for use with real data. The most common ones to examine:

**Gaussian assumption** — many papers model volatility, errors, or noise using a Gaussian distribution, often while acknowledging it is inappropriate. What would a better distributional choice look like for your data?

**Sample moments** — widely used because it is easy. Many better estimation methods exist. Does relying on sample moments drive any of the paper's key results?

**Parameter count** — too many parameters suggests overfitting; too few suggests an underspecified model. Papers often choose parameter counts that made publication easier rather than ones that would work on a real portfolio.

### Similar Techniques

If the literature review uncovered similar techniques to those in the source paper, apply them to the same data where feasible. Prioritise cases where:

- The paper claims to improve on a prior technique but reports incomplete results for the original
- A related paper from the literature review extends, clarifies, or refutes claims made in the source paper

### Overfitting and Bias Checklist

- [ ] **Selection bias** — were the instruments, time frame, or parameters chosen in a way that fits the result?
- [ ] **Look-ahead bias** — does the model use information that would not have been available at trade time?
- [ ] **Out-of-sample deterioration** — does performance degrade significantly on the held-out validation set?
- [ ] Have you tested on the reserved validation set (more recent data)?

---

## Using AI Tools in This Project

Generative AI is part of modern research workflows. Use it deliberately — as an assistant that makes you faster, not as a shortcut that bypasses understanding.

### Where AI Helps

- Writing boilerplate code: downloading data, loading files, setting up environments
- Early literature search: deep research tools can rapidly surface dozens of references and summarise a field, potentially saving days of early-stage work
- Drafting the literature review: once you have chosen key papers, AI can help structure the write-up — you still need to understand the concepts yourself
- Editing and tone: useful for non-native speakers, but start with your own draft

### Where AI Does Not Help

- Explaining the maths: LLMs are not reliable for nuanced mathematical derivations in advanced or esoteric techniques
- Writing code for specific techniques: the more novel the method, the less reliable the output
- Replacing the analytical work: using AI to do the entire replication produces poor deliverables and leaves you without genuine understanding of whether the paper's claims hold up

!!! warning "LLMs are research assistants, not replacements"
    When using AI to generate code, break requests into single functions or small pieces and ask the model to write tests alongside the code. An IDE-integrated tool (Cursor, Claude Code, Copilot) is far more effective than a browser-based chat for this kind of work. The built-in execution environments of browser-based LLMs do not have enough context to reliably test the code they generate.

---

## Master Checklist

Use this to track overall project progress.

### Stage 1 — Summarise
- [ ] Paper précis written (2–3 pages, formal structure)
- [ ] Summary reviewed and gaps identified

### Stage 2 — Hypotheses
- [ ] All hypotheses extracted and enumerated
- [ ] Each hypothesis paired with a validation test

### Stage 3 — Literature Review
- [ ] Key references from the source paper reviewed
- [ ] Similar work identified and summarised
- [ ] Implementation references included
- [ ] Any refuting evidence documented

### Stage 4 — Data
- [ ] Source data described (vendor, instruments, time frame)
- [ ] Equivalent data located and downloaded
- [ ] Data quality checks completed
- [ ] Validation set (recent data) reserved and untouched

### Stage 5 — Model
- [ ] Strategy model type identified
- [ ] Key analytical techniques implemented
- [ ] Results validated against paper at each stage
- [ ] Divergences documented with investigation notes
- [ ] Hypothesis tests run

### Stage 6 — Extend
- [ ] Summary statistics presented
- [ ] Out-of-sample (more recent data) test completed
- [ ] Simplifying assumptions identified and challenged
- [ ] Overfitting and bias checks completed
- [ ] Final conclusions written

---

## Reference

Peterson, B. G. (2016). *Reproducible Research*, Chapter 4. In *Quantitative Trading Strategy Research*. Retrieved from [https://quantstrat.io/wp-content/uploads/2026/05/quantstratbook.pdf](https://quantstrat.io/wp-content/uploads/2026/05/quantstratbook.pdf)

Additional references cited in the chapter:

- Chang, A. C., & Li, P. (2015). Is Economics Research Replicable? Sixty Published Papers from Thirteen Journals Say "Usually Not". *Finance and Economics Discussion Series*, Federal Reserve Board.
- Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.
- Levy, Y., & Ellis, T. J. (2006). A Systems Approach to Conduct an Effective Literature Review in Support of Information Systems Research. *Informing Science Journal*, 9, 181–212.
- Peterson, B. G. (2017). *Overfitting and Bias in Quantitative Finance*.
- Peterson, B. G., Ulrich, J., Humme, P., et al. (2019). *quantstrat: Transaction-oriented infrastructure for constructing trading systems and simulation*. R package.
