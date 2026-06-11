# PeerJ Computer Science — Formal Review Report

**Manuscript:** Agent Memory Decay: Theoretical Framework and Empirical Benchmark
**Recommendation:** Major Revision Required
**Confidence:** High (4/5)

---

## 1. Summary

This paper benchmarks 12 memory decay models for LLM agent systems across 3 datasets, proves an applicability boundary theorem (Theorem 1), and validates findings on a production OS agent (Kylin V10). The central finding is that GRU+Time outperforms all ODE-based models on real-world data.

The paper addresses a relevant problem (model selection for agent memory decay) and provides a useful benchmark. However, several issues prevent acceptance in its current form.

---

## 2. Critical Issues (Must Fix)

### C1. Section 2.4 Missing — Complexity Table Not in Text
**Location:** Section 2
**Issue:** The abstract and introduction promise "computational complexity analysis (Table 3)" but Section 2 jumps from 2.3 (Theorem 1) to 2.5 (Proof). Section 2.4 with the complexity table is missing from the manuscript. The table is referenced but never presented.
**Fix:** Add Section 2.4 "Computational Complexity" with the full complexity comparison table for all 12 models.

### C2. Tables Not Numbered or Presented
**Location:** Sections 4-6
**Issue:** The text references "Table 1", "Table 2", ..., "Table 10" but none of these tables are actually presented in the manuscript. For a benchmark paper, the results tables are the core contribution. Without them, the paper is unreviewable.
**Fix:** Include all 10 tables with actual data. At minimum:
- Table 1: Dataset statistics
- Table 2: Model architectures and parameters
- Table 3: Computational complexity
- Table 4: Main results (MSE, F1 across 12 models on 3 datasets)
- Table 5: Ablation results
- Table 6: Tau analysis
- Table 7: Temporal pattern results
- Table 8: Competition benchmarks
- Table 9: Gate ablation
- Table 10: Cross-platform comparison

### C3. Introduction Too Short
**Location:** Section 1
**Issue:** The introduction is only 3 paragraphs (~150 words). This is insufficient for a journal paper. There is no background on agent memory systems, no motivation beyond one sentence, no discussion of why this question matters, and no roadmap of the paper.
**Fix:** Expand to 1000-1500 words covering: (1) background on LLM agent memory, (2) the decay modeling problem, (3) why model selection matters, (4) gap in existing work, (5) our approach, (6) contributions, (7) paper organization.

### C4. Data Availability Statement Missing
**Location:** End of paper
**Issue:** PeerJ CS requires a Data Availability Statement. The paper claims "all code, data, and proofs are publicly available" but provides no actual URL, DOI, or repository link.
**Fix:** Add a "Data Availability" section with specific GitHub URL and Zenodo DOI.

### C5. Theorem 1 Statement is Informal
**Location:** Section 2.3
**Issue:** The theorem statement uses informal notation (MSE_A, MSE_D, Omega(...)) without defining these precisely. What is MSE_A? The MSE of model class A on what dataset? The Omega notation is used but not defined (is it big-O? big-Omega? Landau notation?).
**Fix:** Make the theorem statement mathematically precise with all terms defined.

### C6. Proof of Part (ii) is Incomplete
**Location:** Section 2.5
**Issue:** The proof of Part (i) is detailed and correct. However, Part (ii) (Discrete Advantage Condition) relies on the claim that "irregular sampling breaks ODE uniform-step assumption" without rigorous justification. The expression MSE_DT <= MSE_A - Omega(epsilon_irregular) is stated but epsilon_irregular is not formally derived. The proof essentially asserts the conclusion.
**Fix:** Provide a rigorous lower bound on the ODE error under irregular sampling, and an upper bound on the discrete+time error, then show the gap.

---

## 3. Major Issues (Should Fix)

### M1. Keep/Forget Label Quality
**Location:** Section 4.3
**Issue:** Labels are derived from cosine similarity (threshold 0.3), not human annotation. This threshold is arbitrary and not validated. The paper does not report: (a) inter-annotator agreement, (b) label distribution (how many keep vs forget?), (c) sensitivity to threshold choice.
**Fix:** Report label distribution. Add a sensitivity analysis varying threshold from 0.2 to 0.5. If possible, validate on a human-annotated subset.

### M2. Statistical Rigor
**Location:** Section 4.3, 5.1
**Issue:** Only 3 random seeds. Standard practice is at least 5 for reliable confidence intervals. The paper claims "paired t-test with Bonferroni correction" but does not report actual p-values for the main comparisons (GRU+Time vs LTC, etc.).
**Fix:** Use 5 seeds. Report p-values and effect sizes (Cohen d) for key comparisons in a table.

### M3. PersonaChat Task Too Easy
**Location:** Section 5.1
**Issue:** All models except LTC/CCT reach F1=0.958 ceiling. This means the task does not differentiate models. The paper acknowledges this but does not propose a harder task or explain why the ceiling exists.
**Fix:** Analyze why the ceiling exists. Is it because the keep/forget distinction is trivial? Or because the threshold is too permissive? Add a harder evaluation (e.g., graded relevance, multi-class keep/update/forget).

### M4. CCT-LNN is a Negative Result Presented as Contribution
**Location:** Sections 5.1-5.3
**Issue:** CCT-LNN is the worst-performing model on PersonaChat (MSE 0.120 vs GRU 0.077) and matches LTC on LPT. The paper proposes CCT-LNN as a "novel variant" but the results show it does not work. This weakens the paper.
**Fix:** Either: (a) remove CCT-LNN and focus on the 11 existing models, or (b) present CCT-LNN honestly as a failed attempt with analysis of why it failed (insufficient categories, overfitting, etc.). Option (b) is more honest and informative.

### M5. "Competition" Terminology Misleading
**Location:** Section 6.2
**Issue:** The paper repeatedly mentions "4/4 competition benchmarks" and "competition pass rate." This implies participation in a third-party evaluation. If these are self-defined metrics for an internal project, this should be clearly stated.
**Fix:** Clarify that these are system-defined benchmarks for an internal OS agent project, not a public competition.

### M6. LPT Dataset Reference Incomplete
**Location:** References
**Issue:** Reference [15] is "Xu X, et al. Long-term memory tracking and evaluation for LLM agents. arXiv:2404.xxxxx, 2024." The arXiv ID is a placeholder (xxxxx). This is not a valid reference.
**Fix:** Find the actual arXiv ID or replace with the correct citation.

### M7. Duplicate References
**Location:** References
**Issue:** References [6] and [21] are both "Li Y, et al. Hybrid Transformer with LNN." References [7] and [20] are both "Akpinar N, et al. Uncertainty LNN." This is sloppy.
**Fix:** Deduplicate references.

### M8. PersonaChat Keep/Forget Binary is Unnatural
**Location:** Section 4.1, 5.1
**Issue:** Treating memory decay as binary (keep/forget) is a simplification. Real decay is gradual (relevance decreases over time). The binary formulation loses information and may explain the F1 ceiling.
**Fix:** Acknowledge this limitation explicitly. Consider adding a graded relevance evaluation as supplementary.

---

## 4. Minor Issues (Consider Fixing)

### m1. Writing Quality
- The introduction reads like bullet points, not prose. Needs smoother transitions.
- "We make four contributions." is abrupt. Use "Our contributions are as follows:" or integrate naturally.
- Several sentences start with "This" or "These" without clear antecedents.

### m2. Missing Related Work on Benchmarks
- No discussion of existing benchmarks for memory systems (e.g., LOCOMO, MemEval).
- No comparison with memory system evaluation methodologies.

### m3. No Error Bars in Text
- Results are reported as "MSE 0.077" without standard deviation. Should be "0.077 +/- 0.001".

### m4. Table 3 Missing from Text
- The computational complexity table is promised but not included.

### m5. Figure Missing
- No figures are included. A benchmark paper should have: (a) bar chart of main results, (b) radar chart of model comparison, (c) tau distribution across categories.

### m6. Hyperlinks in References
- References do not include DOIs or URLs. PeerJ CS expects DOIs where available.

---

## 5. Strengths

1. **Relevant problem:** Model selection for agent memory decay is practically important.
2. **Comprehensive benchmark:** 12 models across 3 datasets is thorough.
3. **Theoretical framework:** Theorem 1 provides a principled way to choose between model classes.
4. **Production validation:** Kylin V10 deployment data adds practical credibility.
5. **Honest reporting:** The paper does not hide negative results (CCT-LNN failure, GRU+Time superiority).

---

## 6. Detailed Recommendations

### Priority 1 (Critical — must fix before review can proceed):
1. Add all 10 tables with actual data
2. Expand introduction to 1000+ words
3. Add Data Availability Statement with real URLs
4. Complete Section 2.4 (complexity table)
5. Make Theorem 1 mathematically precise
6. Complete proof of Part (ii)

### Priority 2 (Major — strongly recommended):
1. Report label distribution and threshold sensitivity
2. Use 5 seeds with p-values and effect sizes
3. Clarify "competition" terminology
4. Fix incomplete/duplicate references
5. Add error bars to all reported numbers
6. Address CCT-LNN negative result explicitly

### Priority 3 (Minor — improve quality):
1. Add figures (bar chart, radar chart, tau distribution)
2. Improve introduction prose quality
3. Add DOIs to references
4. Add related work on memory benchmarks
5. Discuss binary vs graded relevance limitation

---

## 7. Verdict

**Major Revision Required.** The paper addresses a relevant problem and provides a useful benchmark with theoretical grounding. However, the missing tables, too-short introduction, incomplete proofs, and several data quality issues must be addressed before the paper can be properly evaluated. The core findings (GRU+Time superiority, tau uniformity, production validation) are interesting and publishable if presented with full supporting evidence.

---

*Review completed: 2026-06-11*
*Reviewer expertise: Machine Learning, NLP, Agent Systems*
