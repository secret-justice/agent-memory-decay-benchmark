# PeerJ Computer Science - Self-Review Report
# Manuscript: Agent Memory Decay: Theoretical Framework and Empirical Benchmark
# Version: v22

## Overall Assessment: ACCEPT with Minor Revisions

### Strengths
1. S1: Three complementary evaluation paradigms (regression, focal, ranking) is methodologically sound
2. S2: Honest reporting of negative results (MSE cannot differentiate models)
3. S3: Detailed Theorem 1 with full proofs (Part i and ii)
4. S4: Production deployment on Kylin V10 with concrete metrics
5. S5: Computational complexity table for all 8 models
6. S6: NDCG@5 successfully differentiates models (0.533-0.616 vs random 0.500)

### Critical Issues: 0

### Major Issues: 3
M1. LPT dataset missing from Exp-3: Ranking evaluation only on PersonaChat (100 users, 8 categories). LPT (200 users, 5 categories) would strengthen generalizability. Acknowledged in Limitations but should be addressed if possible.
    Recommendation: Add LPT ranking results or provide explicit memory constraint justification.

M2. Statistical significance: Exp-3 uses only 3 seeds with SD 0.003-0.027. Need to confirm whether Neural ODE (0.616) is significantly different from GRU (0.0.604) at p<0.05.
    Recommendation: Add paired t-test between top-3 models.

M3. CCT-LNN underperforms: The proposed model (CCT-LNN) ranks near bottom on ranking (0.536). The paper should more explicitly acknowledge this and explain why category conditioning does not help.
    Recommendation: Add discussion paragraph explaining CCT-LNN failure modes.

### Minor Issues: 5
m1. Abstract mentions "8 architectures" but Table 3 lists 10 models (including ExpDecay, xLSTM). Clarify which 8 are the primary comparison.
m2. PersonaChat keep/forget ratio (92%/8%) was the C0 problem - now resolved by switching to regression/ranking framing. Good.
m3. Reference [15] and [17] have placeholder "xxxxx" in arXiv IDs. Need real IDs.
m4. Table 2 (Model Architectures) is referenced but content not shown - should include all 8 models.
m5. "All code, data, and proofs are publicly available" - no actual DOI/link provided yet.

### Formatting Compliance
- [OK] US Letter (8.5x11)
- [OK] 12pt Times New Roman
- [OK] Line numbers
- [OK] 2.54cm margins
- [OK] Structured abstract (Background/Methods/Results/Conclusions)
- [OK] Left justified
- [NEEDS] Data/Code DOI link

### Verdict
The paper has been substantially improved by the C0 fix. The key narrative shift from "binary classification" to "ranking quality" is scientifically correct and more relevant to the actual task of memory retrieval. The three-experiment methodology (regression -> focal -> ranking) is a strong contribution. With the 3 major issues addressed, this is suitable for PeerJ Computer Science.

Estimated acceptance probability: 65-75% (after addressing M1-M3)
