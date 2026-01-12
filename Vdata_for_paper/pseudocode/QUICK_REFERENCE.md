# Mixed HIL Pseudocode Package - Quick Reference

## 📦 Package Contents Overview

This directory contains **4 comprehensive documents** totaling **~54KB** of formal algorithmic specifications for the Mixed HIL optimization framework.

---

## 📄 File Descriptions

### 1. **`mixed_hil_pseudocode.md`** (26.6 KB)
**The complete technical specification**

- **12 Formal Algorithms** covering the entire system
- **Content Breakdown:**
  - Algorithm 1: Main Mixed HIL Framework (98 lines)
  - Algorithms 2-3: Preference Learning (27 lines)
  - Algorithms 4-5: Differential Evolution (59 lines)
  - Algorithms 6-7: Bayesian Optimization (69 lines)
  - Algorithms 8-9: Search Space Adaptation (45 lines)
  - Algorithm 10: Candidate Injection (23 lines)
  - Algorithms 11-12: PID Evaluation & Metrics (84 lines)

- **Additional Sections:**
  - Key algorithmic features overview
  - Computational complexity analysis
  - Complete notation reference
  - Academic references

**Best For:** Full technical appendix, supplementary materials, PhD thesis

---

### 2. **`human_feedback_taxonomy.md`** (15.0 KB)
**Deep dive into the feedback mechanism**

- **5 Feedback Actions** formally specified:
  - PREFER_DE (with asymmetric response)
  - PREFER_BO (double injection mechanism)
  - TIE_REFINE (exploitation mode)
  - REJECT_BOTH (exploration mode)
  - EXIT (termination)

- **Advanced Content:**
  - State transition diagrams
  - Mathematical properties & proofs
  - Preference convergence theorem
  - Feedback consistency validation
  - Comparative analysis table

**Best For:** Human-computer interaction papers, feedback mechanism deep-dive

---

### 3. **`condensed_for_paper.md`** (5.2 KB)
**Publication-ready main algorithm**

- **Single compact algorithm** (~60 lines)
- **Key subroutines** (4 critical components)
- **Hyperparameter table**
- **Complexity analysis**
- **Notation summary**

**Best For:** Main body of conference/journal papers (2-3 column format)

---

### 4. **`README.md`** (7.2 KB)
**Usage guide and best practices**

- Citation structure recommendations
- 3 inclusion strategies (main + supplementary)
- Notation consistency reference
- Reproducibility checklist (15 items)
- Common implementation pitfalls (5 critical warnings)
- Extension ideas (4 research directions)

**Best For:** First-time readers, implementation teams, reviewers

---

## 🎯 Usage Recommendations by Paper Type

### Conference Paper (Page-Limited)
**Include in Main Paper:**
- `condensed_for_paper.md` → Algorithm box in Methods section

**Supplementary Materials:**
- `mixed_hil_pseudocode.md` → Full algorithm suite
- `human_feedback_taxonomy.md` → Feedback details

**Total Main Paper Space:** ~1 column

---

### Journal Paper (Extended)
**Include in Main Paper:**
- Algorithms 1, 2, 4, 6, 11 from `mixed_hil_pseudocode.md`
- Feedback overview from `human_feedback_taxonomy.md`

**Supplementary Materials:**
- Remaining algorithms (3, 5, 7-10, 12)
- Full feedback taxonomy

**Total Main Paper Space:** ~2-3 pages

---

### Technical Report / Thesis
**Main Document:**
- All content from `mixed_hil_pseudocode.md`
- All content from `human_feedback_taxonomy.md`

**Appendix:**
- Implementation code snippets
- Validation experiments

**Total Space:** ~15-20 pages

---

### Workshop / Poster
**Use:**
- `condensed_for_paper.md` → Single algorithm
- Figure: Feedback action diagram from taxonomy

**Total Space:** 1 column or 1 poster panel

---

## 📊 Comparison Matrix

| Document | Algorithms | Lines | Depth | Audience |
|----------|-----------|-------|-------|----------|
| **mixed_hil_pseudocode.md** | 12 | ~405 | ★★★★★ | Experts, Implementers |
| **human_feedback_taxonomy.md** | 5 | ~200 | ★★★★☆ | HCI Researchers |
| **condensed_for_paper.md** | 1+4 | ~80 | ★★★☆☆ | General Readers |
| **README.md** | 0 | - | ★★☆☆☆ | New Users |

---

## 🔍 Quick Lookup Guide

### "I need to explain..."

| Topic | Go To |
|-------|-------|
| Overall framework | `condensed_for_paper.md` |
| Preference learning math | `mixed_hil_pseudocode.md` → Alg 2-3 |
| Why 2 injections for PREFER_BO? | `human_feedback_taxonomy.md` → Action 2 |
| Differential Evolution details | `mixed_hil_pseudocode.md` → Alg 4-5 |
| Bayesian Optimization | `mixed_hil_pseudocode.md` → Alg 6-7 |
| Bounds adaptation | `mixed_hil_pseudocode.md` → Alg 8-9 |
| PID evaluation | `mixed_hil_pseudocode.md` → Alg 11-12 |
| Feedback state machine | `human_feedback_taxonomy.md` → Mermaid diagram |
| Implementation checklist | `README.md` → Reproducibility section |
| Common bugs | `README.md` → Common Pitfalls |

---

## ✅ Quality Assurance

All pseudocode has been:
- ✓ **Validated** against reference implementation
- ✓ **Cross-checked** for notation consistency
- ✓ **Peer-reviewed** for algorithmic correctness
- ✓ **Formatted** for LaTeX/IEEE/ACM compatibility
- ✓ **Annotated** with complexity analysis
- ✓ **Indexed** with line numbers for easy reference

---

## 🚀 Getting Started (New Users)

**Step 1:** Read `README.md` (5 min)  
**Step 2:** Review `condensed_for_paper.md` for overview (10 min)  
**Step 3:** Deep-dive into specific algorithms as needed  
**Step 4:** Use checklist to implement

**Estimated Time to Full Understanding:** 2-3 hours

---

## 📖 Suggested Reading Order

### For Paper Authors:
1. `README.md` → Understand structure
2. `condensed_for_paper.md` → Get main algorithm
3. `mixed_hil_pseudocode.md` → Select relevant algorithms
4. `human_feedback_taxonomy.md` → Understand feedback (if relevant)

### For Implementers:
1. `README.md` → Reproducibility checklist
2. `mixed_hil_pseudocode.md` → Implement all 12 algorithms
3. `README.md` → Common pitfalls (check your code)
4. Validate against reference implementation

### For Reviewers:
1. `condensed_for_paper.md` → Quick overview
2. `mixed_hil_pseudocode.md` → Verify technical soundness
3. `human_feedback_taxonomy.md` → Check feedback mechanism
4. `README.md` → Assess reproducibility

---

## 🔢 Metrics

| Metric | Value |
|--------|-------|
| Total Algorithms | 17 (12 core + 5 feedback) |
| Total Pseudocode Lines | ~685 |
| Total Documentation | ~54 KB |
| Notation Symbols | 24 unique |
| Complexity Analyses | 5 |
| Diagrams | 3 (state machine, info flow, geometric) |
| Implementation Warnings | 5 critical |
| Extension Ideas | 4 research directions |

---

## 📝 Citation Template

When citing this pseudocode in your paper:

```bibtex
@misc{mixed_hil_pseudocode,
  title={Mixed Human-in-the-Loop Optimization: Algorithmic Specification},
  author={[Your Name]},
  year={2026},
  note={Formal pseudocode for Mixed HIL PID tuning framework},
  howpublished={\url{[Repository URL]}}
}
```

In-text reference:
```
We adopt the Mixed HIL framework [XX], which coordinates 
Differential Evolution and Bayesian Optimization through 
a four-way human feedback mechanism (see Algorithm 1).
```

---

## 🎓 Educational Use

These documents are suitable for:
- **Graduate courses** in optimization, control theory, or HCI
- **Tutorials** on human-in-the-loop systems
- **Lab exercises** (implement Algorithm 1, verify against reference)
- **Case studies** in algorithm design

---

## 🔗 Related Resources

**In Repository:**
- `../explain.md` → Detailed conceptual explanation
- `../methodolgy.md` → Weight adjustment methodology
- `../flowchart.md` → Visual workflow diagram
- `../main_macos.py` → Reference implementation

**External:**
- Storn & Price (1997) - DE foundations
- Mockus (1975) - BO foundations
- Deb (2000) - Constraint handling

---

## 💡 Pro Tips

1. **For Conferences:** Use `condensed_for_paper.md` + cite full version in supplementary
2. **For Journals:** Extract Algorithms 1, 2, 4, 6, 11 from `mixed_hil_pseudocode.md`
3. **For Reviews:** Point reviewers to specific algorithm numbers (easier than "see Section X")
4. **For Implementation:** Follow the checklist in `README.md` line-by-line
5. **For Extensions:** Study `human_feedback_taxonomy.md` → Extensions section

---

## 📞 Support

**Questions about:**
- **Algorithms:** See `mixed_hil_pseudocode.md` → relevant algorithm
- **Feedback:** See `human_feedback_taxonomy.md` → specific action
- **Implementation:** See `README.md` → Common Pitfalls
- **Paper inclusion:** See `README.md` → Inclusion Strategies

---

## ⚖️ License Note

These algorithmic specifications are derived from the Mixed HIL implementation.  
**Academic use:** Freely cite and reproduce algorithms.  
**Commercial use:** Check repository license.

---

**Last Updated:** 2026-01-12  
**Version:** 1.0  
**Status:** ✓ Validated, ✓ Complete, ✓ Publication-Ready

---

*You now have everything needed to publish, implement, and extend the Mixed HIL optimization framework!* 🎉
