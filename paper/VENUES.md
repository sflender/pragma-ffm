# Target venues — relational FFM / cross-sequence encoder paper

**Paper (working title):** *When Two Attentions Aren't Enough: Cross-Sequence Encoding for
Relational Fraud in Financial Foundation Models.*

**Story arc (agreed):**
1. A standard 2-attention FFM (event encoder + history encoder) is **structurally blind** to
   cross-sequence / relational signals — it only ever sees one entity's own history.
2. **Controlled synthetic proof** (ground-truth benchmark): a 3rd "cross-sequence" transformer
   recovers relational fraud a per-sequence model can't; also isolates the proxy-alignment gap.
3. **Real-data evidence** (TalkingData click fraud, sequence-native): the 3rd transformer beats
   the per-sequence FFM (+0.055 PR-AUC) — the cross-user IP signal a per-user model misses.
4. **When it helps + productization**: a boundary map (4 real datasets — the win needs a
   *content/pattern* relational signal, not a count a cheap aggregate captures) + next steps
   (EVT-token caching for a frozen backbone, count-aware readout, cost/latency).

**Format target:** ~8 pages (workshop full paper or conference short/applied track).

> ⚠️ **All dates below are TYPICAL windows and MUST be verified on the official site.**
> Knowledge cutoff is Jan 2026; exact 2026/2027 deadlines are not confirmed here.

## Tier 1 — best fit (workshops + applied/finance)

| Venue | Why it fits | Page limit | Typical deadline (VERIFY) | Notes |
|---|---|---|---|---|
| **NeurIPS 2026 workshops** (TRL / Temporal Graph Learning / **ICBINB** "I Can't Believe It's Not Better") | Laptop-scale, mechanistic, honest positive+negative results are exactly ICBINB/TRL's remit | 4–9 pp | ~late Sept–early Oct 2026 | **Primary target.** Pick the workshop after they post; ICBINB loves the boundary/negative map |
| **ACM ICAIF 2026** (AI in Finance) | Directly on-topic (finance/fraud); accepts focused applied studies | ~8 pp (full) / poster | ~July–Aug 2026 | **Check now — may be imminent.** Strong topical fit |
| **KDD 2027 — Applied Data Science track** | Real-fraud, deployment-minded; ADS values the "when does it help + productization" framing | ~9 pp | ~Feb 2027 (2nd cycle, VERIFY) | High fit for the applied angle; more runway |
| **LoG 2026** (Learning on Graphs) | Covers the graph datasets (Elliptic/Yelp/Amazon) + relational modeling; has extended-abstract track | 9 pp / 4 pp abstract | ~Sept 2026 | Good home for the graph-vs-sequence boundary discussion |

## Tier 2 — conferences (higher bar, 8–9 pp fits)

| Venue | Why it fits | Typical deadline (VERIFY) | Notes |
|---|---|---|---|
| **The Web Conf (WWW) 2027** | Strong fraud/graph/security + industry tracks | ~Oct 2026 | Web-fraud angle (click fraud) lands well |
| **AISTATS 2027** | Controlled benchmark + honest analysis suits the venue's flavor | ~Oct 2026 | Competitive but on-flavor |
| **ICLR 2027** | Mechanistic + real-data; big venue | ~late Sept 2026 | Competitive; would need the story tightened |
| **AAAI 2027** | General; AI-for-social-impact / main | ~Aug 2026 (abstracts) | Competitive |
| **PAKDD 2027** | Applied data mining, fraud-friendly | ~Nov 2026 | Solid mid-tier applied home |

## Likely already closed for 2026 (note only)
CIKM 2026 (~May), ECML-PKDD 2026 (~Mar–Apr), DSAA 2026 (~Jun–Jul). Target their 2027 cycles.

## Recommendation
- **Fastest strong home:** a **NeurIPS 2026 workshop** (ICBINB or TRL) — the honest positive+negative
  structure is a natural fit and the bar suits laptop scale. Check **ICAIF 2026** immediately in case
  its deadline is near.
- **If aiming for an archival conference:** **KDD 2027 ADS** (applied, more runway) or **WWW 2027**
  (web-fraud fit).

## LaTeX setup (once venue chosen)
- Use the venue's official style: `neurips_2026.sty` (workshops usually reuse the main style),
  ACM `acmart` (ICAIF/KDD/WWW/CIKM — `sigconf`), AAAI `aaai2X.sty`, ICLR `iclr2027_conference.sty`.
- Suggested structure (8 pp): Intro · Background/related (TabBERT/BEHRT lineage; recsys HSTU/OneRec) ·
  The three encoders + cross-sequence encoder · Synthetic benchmark (S1–S3) · Real data (TalkingData
  win) · When-it-helps boundary (Elliptic/Yelp/Amazon/IEEE) · Productization (caching, count-aware) ·
  Limitations · Conclusion.
