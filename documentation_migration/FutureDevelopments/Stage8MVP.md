# Stage 8 MVP: Designer-Led Template Approach

**Purpose**: Generate professional PDF reports using designer templates + manual data population instead of full automation

**Status**: ✅ **PROPOSED ALTERNATIVE** - Faster, lower-risk path to production

**Parent Document**: Stage8Planning.md (original 57.5-day automated MVP)

**Trade-off**: Exchange development time (57.5 days → 16.75 days) for manual labor (~25 hours during onboarding)

---

## MVP Deliverables

### What You BUILD (Technical Work - 6.5 days)

**Phase 1: Template Structure Creation** (MUST happen BEFORE designer work - 2 days):
1. ✅ **Hashtag → Client** template structure (COMPLETE - from MLCreativeReports.md)
2. ✅ **Hashtag → Creator** template structure (COMPLETE - from Stage8Planning.md section 1.1)
3. **Handle/Single Competitor → Client** template structure (benchmarking sections, comparison layout)
4. **Handle/Multiple Competitor → Client** template structure (side-by-side comparison structure)

**Phase 2: Data Extraction Scripts** (can happen in parallel with designer - 3 days):
1. `extract_creator_data.py` - Stage 7 JSON → 9 formatted creator reports (Google Sheets)
2. `extract_client_data.py` - Stages 1,6,7 → 1 client executive dashboard (Google Sheets)
3. `extract_competitor_data.py` - Competitor analysis → benchmarking data (Google Sheets)

**Output Format**: ✅ **Google Sheets** (easiest to review/edit before populating templates)

### What DESIGNER BUILDS (Creative Work - 11 days)

**4 Static PDF Templates**:
1. **Template A**: Content Creator Report (2-page, mobile-optimized)
2. **Template B**: Client Executive Report (3-page, intelligence dashboard)
3. **Template C**: Single Competitor Report (3-page, benchmarking)
4. **Template D**: Comparison Competitor Report (4-page, side-by-side)

**Branding Package**:
- Visual identity system (colors, fonts, spacing, grids)
- Chart templates (bar charts, star ratings, timeline graphics)
- Icon library + brand assets (logos, dividers, backgrounds)

**Editable Format**:
- Adobe InDesign files with labeled text boxes
- OR Canva Pro templates with text fields
- OR Figma templates with text layers

---

## Workflow Per Report Type

### Hashtag Analysis → Content Creators (9 PDFs)

**Frequency**: Onboarding (~5 times), then rarely needed

| Step | Who | Time | Details |
|------|-----|------|---------|
| 1. Run pipeline Stages 1-7 | Automated | Auto | Existing ML pipeline |
| 2. Extract data + QR codes | Script | 30 sec | `python extract_creator_data.py --hashtag nutrition` → Google Sheet + 18 QR PNGs |
| 3. Review data | You | 15 min | Open Google Sheet, verify accuracy, edit if needed |
| 4. Populate Template A (x9) | You | ~3 hrs | Copy-paste from Sheet + insert 2 QR code images per report (~20 min each) |
| 5. Export PDFs | You | 5 min | Save as PDF from InDesign/Canva |

**Total Manual Time per Hashtag**: ~3.5 hours (for 9 creator PDFs, includes QR code insertion)

**Onboarding Total**: ~17.5 hours across 5 hashtags

---

### Hashtag Analysis → Client Executive (1 PDF)

**Frequency**: Onboarding (~5 times), biweekly ongoing

| Step | Who | Time | Details |
|------|-----|------|---------|
| 1. Run pipeline Stages 1-7 | Automated | Auto | Existing ML pipeline |
| 2. Extract data | Script | 30 sec | `python extract_client_data.py --hashtag nutrition` → Google Sheet |
| 3. Review data | You | 10 min | Open Google Sheet, verify bucket distributions, view counts, formulas |
| 4. Populate Template B | You | 20 min | Copy-paste from Sheet into 3-page template |
| 5. Export PDF | You | 2 min | Save as PDF |

**Total Manual Time per Hashtag**: ~30 min (for 1 client PDF)

**Onboarding Total**: ~2.5 hours across 5 hashtags

**Ongoing**: ~30 min every 2 weeks

---

### Competitor Analysis → Client (Single or Comparison)

**Frequency**:
- Single: Onboarding (~5 times)
- Comparison: Onboarding (~2 times)

**Single Competitor Report**:

| Step | Who | Time | Details |
|------|-----|------|---------|
| 1. Run pipeline Stages 1-7 for competitor | Automated | Auto | Existing ML pipeline |
| 2. Extract competitor data | Script | 30 sec | `python extract_competitor_data.py --competitor @rival_brand` → Google Sheet |
| 3. Review benchmarks | You | 10 min | Open Google Sheet, verify competitor vs client gaps |
| 4. Populate Template C | You | 25 min | Copy-paste from Sheet into Template C |
| 5. Export PDF | You | 2 min | Save as PDF |

**Total Manual Time per Competitor**: ~40 min

**Onboarding Total**: ~3.5 hours (5 single competitor reports)

---

**Comparison Report (Multi-Competitor)**:

Same workflow as single, but populate Template D with side-by-side data.

**Total Manual Time per Comparison**: ~60 min (more data to organize)

**Onboarding Total**: ~2 hours (2 comparison reports)

---

## Onboarding Phase Manual Work

**Assuming 5 hashtag analyses + 5 single competitors + 2 comparison reports:**

| Report Type | Quantity | Time Each | Total Time |
|-------------|----------|-----------|------------|
| Creator PDFs (9 per hashtag) | 5 hashtags | 3.5 hrs | 17.5 hrs |
| Client PDFs (1 per hashtag) | 5 hashtags | 30 min | 2.5 hrs |
| Single Competitor PDFs | 5 reports | 40 min | 3.5 hrs |
| Comparison PDFs | 2 reports | 60 min | 2 hrs |
| **TOTAL ONBOARDING** | - | - | **25.5 hrs** |

**Ongoing (Post-Onboarding)**:
- Biweekly client reports: ~30 min every 2 weeks (~2 hours/month)

---

## MVP Task Breakdown

### Section 0: Template Structure Creation (2 days) - **MUST COMPLETE BEFORE DESIGNER STARTS**

| # | Task | Owner | Effort | Status | Notes |
|---|------|-------|--------|--------|-------|
| 0.1 | Hashtag → Client structure | You | 0 days | ✅ **COMPLETE** | From MLCreativeReports.md |
| 0.2 | Hashtag → Creator structure | You | 0 days | ✅ **COMPLETE** | Stage8MVP_Reports.md Section 2 (all 6 issues resolved) |
| 0.3 | Handle/Single Competitor → Client | You | 0.75 days | ✅ **COMPLETE** | Stage8MVP_Reports.md Section 3 (4-page structure, dynamic fields). **REQUIRES 2 PIPELINE RUNS**: (1) `--analysis-mode recent --period 90` for posting frequency/hashtags, (2) `--analysis-mode top` for creative patterns/QR codes. Total: 4 runs per report (competitor + client baseline). |
| 0.4 | Handle/Multiple Competitor → Client | You | 0.5 days | ⏸️ **TODO** | Side-by-side comparison structure |

**Deliverables**: 4 content structure documents (similar to MLCreativeReports.md) defining:
- Page count and purpose
- Section titles and content requirements
- Data fields needed per section
- Chart/visual requirements
- Mobile optimization specs (for Template A)

**Critical Path**: Designer cannot start until all 4 structures are complete

---

### Section 1: Designer Templates (8 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 1.1 | Design Template A (Content Creator) | Designer | 2 days | 2-page, mobile-optimized, includes 2 QR code placeholders |
| 1.2 | Design Template B (Client Executive) | Designer | 2 days | 3-page, intelligence dashboard |
| 1.3 | Design Template C (Single Competitor) | Designer | 2 days | 3-page, benchmarking vs client |
| 1.4 | Design Template D (Comparison) | Designer | 2 days | 4-page, side-by-side multi-competitor |

**Template A Requirements** (from Issue 1 resolution - see Stage8MVP_Reports.md):
- Include 2 QR code placeholders (~1" x 1" each):
  - **QR Code 1**: After "The Proof" section (links to top performer video)
  - **QR Code 2**: In "Contrastive Analysis" section (links to bottom performer video)
- Labels: "Scan to watch: Top Performer Using This Pattern (520K views)" and "Bottom Performer - Don't Do This (95K views)"
- Technical: Leave square placeholder boxes for QR code image insertion during manual workflow

**Deliverables**: 4 editable PDF templates (InDesign/Canva/Figma) with clearly labeled text boxes

**Template Requirements**:
- **Labeled placeholders** for all data fields (e.g., `[PATTERN_NAME]`, `[AVG_VIEWS]`, `[ENGAGEMENT_DIFFERENCE]`)
- **Mobile-optimized** (Template A): Min 12pt body, 16pt+ headings, portrait layout
- **Chart templates**: Pre-designed bar charts, star ratings, timeline graphics
- **Brand consistency**: Tumi Labs colors, fonts, logo placement

---

### Section 2: Branding Package (3 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 2.1 | Create visual identity system | Designer | 1 day | Colors, fonts, spacing, grids |
| 2.2 | Create chart templates | Designer | 1 day | Bar charts, star ratings, timelines |
| 2.3 | Create icon library + assets | Designer | 1 day | Logos, dividers, backgrounds |

**Deliverables**:
- Brand style guide (PDF)
- Chart template library (editable files)
- Asset package (PNG/SVG files)

---

### Section 3: Data Extraction Scripts (3.25 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 3.1 | Build `extract_creator_data.py` + QR generation | Developer | 1.25 days | Stage 7 → 9 creator datasets + 18 QR code images |
| 3.2 | Build `extract_client_data.py` | Developer | 1 day | Stages 1,6,7 → client dashboard data (Google Sheets) |
| 3.3 | Build `extract_competitor_data.py` | Developer | 1 day | Competitor analysis → benchmarking data (Google Sheets) |

**QR Code Addition** (from Issue 1 resolution):
- Task 3.1 now includes QR code generation (+0.25 days effort)
- Generates 2 QR codes per formula (18 total per hashtag: 9 formulas × 2 codes)
- Maps Stage 2 video URLs (top/bottom cluster) to Stage 7 formulas
- Uses Python `qrcode` library to generate PNG files
- Output: `{hashtag}_{bucket}_{formula}_top.png` and `_bottom.png`

**Script Requirements**:
- **Input**: JSON files from Stages 1, 6, 7 (existing outputs)
- **Output**: ✅ **Google Sheets** with clearly labeled sections and data fields
- **Google Sheets API**: Use `gspread` or `google-sheets-python-api` for sheet creation
- **Sheet Structure**: One sheet per report type, with formatted headers and data rows
- **Error handling**: Clear error messages if JSON missing/malformed
- **CLI interface**: Simple command-line invocation

---

#### 3.1: `extract_creator_data.py`

**Purpose**: Extract 9 creative formulas from Stage 7 JSON → formatted datasets

**Input**:
- Stage 7 `winning_formulas.json` (3 winning buckets × 3 formulas each)

**Output Format** (per formula):
```
=== FORMULA 1: 18-33s Bucket ===
Pattern Name: [The Question Hook Formula]
Duration: [18-33s]
Hashtag: [#nutrition]
Confidence: [87%]

--- THE PROOF ---
Videos using this pattern: [8.4%] avg engagement
Videos NOT using this: [3.1%] avg engagement
Engagement difference: [2.7x MORE ENGAGEMENT]

--- CONTRASTIVE ANALYSIS ---
Top performers do THIS:
✅ [Ask question in first 2s (avg 3.2 questions in hook)]
✅ [Show product by 5 seconds (immediate visual payoff)]
✅ [Use 5-7 text overlays (keep attention with text)]

Bottom performers do THIS:
❌ [Generic opening/statement (0.8 questions avg)]
❌ [Product reveal after 10+ seconds (viewers already scrolled)]
❌ [No text overlays (viewers get bored/confused)]

--- PATTERN SUMMARY ---
1️⃣ Hook (0-3s): [Ask compelling question]
2️⃣ Show (3-15s): [Reveal product + explain benefit]
3️⃣ Prove (15-33s): [Demonstrate result + CTA]

--- SECOND-BY-SECOND TIMELINE ---
⏱️ 0-2 seconds: THE QUESTION
  Say: [Did you know [surprising fact about topic]?]
  Visual: [Your face, direct to camera]
  Text overlay: [The question (animated in)]

⏱️ 3-5 seconds: SHOW THE THING
  Visual: [Close-up of product/ingredient]
  Text overlay: [Product name]
  Say: [This is [product name]]

[... continues for all timestamps ...]

--- PRE-POST CHECKLIST ---
□ [Question in first 2 seconds?]
□ [Product visible by 5 seconds?]
□ [5-7 text overlays placed?]
□ [2-3 scene changes in middle?]
□ [Clear CTA at end?]
```

**CLI Usage**:
```bash
python extract_creator_data.py --client acme --hashtag nutrition --mode top --strategy engagement
```

**Output**: Google Sheet created with URL logged to console and saved to:
`/data/clients/acme/hashtags/nutrition/top_engagement/extracted_creator_reports_sheet_url.txt`

**Google Sheet Structure**:
- **Sheet Name**: "nutrition_creator_reports"
- **9 tabs**: One per formula (Formula_1_18-33s, Formula_2_18-33s, ..., Formula_9_60-90s)
- **Each tab structure**: Sections as rows with labeled headers (Pattern Name, The Proof, Contrastive Analysis, etc.)

---

#### 3.2: `extract_client_data.py`

**Purpose**: Extract hashtag intelligence dashboard data for client executive report

**Input**:
- Stage 1 `winner_analysis.json` (bucket distributions)
- Stage 1 `cluster_analytics.json` (duration stats)
- Stage 6 `rf_video_analysis.json` (ML metrics)
- Stage 7 `winning_formulas.json` (formula names for Page 3)

**Output Format**:
```
=== PAGE 1: SCALE OF ANALYSIS ===
Hashtag: [#nutrition]
Analysis Period: [Past 2-3 months]
Videos Analyzed: [480]
Analysis Mode: [Top performers (engagement-based)]

Total Video Duration: [6.2 hours of content]
Duration Range: [0-120 seconds (8 distinct buckets)]
Content Elements Tracked: [60+ features per video]

--- ANALYSIS METHOD ---
Multi-dimensional machine learning and AI content analysis:

• Visual & Behavioral Pattern Recognition - Advanced ML analyzed 60+ features
  per video (eye contact, pacing, energy levels, scene transitions, gesture
  frequency) to identify what separates top performers from bottom performers

• Content & Messaging Intelligence - AI-powered analysis of video transcripts
  and captions identified trending hook strategies, audience pain points,
  keywords, and engagement tactics unique to #nutrition content

• Formula Discovery - K-Means clustering revealed 3-5 distinct creative strategies
  per video length, validated by Random Forest classification models

Result: 9 proven formulas combining both "how to present" (visuals, pacing)
and "what to say" (hooks, messaging) for complete creative guidance.

=== PAGE 2: HASHTAG INTELLIGENCE DASHBOARD ===

--- SECTION 1: DURATION DISTRIBUTION ---
0-3s:   [8%]
3-9s:   [12%]
9-13s:  [15%]
13-18s: [22%] ← HIGH VOLUME
18-33s: [28%] ← HIGHEST VOLUME
33-60s: [12%]
60-90s: [2%]
90-120s: [1%]

Key Insight: [65% of #nutrition content is 13-33s]

--- SECTION 2: PERFORMANCE BY DURATION ---
0-3s:    [125K] avg views  ⭐⭐
3-9s:    [210K] avg views  ⭐⭐
9-13s:   [380K] avg views  ⭐⭐⭐
13-18s:  [520K] avg views  ⭐⭐⭐⭐
18-33s:  [490K] avg views  ⭐⭐⭐⭐
33-60s:  [310K] avg views  ⭐⭐⭐
60-90s:  [180K] avg views  ⭐⭐
90-120s: [95K] avg views   ⭐

Sweet Spot: [13-33s (highest views + sufficient volume)]

--- SECTION 3: CREATOR PROFILE PRIORITIES ---
TIER 1 (Immediate Hire):
• [13-18s Creators (highest performance: 520K avg views)]
• [18-33s Creators (strong performance + volume: 490K avg views)]
• [33-60s Creators (proven success: 310K avg views)]

Note: These are the 3 winning buckets where top performers cluster most.
Your creative reports focus exclusively on these durations.

=== PAGE 3: YOUR CREATIVE REPORTS ===

--- REPORT DISTRIBUTION ---
Duration Bucket 13-18s:
  • Formula 1: [The Question Hook Formula]
  • Formula 2: [The Fast-Paced Product Demo]
  • Formula 3: [The Myth-Busting Reveal]

Duration Bucket 18-33s:
  • Formula 4: [The Transformation Story]
  • Formula 5: [The Ingredient Deep-Dive]
  • Formula 6: [The Side-by-Side Comparison]

Duration Bucket 33-60s:
  • Formula 7: [The Step-by-Step Tutorial]
  • Formula 8: [The Expert Interview Format]
  • Formula 9: [The Before-After Journey]

Each 2-page report includes:
  • Proof with numbers (engagement differences)
  • Second-by-second execution guide
  • Pre-post checklist
```

**CLI Usage**:
```bash
python extract_client_data.py --client acme --hashtag nutrition --mode top --strategy engagement
```

**Output**: Google Sheet created with URL logged to console and saved to:
`/data/clients/acme/hashtags/nutrition/top_engagement/extracted_client_dashboard_sheet_url.txt`

**Google Sheet Structure**:
- **Sheet Name**: "nutrition_client_dashboard"
- **3 tabs**: Page_1_Scale_of_Analysis, Page_2_Hashtag_Intelligence, Page_3_Your_Reports
- **Each tab structure**: Sections as rows with labeled data fields

---

#### 3.3: `extract_competitor_data.py`

**Purpose**: Extract competitor benchmarking data for single or comparison reports

**Input**:
- Competitor Stage 7 `winning_formulas.json`
- Competitor Stage 6 ML analysis JSONs
- Competitor Stage 1 `winner_analysis.json` (bucket distribution)
- Competitor metadata (handle, posting frequency, top hashtags)
- Client baseline data (for benchmarking)

**Output Format (Single Competitor)**:
```
=== COMPETITOR OVERVIEW ===
Competitor Handle: [@rival_brand]
Client Baseline: [@acme_nutrition]
Analysis Period: [Past 2-3 months]

--- POSTING ACTIVITY ---
Competitor Posts (analyzed): [82 videos]
Client Posts (analyzed): [65 videos]
Posting Frequency Gap: [Competitor posts 26% more]

--- PERFORMANCE BENCHMARKING ---
Competitor Avg Views: [450K]
Client Avg Views: [320K]
Performance Gap: [Competitor gets 41% more views]

Competitor Avg Engagement: [6.2%]
Client Avg Engagement: [4.8%]
Engagement Gap: [Competitor gets 29% more engagement]

--- TOP BUCKETS COMPARISON ---
Competitor Top Bucket: [18-33s (35% of content)]
Client Top Bucket: [13-18s (28% of content)]

Competitor #2 Bucket: [13-18s (22% of content)]
Client #2 Bucket: [18-33s (25% of content)]

--- CREATIVE PATTERNS (Competitor Top 3 Formulas) ---
Formula 1: [The Ingredient Shock Hook]
  Bucket: [18-33s]
  Performance: [8.1% avg engagement]

Formula 2: [The Product Transformation Demo]
  Bucket: [18-33s]
  Performance: [7.8% avg engagement]

Formula 3: [The Quick Win Tutorial]
  Bucket: [13-18s]
  Performance: [7.4% avg engagement]

--- HASHTAG STRATEGY ---
Competitor Top Hashtags:
  1. [#nutrition (45% of posts)]
  2. [#healthylifestyle (38% of posts)]
  3. [#wellness (32% of posts)]

Client Top Hashtags:
  1. [#nutrition (62% of posts)]
  2. [#fitness (28% of posts)]
  3. [#healthtips (22% of posts)]

Insight: [Competitor diversifies hashtags more, client over-indexes on #nutrition]

--- OPPORTUNITIES ---
Gap 1: [Competitor dominates 18-33s bucket - client should invest here]
Gap 2: [Competitor uses shock hooks effectively - client lacks this pattern]
Gap 3: [Competitor hashtag diversification strategy outperforms client focus]
```

**Output Format (Comparison Report - Multi-Competitor)**:
```
=== MULTI-COMPETITOR COMPARISON ===
Client Baseline: [@acme_nutrition]
Competitors Analyzed: [@rival_brand, @competitor2, @competitor3]

--- PERFORMANCE LEADERBOARD ---
Rank 1: [@rival_brand] - [450K avg views, 6.2% engagement]
Rank 2: [@competitor2] - [410K avg views, 5.8% engagement]
Rank 3: [@acme_nutrition] - [320K avg views, 4.8% engagement] ← CLIENT
Rank 4: [@competitor3] - [280K avg views, 4.1% engagement]

--- BUCKET STRATEGY COMPARISON ---
                    | Client  | Rival   | Comp2   | Comp3   |
--------------------|---------|---------|---------|---------|
Top Bucket          | 13-18s  | 18-33s  | 18-33s  | 13-18s  |
% in Top Bucket     | 28%     | 35%     | 40%     | 25%     |
Top Bucket Avg Views| 520K    | 480K    | 510K    | 390K    |

Insight: [Rival and Comp2 dominate 18-33s with high volume - client should shift focus]

--- CREATIVE FORMULA COMPARISON ---
Best Performer Overall: [@rival_brand - The Ingredient Shock Hook (8.1% engagement)]
Client Best Formula: [The Question Hook Formula (5.2% engagement)]
Gap: [Client's best formula underperforms market leader by 2.9 percentage points]

--- HASHTAG STRATEGY COMPARISON ---
Most Diverse: [@rival_brand (uses 12 hashtags regularly)]
Least Diverse: [@acme_nutrition (uses 5 hashtags regularly)]
Recommendation: [Expand hashtag portfolio to reach broader audience]

--- KEY TAKEAWAYS ---
1. [Client ranks 3rd out of 4 in performance - 41% gap vs leader]
2. [Shift content focus to 18-33s bucket (where top 2 competitors dominate)]
3. [Adopt shock hook formula similar to @rival_brand's approach]
4. [Diversify hashtag strategy - currently too narrow]
```

**CLI Usage**:
```bash
# Single competitor
python extract_competitor_data.py --client acme --competitor rival_brand

# Comparison (multiple competitors)
python extract_competitor_data.py --client acme --competitors rival_brand,competitor2,competitor3
```

**Output**: Google Sheets created with URLs logged to console and saved to:
- Single: `/data/clients/acme/competitors/rival_brand/top_engagement/extracted_competitor_sheet_url.txt`
- Comparison: `/data/clients/acme/competitors/comparison_sheet_url_2025-01-28.txt`

**Google Sheet Structure (Single)**:
- **Sheet Name**: "rival_brand_vs_acme"
- **1 tab**: Single sheet with sections as rows (Competitor Overview, Performance Benchmarking, Creative Patterns, Hashtag Strategy, Opportunities)

**Google Sheet Structure (Comparison)**:
- **Sheet Name**: "acme_competitor_comparison"
- **1 tab**: Single sheet with comparison tables (Performance Leaderboard, Bucket Strategy Comparison, Formula Comparison, Hashtag Strategy, Key Takeaways)

---

### Section 4: Documentation (1 day)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 4.1 | Write extraction script instructions | Developer | 0.5 days | CLI usage, troubleshooting |
| 4.2 | Create template population guide | Designer + Developer | 0.5 days | Step-by-step copy-paste workflow |

**Deliverables**:
- `STAGE8_EXTRACTION_GUIDE.md` - How to run scripts, interpret output
- `TEMPLATE_POPULATION_GUIDE.md` - How to populate each template with extracted data
- Video tutorial (optional): 10-min walkthrough of full workflow

---

### Section 5: Testing (0.5 days)

| # | Task | Owner | Effort | Notes |
|---|------|-------|--------|-------|
| 5.1 | Test extraction scripts on real data | Developer | 0.25 days | Run on actual Stage 1-7 outputs |
| 5.2 | Generate 1 sample PDF of each type | Designer + Developer | 0.25 days | Validate template + data integration |

**Test Cases**:
- Extract data from `#nutrition` hashtag analysis (output to Google Sheets)
- Open Google Sheets and verify data accuracy
- Populate all 4 templates from Google Sheets data
- Export PDFs and verify mobile rendering (Template A)
- Confirm all placeholders populated correctly

---

## Total MVP Effort: ~17 days

| Section | Tasks | Effort |
|---------|-------|--------|
| Section 0: Template Structures | 4 tasks | 0 days (Tasks 0.1, 0.2 ✅ COMPLETE, 0.3, 0.4 remaining) |
| Section 1: Designer Templates | 4 tasks | 8 days (includes QR code placeholders in Template A) |
| Section 2: Branding Package | 3 tasks | 3 days |
| Section 3: Data Extraction Scripts (Google Sheets) | 3 tasks | 3.25 days (includes QR code generation) |
| Section 4: Documentation | 2 tasks | 1 day |
| Section 5: Testing | 2 tasks | 0.5 days |
| **TOTAL** | **18 tasks** | **15.75 days (~3 weeks)** (2 tasks complete, 16 remaining) |

**Scope Changes from Issue Resolutions**:
- Task 0.2 ✅ COMPLETE: Hashtag → Creator template (Stage8MVP_Reports.md Section 2)
- Task 3.1 updated: +0.25 days for QR code generation (Issue 1: Visual Examples)
- Template A updated: Includes 2 QR code placeholders per report

**Parallelizable**: Designer work (Sections 1-2: 11 days) + Development work (Section 3: 3.25 days) can run simultaneously after Section 0 complete

**Critical Path**:
1. Section 0 (0.75 days remaining: Tasks 0.3, 0.4) → BLOCKS everything
2. Section 1-2 (11 days) designer work in parallel with Section 3 (3 days) dev work
3. Section 4-5 (1.5 days) sequential after above

**Actual Calendar Time**: ~2.5 weeks (if parallelized) to ~3.5 weeks (if sequential)

---

## Phased Approach Recommendation

### Phase 1 (NOW): Designer Template MVP
**Timeline**: 3.5 weeks
**Effort**: 16.75 days development (1.25 days template structures + 3 days scripts + 11 days designer + 1.5 days docs/testing)
**Use for**: Onboarding (5 hashtags, 5-7 competitors)
**Manual time investment**: ~25 hours total onboarding + ~2 hrs/month ongoing

**Why start here**:
- Get to production 10x faster
- Validate report content and design with real clients
- Learn what clients actually want before automating

---

### Phase 2 (IF NEEDED): Partial Automation
**Trigger**: Clients request weekly reports OR 5+ active clients
**Timeline**: 2 weeks
**Focus**: Automate ONLY the most repetitive part (Creator PDFs - 9 per hashtag)

**What to automate**:
- Creator report generation (Template A) - saves ~3 hours per hashtag
- Keep client + competitor reports manual (less frequent)

**Development**: ~10 days
**Savings**: ~15 hours/month (if running 5 hashtag analyses/month)

---

### Phase 3 (SCALE): Full Automation
**Trigger**: 10+ active clients OR 20+ hours/month manual work
**Timeline**: 8 weeks
**Scope**: Implement full Stage8Planning.md

**What to automate**:
- All 4 report types (creator, client, single competitor, comparison)
- Full PDF generation engine
- Batch processing, error handling, version control

**Development**: ~50 days (original automated MVP)
**Savings**: All manual work eliminated

---

## Required Resources

### Software Licenses
- **Adobe InDesign** (~$30/month) OR **Canva Pro** (~$13/month) OR **Figma** (free tier OK)
- **Python 3.8+** (free)
- ✅ **Google Workspace** (REQUIRED - for Google Sheets API access)

### Skills Needed
- **Designer**: Adobe InDesign/Canva proficiency, brand identity design
- **Developer**: Python, JSON parsing, basic data transformation
- **You**: Willingness to spend ~25 hours on manual work during onboarding

### Time Commitment
- **Upfront**: 3 weeks (designer + dev work in parallel)
- **Onboarding**: ~25 hours manual work across 5 hashtags + 7 competitors
- **Ongoing**: ~30 min every 2 weeks (biweekly client reports)

---

## Success Criteria

### MVP Complete When:
1. ✅ All 4 designer templates finalized and tested
2. ✅ All 3 extraction scripts working on real Stage 1-7 data
3. ✅ 1 sample PDF generated for each template type
4. ✅ Documentation complete (extraction + template guides)
5. ✅ Mobile rendering validated for Template A (creator reports)

### Production-Ready When:
1. ✅ 5 hashtag analyses completed (45 creator PDFs + 5 client PDFs generated)
2. ✅ 5 single competitor reports generated
3. ✅ 2 comparison reports generated
4. ✅ Client feedback collected and templates iterated
5. ✅ Team trained on extraction + population workflow

---

## Next Steps

### Immediate Actions (Week 1)
1. ✅ **COMPLETE Section 0 first** - Create 2 remaining template structures (0.3, 0.4) before anything else
2. ✅ Select design software (InDesign vs Canva vs Figma)
3. ✅ Hire/assign designer
4. ⏸️ **WAIT for Section 0** - Designer cannot start until all template structures complete
5. ⏸️ **WAIT for Section 0** - Developer can start Section 3 after Section 0 complete

### Week 2
1. ✅ Designer completes Templates A + B
2. ✅ Developer completes scripts 3.1 + 3.2
3. ✅ Test extraction scripts on real data
4. ✅ Generate first sample PDFs (creator + client)

### Week 3
1. ✅ Designer completes Templates C + D
2. ✅ Developer completes script 3.3
3. ✅ Generate sample PDFs (competitor single + comparison)
4. ✅ Write documentation (extraction + population guides)
5. ✅ Final testing and validation

### Week 4+ (Production Use)
1. ✅ Run first onboarding hashtag analysis → generate 9 creator + 1 client PDFs
2. ✅ Collect client feedback
3. ✅ Iterate on templates if needed
4. ✅ Continue onboarding (remaining 4 hashtags + 7 competitors)

---

## Open Questions for Decision

1. ✅ **Output Format** - RESOLVED: Google Sheets (easiest to review/edit)
2. **Design Software**: Adobe InDesign, Canva Pro, or Figma? (Affects designer workflow and license cost)
3. **Template D Priority**: Should we defer Comparison Report template to Phase 2? (Only needed 2 times during onboarding vs 5 times for single competitor)
4. **Quality Assurance**: Who will review PDFs before sending to clients? (Peer review vs self-review)

**Recommendation**: Answer questions 2-4 before starting development to avoid rework.

---

## Appendix: File Structure

### Extraction Script Outputs

```
/data/clients/{client_id}/
├── hashtags/{hashtag}/{mode}_{strategy}/
│   ├── extracted_creator_reports_sheet_url.txt    # Google Sheets URL (Output of 3.1)
│   ├── extracted_client_dashboard_sheet_url.txt   # Google Sheets URL (Output of 3.2)
│   └── final_reports/
│       ├── nutrition_18-33s_formula_1.pdf         # Manually generated from Template A
│       ├── nutrition_18-33s_formula_2.pdf
│       ├── ... (9 creator PDFs total)
│       └── nutrition_client_report.pdf            # Manually generated from Template B
│
└── competitors/{competitor_handle}/{mode}_{strategy}/
    ├── extracted_competitor_sheet_url.txt         # Google Sheets URL (Output of 3.3 single)
    ├── comparison_sheet_url_2025-01-28.txt        # Google Sheets URL (Output of 3.3 comparison)
    └── final_reports/
        ├── rival_brand_vs_acme_intel.pdf          # Manually generated from Template C
        └── acme_competitor_comparison.pdf         # Manually generated from Template D
```

**Note**: Google Sheets URLs are also logged to console during script execution for easy access

### Designer Template Files

```
/design_assets/
├── templates/
│   ├── Template_A_ContentCreator_v1.indd      # InDesign source
│   ├── Template_B_ClientExecutive_v1.indd
│   ├── Template_C_SingleCompetitor_v1.indd
│   └── Template_D_Comparison_v1.indd
│
├── brand_assets/
│   ├── TumiLabs_Logo.svg
│   ├── icon_library/
│   ├── chart_templates/
│   └── brand_style_guide.pdf
│
└── documentation/
    ├── STAGE8_EXTRACTION_GUIDE.md
    └── TEMPLATE_POPULATION_GUIDE.md
```

---

**Status**: ✅ **READY TO START** - Section 0 (template structures) is the critical path blocker. Complete Tasks 0.3, 0.4 before designer/dev work begins.

**Next Action**: Create Handle/Single Competitor → Client template structure (Task 0.3)
