  You're right - I was overloading SystemArchitecture.md. Here's a better separation of
  concerns:

  📄 Document 1: BUSINESS_CONTEXT.md (NEW - 1 page)

  Purpose: Answer "Why does RumiAI exist?"
  - Business problem (TikTok virality prediction)
  - Market opportunity (creators need data)
  - Our hypothesis (temporal patterns predict success)
  - Success metrics (what makes this valuable)
  - ROI/Value proposition

  📄 Document 2: QUICK_REFERENCE.md (UPDATE)

  Purpose: Navigation hub + operational basics
  - What is RumiAI? (1 paragraph)
  - Document reading order (START HERE section)
  - Key commands (how to run)
  - Key facts (processing time, output format)
  - Directory map to other docs
  Keep as the "front door" - tells you WHERE to go

  📄 Document 3: SystemArchitecture.md (FOCUSED)

  Purpose: Pure technical architecture
  - System components and data flow
  - Service orchestration details
  - Technical dependencies
  - Performance characteristics
  - File structure and code organization
  Remove business context, remove "getting started" - pure architecture

  📄 Document 4: ML_ROADMAP.md (SIMPLIFIED from MLProjectsGrassrootsv2.md)

  Purpose: Future development plans
  - Current state vs future vision
  - ML model training pipeline
  - Planned optimizations
  - Research directions
  Extract just the roadmap parts, make it concise


  🎯 Proposed Reading Flow for Fresh CLI

  graph TD
      A[QUICK_REFERENCE.md<br/>Start Here - 2 min] --> B{What do you need?}
      B -->|Why exists?| C[BUSINESS_CONTEXT.md<br/>1-2 min]
      B -->|How it works?| D[SystemArchitecture.md<br/>10-15 min]
      B -->|What's coming?| E[ML_ROADMAP.md<br/>5 min]
      B -->|Modify service?| F[services/*.md<br/>Phase 1 docs]
      B -->|Add features?| G[features/*.md<br/>Phase 2 docs]
      C --> D
      D --> E