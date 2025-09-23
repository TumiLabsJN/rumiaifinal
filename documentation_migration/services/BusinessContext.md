  📄 Document 1: BUSINESS_CONTEXT.md (NEW - 1 page)

  Purpose: Answer "Why does RumiAI exist?"
  - Business problem (TikTok virality prediction)
  - Market opportunity (creators need data)
  - Our hypothesis (temporal patterns predict success)
  - Success metrics (what makes this valuable)
  - ROI/Value proposition


# Introduction
RumiAI is part of RippleOS, a consultancy product by my company Tumi Labs. RippleOS helps automate and maximize growth via content creators on TikTok.
RumiAI is not a client facing product, it is used for internal processes as part of the consultancy that we do. 

# Business Problem
Content Creator Marketing is not easy and is resource intensive. Extracting viral creative patterns per hashtag/competitor/creator is a resource-heavy operation which is done manually by teams. Manual dissection of viral trends ends up being processed semantically and without structure, making it difficult to create replicable reports. Replicating these creative patterns is crucial as it directly impacts how many views and engagement a video gets, which consequently impacts how many sales a brand can generate through its' content creators. 

Knowing the viral creative patterns, to in turn coach the brands' affiliate content creators will maximize the time-to-sales of these content creators who will most likely value this type of assistance we give them, as it is uncommon in this industry. Normally content creators are paid per video or paid through commissions and left to fend for themselves

Automating this analysis and
Build a Machine Learning training pipeline on top of RumiAI's Python-only processing system to identify and extract viral creative patterns from TikTok videos, segmented by client industry and video duration.

### Business Model Clarification
- **Clients**: Brands (e.g., nutritional supplement companies, functional drink companies)
- **Affiliates**: Content creators who promote these brands through TikTok videos
- **Value Chain**: We analyze viral content → Generate creative recommendations → Provide reports to affiliates → Affiliates create better promotional content for brands

### Core Value Proposition
Transform raw video analysis data (>100 features per video, exact count TBC) into **duration-specific** actionable creative insights delivered to brand affiliates, recognizing that successful patterns vary dramatically between 15-second and 120-second content. Each duration bucket receives its own ML model and creative recommendations.


### Primary Customer: Brands/Clients
- **Definition**: Companies (e.g., nutritional supplement brands, functional drink companies) who pay for our ML-driven content strategy services
- **What they receive**: Executive-level reports showcasing analysis depth and strategic insights
- **Role**: Pay for service, receive high-level insights, but DO NOT execute content strategies themselves
- **Relationship**: All interactions with content creators flow through Tumi Labs (no direct brand-creator relationship)

## 📊 1.6 Stakeholder Model & Value Flow

### Primary Customer: Brands/Clients
- **Definition**: Companies (e.g., nutritional supplement brands, functional drink companies) who pay for our ML-driven content strategy services
- **What they receive**: Executive-level reports showcasing analysis depth and strategic insights
- **Role**: Pay for service, receive high-level insights, but DO NOT execute content strategies themselves
- **Relationship**: All interactions with content creators flow through Tumi Labs (no direct brand-creator relationship)

### Value Delivery Chain
```
Tumi Labs (ML Analysis & Intermediary) → Brands/Clients (Pay) → UGC Factories/Content Creators (Execute)
                     ↑__________________________|
                     All communication flows through Tumi Labs
```

### Content Execution Partners
1. **UGC Factories** (User Generated Content Factories)
   - Professional content production companies
   - Receive identical PDF reports as individual creators
   - Execute content strategies at scale

2. **Content Creators/Affiliates** (terms used interchangeably)
   - Individual creators who promote brands
   - Receive identical PDF reports as UGC factories
   - Implement viral content strategies based on our ML insights

### Deliverable Differentiation
- **For Brands/Clients**: High-level executive reports demonstrating research depth and ROI potential
- **For UGC/Creators**: Actionable PDF reports with specific creative strategies to test and implement (identical for both UGC and individual creators)
- **Key Distinction**: Brands receive insights for strategic understanding; Creators receive instructions for tactical execution

### Business Model Clarification
- **Model Type**: B2B2C with Tumi Labs as complete intermediary
- **Revenue Source**: Brands/Clients pay Tumi Labs
- **Value Creation**: ML insights enable creators to produce higher-performing content
- **Success Metric**: Increased viral/popular content rates for brand-affiliated creators
- **Communication Flow**: All brand-creator interactions managed through Tumi Labs