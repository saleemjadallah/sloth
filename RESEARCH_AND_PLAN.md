# Sloth — AI Ad Creation Platform
## Research & Architecture Plan (March 2026)

---

## 1. EXECUTIVE SUMMARY

Build a self-hosted AI ad creation platform that replaces $100-400/mo SaaS tools (Superscale.ai, Arcads) by directly orchestrating the same underlying AI APIs they use. The platform will:

1. **Analyze any website** to extract brand identity (voice, colors, logos, audience)
2. **Generate ad scripts** using LLMs with platform-specific best practices baked in
3. **Create video/image ads** using NanoBanana Pro (images) + Veo 3.1/Runway (video)
4. **Post to socials** via Zernio with optimized captions for Meta & TikTok

**Cost advantage:** Instead of paying $11/video (Arcads) or $49-399/mo (Superscale), you pay raw API costs: ~$0.05-0.50/image, ~$0.50-6/video, ~$0.01-0.05/LLM call.

---

## 2. COMPETITIVE INTELLIGENCE

### What Superscale.ai Actually Does
- **Pricing:** $49-399/mo
- **Core tech:** ElevenLabs v3 (voice), Higgsfield (avatars), LLM (scripts/research)
- **Real value:** Workflow orchestration — paste URL → research → generate → edit → publish
- **70+ ad templates**, 200+ AI characters, built-in video editor, competitor research tool
- **What's replicable:** Everything except the curated template library with performance data

### What Arcads Actually Does
- **Pricing:** $110+/mo, ~$11/video
- **Core tech:** Motion-captured real performers (their moat), TTS engine, Sora 2 Pro + Veo 3.1 on premium tiers
- **Real value:** Massive avatar library + batch generation (CSV upload → hundreds of variants)
- **What's replicable:** The pipeline. Not easily the motion-capture actor library (but AI avatars are catching up fast)

### Our Advantage
- **Direct API access** to the same underlying models at raw cost
- **Multi-LLM access** — can use best model for each task
- **Custom to your businesses** — no generic templates, tailored to your brands
- **No per-seat/per-video pricing** — only pay for what you generate

---

## 3. TECH STACK & ARCHITECTURE

### Core Architecture

```
┌─────────────────────────────────────────────────────┐
│                 FRONTEND (Next.js)                    │
│  Dashboard · Brand Manager · Ad Builder · Analytics  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               API GATEWAY (FastAPI)                   │
│         Auth · Rate Limiting · Job Dispatch           │
└──┬──────────┬──────────┬──────────┬────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│Brand │ │Script  │ │Creative│ │Publishing│
│Engine│ │Engine  │ │Engine  │ │Engine    │
└──┬───┘ └───┬────┘ └───┬────┘ └────┬─────┘
   │         │          │           │
   ▼         ▼          ▼           ▼
Firecrawl  LLMs     Video/Image   Zernio
Brand.dev  (Claude,  Gen APIs      API
           GPT,      (Veo 3.1,
           Gemini)   NanoBanana,
                     Runway)
                        │
                  ┌─────▼──────┐
                  │ BullMQ +   │
                  │ Redis      │
                  │ (Job Queue)│
                  └─────┬──────┘
                        │
              ┌────┬────┼────┬────┐
              ▼    ▼    ▼    ▼    ▼
            Text Image Video Audio Caption
            Worker Worker Worker Worker Worker
```

### Tech Choices

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | Next.js + TypeScript + Tailwind | SSR, streaming LLM responses, fast iteration |
| **Backend API** | FastAPI (Python) | Async, great for AI pipeline orchestration |
| **Job Queue** | BullMQ + Redis | Battle-tested for video gen jobs, retries, progress tracking |
| **Database** | PostgreSQL | Brand profiles, campaigns, generated assets metadata |
| **Storage** | Cloudflare R2 | Generated videos/images, cheaper than S3 |
| **LLM Orchestration** | LangChain or direct SDK calls | Multi-model routing (Claude for polish, GPT for brainstorm, Gemini for multimodal) |
| **Auth** | Clerk or NextAuth | Quick setup, supports multiple users if needed |
| **Deployment** | Vercel (frontend) + Railway/Fly.io (backend) | Simple, scales when needed |

---

## 4. THE FOUR ENGINES

### Engine 1: Brand Analysis

**Input:** A website URL
**Output:** Complete brand profile (JSON)

**How it works:**
1. **Firecrawl** scrapes the website → extracts markdown content + branding (logo, colors, fonts)
2. **Brand.dev / Brandfetch** supplements with high-quality logos, color palettes
3. **LLM analysis pass** (Claude) reads the markdown and extracts:
   - Brand voice & tone (casual/formal, fun/serious, etc.)
   - Value propositions
   - Target audience demographics
   - Key products/services
   - Competitive positioning
   - Emotional triggers

**Brand Profile Schema:**
```json
{
  "company_name": "string",
  "logo_url": "string",
  "colors": { "primary": "#hex", "secondary": "#hex", "accent": "#hex" },
  "fonts": { "heading": "string", "body": "string" },
  "voice": { "tone": "string", "style": "string", "personality_traits": ["string"] },
  "value_propositions": ["string"],
  "target_audience": { "demographics": "string", "pain_points": ["string"], "desires": ["string"] },
  "products": [{ "name": "string", "description": "string", "key_benefits": ["string"] }],
  "competitors": ["string"],
  "industry": "string"
}
```

**APIs & Costs:**
| Service | Purpose | Cost |
|---------|---------|------|
| Firecrawl | Scrape + branding extraction | ~$0.01-0.05/page |
| Brand.dev | Logo + color supplement | Free tier available |
| Claude API | Brand voice analysis | ~$0.01-0.05/analysis |

---

### Engine 2: Script Generation

**Input:** Brand profile + platform + ad type + product
**Output:** Multiple ad script variations with captions

**How it works:**
1. Takes brand profile as context
2. Applies platform-specific rules (see best practices below)
3. Generates 5-10 script variations per request
4. Each script follows the Hook → Problem → Solution → Proof → CTA structure
5. Generates platform-optimized captions with each script

**Platform Rules (Baked Into Prompts):**

**Meta (Facebook/Instagram):**
- Video: 15-30 sec, 9:16 vertical, 1080x1920
- Caption: Front-load first 125 chars (truncation point), max 2,200
- Headline: 40 chars max
- Max 5 hashtags (Instagram 2026 cap)
- UGC style: raw/phone-shot aesthetic, 30%+ higher CTR than polished

**TikTok:**
- Video: 9-15 sec sweet spot, 9:16 vertical, 1080x1920
- Caption: 100 chars max (including hashtags)
- Safe zones: keep text away from top 200px, right 120px, bottom 300px
- Switch visuals every 2-3 seconds
- Mobile-shot style: 63% higher conversion than studio
- Refresh creative every 7 days

**Script Archetypes:**
| Type | Structure | Best For |
|------|-----------|----------|
| Problem/Solution | Pain → product → relief | Most products |
| Testimonial | Story → discovery → results | Trust building |
| Unboxing | Anticipation → reveal → reaction | Physical products |
| Before/After | Old state → product → transformation | Results-driven |
| Tutorial | "Let me show you..." → demo | Complex products |
| Comparison | Competitor vs. product | Market differentiators |

**Hook Formula Bank (Top Performers):**
- "Stop doing [X], start doing [Y]"
- "Why I ditched [competitor] for good"
- "I started [using product] and here's what happened"
- "Nobody talks about this but..."
- "POV: [pain point] is finally gone"
- "[Number] [niche] hacks you've never heard of"

**LLM Strategy:**
| Task | Best Model | Why |
|------|-----------|-----|
| Brainstorm variations | GPT-5.2 | More diverse, creative hooks |
| Final copy polish | Claude Opus 4.6 | Best brand voice consistency |
| Multilingual | Gemini 3 Pro | Best multilingual capability |

**Cost:** ~$0.01-0.05 per script batch (5-10 variations)

---

### Engine 3: Creative Generation

**Input:** Ad script + brand profile + asset type (image/video)
**Output:** Ready-to-post ad creative

**Image Generation (NanoBanana Pro / Gemini 3 Pro Image):**
- 4K resolution, industry-leading text rendering
- Perfect for: static ads, product shots, infographics, carousel images
- Identity preservation across up to 5 subjects
- API: Gemini API (`gemini-3-pro-image-preview`) or Vertex AI
- **Cost: $0.05-0.24/image**

**Video Generation (Multi-Provider Strategy):**

| Provider | Best For | API Cost/sec | Notes |
|----------|---------|-------------|-------|
| **Veo 3.1** (primary) | 4K video + audio sync | $0.10 (fast) - $0.75 (w/ audio) | Native 9:16, <120ms lip-sync |
| **Runway Gen-4.5** (premium) | Highest quality creative | $0.05-0.12 | Top benchmark scores |
| **MiniMax Hailuo 2.3** (budget) | Bulk batch creation | $0.04-0.09 | Best cost per video |
| **Seedance 2.0** (multilingual) | Multi-language lip-sync | Varies | 8+ language lip-sync |

**Video Generation Pipeline:**
1. Generate reference frames with NanoBanana Pro (brand-consistent imagery)
2. Feed reference frames + script to video model (Veo 3.1 "Ingredients to Video" accepts 4 refs)
3. Generate audio/voiceover (Veo 3.1 native audio OR ElevenLabs API)
4. Scene Extension for videos >8 sec
5. Post-processing: add captions, brand watermark, safe-zone-aware text overlays

**Cost per finished ad (estimated):**
| Component | Cost |
|-----------|------|
| Brand analysis | $0.05 |
| Script generation (10 variants) | $0.05 |
| Reference images (2-4) | $0.20-0.50 |
| Video generation (15 sec) | $1.50-7.50 |
| Audio/voice | $0.10-0.50 |
| **Total per ad** | **$1.90-8.60** |

vs. Arcads at $11/video + $110/mo subscription.

---

### Engine 4: Publishing

**Input:** Generated creative + caption + target platforms + schedule
**Output:** Published/scheduled posts

**Zernio (formerly Late.dev) Integration:**
- Single API for 14 platforms (Instagram, Facebook, TikTok, YouTube, Twitter, LinkedIn, etc.)
- Supports: text, images, videos, Reels, Stories, Shorts, carousels
- Scheduling, drafts, queues, analytics
- SDK: `@zernio/node` (Node.js) or `zernio-sdk` (Python)
- **Cost: $16-49/mo** (Build/Accelerate plan)

**Publishing Workflow:**
1. Upload generated media via Zernio media upload
2. Attach platform-optimized caption (different per platform)
3. Schedule or publish immediately
4. Cross-post to multiple platforms in one API call
5. Track analytics via Zernio analytics endpoint

**Caption Optimization Per Platform:**
| Platform | Max Length | Key Rules |
|----------|-----------|-----------|
| Instagram | 2,200 chars | Hook in first 125 chars, max 5 hashtags, keywords > hashtags |
| Facebook | 2,200 chars | Hook in first 80 chars, storytelling + social proof |
| TikTok | 100 chars | Ultra-concise, hashtags count toward limit, no clickable links |

**Fallback:** Direct Meta Business API + TikTok Content Posting API if Zernio ever becomes limiting (but this adds significant engineering complexity).

---

## 5. DATABASE SCHEMA (Core Tables)

```sql
-- Brand profiles extracted from websites
brands (
  id, user_id, name, website_url, logo_url,
  colors JSONB, fonts JSONB, voice JSONB,
  value_propositions JSONB, target_audience JSONB,
  products JSONB, industry,
  created_at, updated_at
)

-- Generated ad campaigns
campaigns (
  id, brand_id, name, objective,
  target_platforms TEXT[], -- ['instagram', 'tiktok', 'facebook']
  status, -- draft, generating, ready, published
  created_at
)

-- Individual ad creatives within a campaign
ad_creatives (
  id, campaign_id, brand_id,
  script_text, caption_meta, caption_tiktok,
  video_url, thumbnail_url, image_urls TEXT[],
  ad_type, -- 'ugc_video', 'static', 'carousel', 'slideshow'
  platform_specs JSONB, -- aspect ratio, duration, etc.
  generation_cost DECIMAL,
  status, -- pending, generating, ready, published, failed
  created_at
)

-- Job tracking for async generation
generation_jobs (
  id, ad_creative_id, job_type, -- 'image', 'video', 'audio', 'script'
  provider, -- 'veo_3.1', 'nanobanana', 'runway', 'claude'
  status, progress_pct, error_message,
  input_params JSONB, output_url,
  cost DECIMAL,
  started_at, completed_at
)

-- Social media publishing records
publications (
  id, ad_creative_id, platform,
  zernio_post_id, published_url,
  scheduled_for, published_at,
  analytics JSONB -- clicks, views, engagement
)
```

---

## 6. MVP FEATURE ROADMAP

### Phase 1: Foundation (Week 1-2)
- [ ] Project scaffolding (Next.js + FastAPI + PostgreSQL + Redis)
- [ ] Brand Analysis Engine — paste URL → get brand profile
- [ ] Basic dashboard to view/manage brand profiles

### Phase 2: Script Engine (Week 3)
- [ ] Script generation with multi-LLM support
- [ ] Platform-specific caption generation
- [ ] Hook library + script archetype templates
- [ ] Script variation management UI

### Phase 3: Creative Engine (Week 4-5)
- [ ] Image generation via NanoBanana Pro API
- [ ] Video generation via Veo 3.1 API
- [ ] BullMQ job queue for async video generation
- [ ] Asset library (view/download/manage generated creatives)
- [ ] Cost tracking per generation

### Phase 4: Publishing Engine (Week 6)
- [ ] Zernio integration for cross-platform posting
- [ ] Caption optimization per platform
- [ ] Scheduling + queue management
- [ ] Publishing status tracking

### Phase 5: Polish & Scale (Week 7-8)
- [ ] Batch generation (generate 10-50 ad variations at once)
- [ ] A/B test variant creation
- [ ] Analytics dashboard (via Zernio analytics)
- [ ] Multi-brand support (switch between your businesses)
- [ ] Cost optimization (route to cheapest provider per job type)

---

## 7. API KEYS & ACCOUNTS NEEDED

| Service | Purpose | Signup |
|---------|---------|--------|
| **Firecrawl** | Website scraping + branding | firecrawl.dev |
| **Google Cloud (Vertex AI)** | Veo 3.1 + NanoBanana Pro | console.cloud.google.com |
| **Anthropic** | Claude API for scripts/analysis | console.anthropic.com |
| **OpenAI** | GPT for brainstorming variations | platform.openai.com |
| **Google AI Studio** | Gemini API (alternative to Vertex) | aistudio.google.com |
| **Runway** | Gen-4.5 video (premium quality) | dev.runwayml.com |
| **MiniMax** | Hailuo video (budget option) | platform.minimax.io |
| **Zernio** | Social media publishing | zernio.com |
| **ElevenLabs** | Voice generation (optional) | elevenlabs.io |
| **Cloudflare** | R2 storage for assets | cloudflare.com |

---

## 8. COST COMPARISON

### Current: Using SaaS Tools
| Tool | Monthly Cost |
|------|-------------|
| Arcads (20 videos) | $220/mo |
| OR Superscale (mid-tier) | $99/mo |
| Social posting tool | $20-50/mo |
| **Total** | **$120-270/mo** |

### With Sloth (Self-Built)
| Component | Monthly Cost (est. 50 ads/mo) |
|-----------|-------------------------------|
| Veo 3.1 video generation | $50-150 |
| NanoBanana Pro images | $5-15 |
| LLM calls (scripts/analysis) | $5-10 |
| Zernio (Accelerate) | $49 |
| Firecrawl | $16 |
| Hosting (Vercel + Railway) | $20-40 |
| Cloudflare R2 storage | $5-10 |
| **Total** | **$150-290/mo** |

**Break-even:** At ~50 ads/month, costs are similar. But with Sloth you get:
- No per-video caps
- Full control over quality and models
- Custom to your brands
- Scales down when you need fewer ads
- Can add new AI models instantly as they drop

At **100+ ads/month**, Sloth becomes dramatically cheaper ($200-350 vs $440+ on Arcads).

---

## 9. KEY RISKS & MITIGATIONS

| Risk | Mitigation |
|------|-----------|
| AI video quality not good enough for ads | Multi-provider fallback (Veo → Runway → MiniMax); quality improves monthly |
| API rate limits during batch generation | BullMQ queues with backoff; spread across providers |
| TikTok/Meta reject AI-generated content | Follow platform guidelines; SynthID watermarking; consider FTC disclosure |
| Zernio API changes or goes down | Abstract publishing behind interface; fallback to direct Meta/TikTok APIs |
| Video generation costs spike | Budget tracking per job; auto-route to cheapest provider meeting quality threshold |
| Lip-sync quality for UGC talking-heads | Use Veo 3.1 (<120ms sync) or ElevenLabs + dedicated lip-sync API |

---

## 10. LEGAL CONSIDERATIONS

- **FTC (US):** Fake testimonials by non-existent people = $51K fine per violation. If using AI avatars, must disclose.
- **Platform policies:** Meta auto-labels AI images (not video yet). TikTok requires disclosure of AI-generated content.
- **SynthID:** Veo 3.1 embeds watermarks automatically. Runway may not.
- **Recommendation:** Add "Created with AI" disclosure to ad copy when using AI avatars/voices. For product demos and non-testimonial formats, risk is lower.

---

*Research completed March 25, 2026. Ready to begin implementation.*
