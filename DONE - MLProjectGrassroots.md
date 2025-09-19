# 1. **🎯 Project Overview**

## Short description of the system

A Machine Learning engine built on-top of RumiAI and its’ main script: python3 scripts/rumiai_runner.py, to train data from videos with Machine Learning to reverse engineer the main creative elements we should test from videos per client. Each client will be in different industries (Nutritional supplements vs Functional Drink) so the creative elements that make videos viral per industry will differ.

For example, Client A needs top 30 videos of #nutrition, in each subsegment: 0-15 | 16-30s | 31-60s | 61-90s | 91-120s with the oldest videos from 01/05/2025

We specify: Client, Hashtag Name, Hashtag URL, Number of Videos, Date 

## Why it matters / what problem it solves

This is the main use of RumiAI  - to identify which are the best performing creative variables that engage and hook audiences in TikTok, to create Creative Reports on the top A/B tests we should do through our Content Creators. Each client will be in different industries (Nutritional supplements vs Functional Drink) so the creative elements that make videos viral per industry will differ.

Very important to 

## Who it's for (end users or downstream systems)

Everything is for internal company processes, or myself - with the exception of the final output from this product. The creative elements identified through ML that should be tested need to be presented in an easy to comprehend way, and easy to execute said insights. 

## 2. **📐 System Goals and Non-Goals**

### Goals - Key Functionalities

- Analyze batches of up to 200 videos through scripts/rumiai_runner.py . Can be one by one, doesn’t need to be in parallel
    - If one of the videos being processed fails, the system must have a way to continue from where it left off.
- Train data per clients. Each client will have their own hashtags, and these hashtags must be grouped by client in case we’d need to cross reference content from other videos.
- Create creative reports per hashtag analysis, with up to 5 creative element mixes that should be tested

### 🚫 Non-Goals (things that are out of scope)

- Analyze videos in parallel
- Analyze videos over 120 seconds
- 

## **3. 🧱 Key Components and Flow**

### Diagram (or a step-by-step list)

Script: export MLAnalysis=true &&\ 
write python3 scripts/rumiai_runner.py  

We could use same rumiai_runner.py script but add feature labels to execute this flow. **After sending script, we get asked:**

1. **Client Name**
    1. example: Stateside Grower
        1. Extra Info: This flow would be best: First reply from prompt: “Is this a new client?”
            1. if NO - then we pick from a list of clients. 
            2. If YES - we get asked for client name 
2. **Name of Hashtag:**
    1. example: #nutrition 
        1. Extra Info: Hashtags would be saved per Client Name. a Client like Stateside Grower can have multiple hashtags associated to it 
3. **URL of the hashtag searched on TikTok**
    1. example: https://www.tiktok.com/search?q=%23nutrition&t=1754933430897
4. **Video count per Subgroup we want to analyze**
    1. Extra Info: 50, which would mean 50 videos of each segment 0-15 | 16-30s | 31-60s | 61-90s | 91-120s - or, total of 200 videos
5. **Date limit of videos DD/MM/YYYY**
    1. Extra Info: If date is 01/03/2025 , the logic of selecting 30 videos must be for videos that were launched after 01/03/2025
6. **Apify Video Selection:** Based on Criteria
    1. Extra Info: Video selection has to be based on the conditional Date limit of Videos | Video count per Subgroup 
7. **Video Download:** Videos that match the criteria are downloaded locally and run through whole analysis process one by one.
    1. Extra Info: Initially, lets keep all the videos to test the ML logic and workflow. 
        1. Disclaimer: This is an idea, I do not know whats the best way to do this
8. **Analysis Starts:** rumiai_runner.py starts the analysis per video. Analysis should be saved in \MLAnalysis\[Client Name]\[Hashtag Name]
    1. Extra Info: 
        1. Disclaimer: This is an idea, I do not know whats the best way to do this
9. **ML:** The analysis output JSON is used to train our ML engine per video, by CoreMetrics and the 400+ features we track.
    1. Extra Info: 
        1. Disclaimer: This is an idea, I do not know whats the best way to do this. 
        2. Questions: 
            1. What type of ML do we use for this stage?
10. **ML Analysis:** We send the aggregate analysis outputs to Claude via API for analysis 
    1. **Extra Info: Will we have an issue with JSON size?**
        1. Disclaimer: This is an idea
11. **Creative Report Output:** After training per hashtag and video duration, we get 5 Creative Reports of the most repeated practices
    1. Extra Info: 
        1. Disclaimer: Maybe more creative reports ?

### 4. **🧩 Data Flow / Input & Output Contracts**

I’m not entirely sure. Would have to check .md’s of rumiai_runner.py

- Data types at each step
- Required formats, assumptions
- Example input/output if possible

> Helps downstream LLMs define interfaces clearly.
> 

---

### 5. **📦 Dependencies**

I’m not entirely sure. Would have to check .md’s of rumiai_runner.py

- Internal services
- External APIs
- Libraries or models (e.g., Whisper, Claude, OpenCV, ffmpeg)

> Useful for auto-generating environment setups later.
> 

---

### 6. **⚠️ Known Risks / Complexity Areas**

**Not understanding the Repo well enough to connect the current flow,** 

**The codebase is monolithic, there are a lot of upstream services that ML will depend on** 

**What happens if one of the videos fails (doesn’t pass Service Contract)**

- I think we should continue the video processing and just provide a report of the videos that failed to re analyze them and integrate them to the output

**What kind of JSON will we be sending to Claude via API? We will have output of 50 videos, this won’t be a small file**