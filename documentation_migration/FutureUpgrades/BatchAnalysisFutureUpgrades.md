Future Upgrades

P1 = Priority 1
P2 = Priority 2

## P1 

### Stage 1
#### Video Selection by Engagement 
You can modify the logic of Video Discovery and Selection to include engagement ratings for each bucket
_right now its only for top 3 winning_


### Stage 8
#### Reporting Tables - Full Bucket Data Obtention
Currently you report top 3 buckets of videos and engagement rating because you do not collect data on the "losing" buckets.

You could collect this data to have richer tables in Reporting

#### Specific Analysis of CREATED content
_not sure if stage 8_
Analysis of the non-reposted content, and the structure of that content 

#### Content Analysis 
**Hashtag Breakdown:** Currently you do not identify the types of hashtag used in posts.
```
It's doable, but you'd need to classify hashtags (through LLM) to understand which are 
- Mass Communication: #fyp, #wellnesstok ... etc
- Niche groups: 

Option 1: Leave it to LLM to discover (Would be an educated "guess")
- Pros: Easier, faster.

Option 2: Study hashtag classifications per segment and via Python, cross reference the hashtags of a post (obtained through description, you have it all)
- With a central database of what is considered MASSIVE
    - Whatever is not massive, is niche

Pros: No need to mess with LLM Prompt and Stage 7. This seems doable...
```
#### Content Analysis Qualitative CTA 
ContentAnalysisCHILDTI.md → Update prompt to include CTA



## P2
