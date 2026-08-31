---
name: humanizer
description: Detect and remove AI writing patterns to produce natural, human-sounding prose
category: creative
contexts: [heartbeat, chat]
aliases: [humanize, natural, rewrite, prose, tone, polish]
bound_tools: [humanize_text, post_process_output]
---

# AI Writing Pattern Detection and Removal

Analyze text for telltale AI-generated writing patterns and rewrite it to sound natural, varied, and human-authored.

## When to Use

- When the user asks to "humanize" or "make this sound more natural" or "remove the AI voice"
- When drafting outgoing communications (emails, posts, messages) that should not read as AI-generated
- When editing content that will be published under the user's name
- When the user flags specific text as "sounding too AI"

## Step-by-Step Methodology

1. **Analyze the input**: Before rewriting, identify the specific AI patterns present. Common tells include:
   - **Hedging openers**: "It's important to note that", "It's worth mentioning", "Interestingly"
   - **Formulaic structure**: Intro paragraph, three body paragraphs, conclusion that restates the intro
   - **Exhaustive listing**: Every possible angle covered with equal weight, nothing omitted or prioritized
   - **Sycophantic tone**: "Great question!", "Absolutely!", "That's a fantastic point"
   - **Hollow intensifiers**: "Incredibly", "Remarkably", "Fundamentally", "Essentially"
   - **Emoji and exclamation overuse**: Forced enthusiasm that reads as performative
   - **Symmetrical phrasing**: "Not only X, but also Y", parallel constructions repeated throughout
   - **Meta-commentary**: "Let me break this down", "Here's the thing", "To put it simply"

2. **Determine the target voice**: Ask or infer the desired tone. Options include:
   - Casual conversational (texting a friend)
   - Professional but warm (work email)
   - Authoritative and concise (executive brief)
   - The user's own voice (if prior writing samples exist in memory, match them)

3. **Rewrite with human patterns**: Apply transformations that mimic natural writing:
   - **Vary sentence length dramatically**. Mix 4-word fragments with 30-word run-ons. Real people do not write in uniform 15-word sentences.
   - **Use contractions and colloquialisms** appropriate to the register.
   - **Cut the throat-clearing**. Start sentences with the point, not a preamble.
   - **Allow imperfection**. Occasional sentence fragments, starting with "And" or "But", ending with prepositions -- these are features of natural prose, not bugs.
   - **Prioritize ruthlessly**. Real writers omit what does not matter. Drop the "comprehensive" instinct.
   - **Use specific details** instead of generic descriptors. "The 3pm demo" not "the upcoming presentation."

4. **Call the tool**: Pass the original text and target voice to `humanize_text`, which applies rule-based pattern removal and optionally an LLM rewrite pass.

5. **Review the output**: Check that the rewrite preserved the original meaning and key information. Humanizing should change the voice, not the substance.

## Quality Guidelines

- The goal is natural prose, not dumbed-down prose. Sophisticated vocabulary and complex ideas are fine; robotic cadence and formulaic structure are not.
- Never add falsehoods or change factual claims during humanization. This is a voice transformation, not a content edit.
- When the user's own writing samples exist in memory, use them as the gold standard for voice matching.
- If the input text is already natural-sounding, say so rather than rewriting for the sake of it.
- Do not over-correct. Some AI patterns (clear structure, logical flow) are also features of good writing. Remove the tells, keep the clarity.
- Respect the register. A legal brief should not sound like a blog post, and a casual message should not sound like a press release.
