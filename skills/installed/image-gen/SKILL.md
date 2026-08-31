---
name: image-gen
description: Generate images from text descriptions using DALL-E or compatible providers
category: creative
requires:
  tools: [generate_image]
  env: [OPENAI_API_KEY]
contexts: [heartbeat, chat]
aliases: [image, picture, draw, illustration, generate, visual, art]
bound_tools: [generate_image]
---

# Image Generation Workflow

Create images from natural language descriptions using AI image generation models (DALL-E 3 or compatible).

## When to Use

- When the user explicitly requests an image ("draw me", "generate a picture of", "create an image")
- When a creative goal involves visual output (social media graphics, concept art, illustrations)
- When a meeting or presentation needs a quick visual and the user asks for one
- Never generate images speculatively during heartbeats unless a goal explicitly calls for it

## Step-by-Step Methodology

1. **Clarify the request**: If the user's description is vague, ask a follow-up before generating. A clear prompt produces vastly better results than a guess. Key dimensions to clarify: subject, style, mood, aspect ratio, and any text that should appear.
2. **Craft the prompt**: Translate the user's intent into a detailed image generation prompt. Good prompts include:
   - The primary subject and its pose or arrangement
   - Art style or medium (photorealistic, watercolor, vector illustration, etc.)
   - Lighting and color palette
   - Composition notes (close-up, wide shot, centered, rule of thirds)
   - What to exclude (negative prompt elements phrased positively, e.g., "clean background" instead of "no clutter")
3. **Set parameters**: Choose the appropriate size and quality setting. Default to standard quality unless the user requests high quality. Respect the cost implications -- high-quality, large images are more expensive.
4. **Generate**: Call `generate_image` with the crafted prompt. The tool returns a URL or local path to the generated image.
5. **Present and iterate**: Show the result to the user. If they want adjustments, refine the prompt and regenerate. Keep the previous prompt as a starting point to maintain continuity.
6. **Store if valuable**: If the generated image is tied to a goal or project, store a memory noting what was generated and the prompt used, so it can be reproduced or iterated on later.

## Quality Guidelines

- Always respect content policies. Do not generate images of real people, violent content, or anything the user has not explicitly requested.
- Be transparent about cost. Image generation consumes API credits; inform the user if they are making many requests.
- Do not generate images during heartbeats unless an active goal specifically requires visual output. Image generation is expensive.
- When the user asks for a specific style, honor it precisely rather than defaulting to photorealistic.
- If generation fails (API error, content policy rejection), explain the failure clearly and suggest prompt modifications.
- Store the successful prompt alongside the image reference so the user can reproduce or refine the result later.
