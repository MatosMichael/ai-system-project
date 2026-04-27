# Final AI-system project

## Original Project Summary

The original project I am building off of was the song recommender project. The original intention of this project was to use certain metrics from a user-profile and run them through an algorithm and decide based off of that certain genres / songs that the user would potentially like. The metrics used were minutes listened to a specific genre, number of skips, number of liked songs in a specific genre, etc.

## Title & Summary

After implenetation for the revised project the user will be able to describe and ask in plain language what they want. The AI agent will parse that into specific mood categories / energy levels to more accurately recommend. This matters because it helps an issue I personally deal with sometimes when trying to find new music. It is easier to put into my own words what I am personally looking / feeling rather than "show me songs that are similar to this song" aka the radio... which is what is currently built into most music apps.

## Architecture Overview

The system follows a 5-step process.

1. User input - where they describe what they want. ex "Show me upbeat song"
2. Main.py handles the conversation and gets the Agent involved.
3. Agent deals with user input
4. Recommender ranks
5. Testing

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

   ```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Sample Interactions

An example of one input is "I'm getting ready to go to the gym." The output gives 5 song reccomendations based off of that specific mood then ranks the songs on how close they relate to your mood. Along with the scoring and reasoning.
#1. GYM HERO
Artist: Max Pulse | Genre: pop
Score: 1.00/1.00 🟢 EXCELLENT
Why: genre match: pop (from profile) (+3.0) | mood match: intense (confidence: 0.80) (+2.0) | energy fit: 0.93 vs target 0.90 (confidence: 0.80) (+2.0) | acoustic preference: produced/electric (confidence: 0.80) (+1.5) | engagement bonus: highly engaging (+1.5)

#2. SUNRISE CITY
Artist: Neon Echo | Genre: pop
Score: 0.92/1.00 🟢 EXCELLENT
Why: genre match: pop (from profile) (+3.0) | related mood fit: happy ~ intense (confidence: 0.80) (+1.2) | energy fit: 0.82 vs target 0.90 (confidence: 0.80) (+2.0) | acoustic preference: produced/electric (confidence: 0.80) (+1.5) | engagement bonus: highly engaging (+1.5).

Another example is telling it that you need some slow pace music because you are about to study.
#1. SPACEWALK THOUGHTS
Artist: Orbit Bloom | Genre: ambient
Score: 0.57/1.00 🟠 FAIR
Why: related mood fit: chill ~ focused (confidence: 0.80) (+1.2) | energy fit: 0.28 vs target 0.20 (confidence: 0.80) (+2.0) | acoustic preference: highly acoustic (confidence: 0.80) (+1.5) | engagement bonus: moderately engaging (+1.0)

#2. FOCUS FLOW
Artist: LoRoom | Genre: lofi
Score: 0.55/1.00 🟠 FAIR
Why: mood match: focused (confidence: 0.80) (+2.0) | energy fit: 0.40 vs target 0.20 (confidence: 0.80) (+1.0) | acoustic preference: highly acoustic (confidence: 0.80) (+1.5) | engagement bonus: moderately engaging (+1.0)

### Design Decisions

an issue I had come across was creating the model more flexible. at the start it was only able to take in certain simple inputs such as “I want upbeat.”… not “I’m about to workout, recommend me something.” I had two options to fix this issue. An offline approach which would be a deterministic extractor or an LLM, and I had gone with the offline option, without requiring an API.
This way the design broadend its "vocabulary" and was able to relate many more words to certain moods to better recommend songs.

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Testing Summary

With the first implementation what did not work was more "complex" inputs that the agent had no idea how to parse and give it a certain mood. Moreover the scoring seemed too harsh at the start. Giving the recommendation a poor score even though it should have been higher, so there was some adjusting there.

## Reflection

This project taught me some limitations of AI, and how to make it do a better job engaging with the task at hand for a better output.

## Reliability and Evaluation

After the AI had ran a few tests it reported: 6 out of 6 checks passed; confidence averaged 0.59 on mixed prompts, and recommendation quality improved after adding validation rules, intent presets, and similarity-based scoring.

## Reflection and Ethics

It has a dataset bias because of the small catalog of songs that it is given. Unlike modern music apps which have access to an entire database of songs. Moreover, it has some language bias as some wording may be found difficult to interpret based off of what it knows.
The AI can be missused only in some low-risk ways. Such as being able to manipulate output scores with certain responses. Using specific keywords that they know the AI "likes."
A helpful suggestion that the AI gave was using the offline approach of a determinisitc intent-and-slot extractor. This was because previously the AI was very limited to the inputs it was able to understand. And a flaw during my collaboration with AI during this project was when I had it implement the initial system design chart.

## Portfolio Artifact

### GitHub link

https://github.com/MatosMichael/ai-system-project.git

### Short reflection

This project demonstrates my ability to build and implement AI. Along with dealing with roadblocks and challenges to still achieve expected outcome.

### Loom video walkthrough link

https://www.loom.com/share/7b37d41b1442404ca6f2a65a09112577
