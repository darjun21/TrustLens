# TrustLens — AI Reliability Scorecard

TrustLens checks whether an AI's answers can be trusted. You give it AI 
answers plus the source each answer came from, and it scores how well 
each answer is backed by that source — then flags and highlights the 
parts the AI made up.

Built for Week 1 of Mastering Agentic AI.

## What it does
- Scores each AI answer 0-100 for how grounded it is
- Labels each TRUSTED, REVIEW, or BLOCKED
- Highlights the made-up words in flagged answers
- Shows charts and filters

## Run it
pip install streamlit pandas plotly
streamlit run app.py
