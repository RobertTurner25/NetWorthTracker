# Agent Instructions

You are an AI coding assistant working inside this repository.
Your goal is to produce clean, correct, production-quality Python code
with a fully functional Streamlit user interface.

This project is a personal finance / investment tracking application.

---

## Project Context

- Language: Python 
- UI Framework: Streamlit
- Data Storage: CSV files (local, persisted)
- Hosting Target: Streamlit Community Cloud (free tier)
- Primary User: single user (no auth required)
- Access Pattern: desktop + mobile browser

The application must:
- run without a separate backend server
- be easy to deploy on free cloud hosting
- work reliably on mobile devices

---

## Core Requirements (Non-Negotiable)

- The app MUST be runnable with:
  `streamlit run <entry_file>.py`
- No paid services
- No API keys required for core functionality
- No background workers or daemons
- All UI must be functional, not mocked
- All code must be copy-paste runnable

---

## Data Handling Rules

- Data is stored in CSV files
- CSV updates must be safe and idempotent
- Never silently drop user data
- Explicitly handle missing or empty files
- Prefer simple schemas over clever abstractions

Allowed:
- pandas
- standard library

Avoid:
- databases
- ORMs
- async IO
- background schedulers

---

## UI / UX Expectations

- Use Streamlit components only
- UI must be intuitive without instructions
- Editing data must be possible via the UI
- Forms should validate inputs
- Buttons must have clear effects
- Show user feedback (success / error messages)

Mobile considerations:
- Avoid tiny tables when possible
- Prefer forms and grouped inputs
- Use wide layout responsibly

---

## Charts & Visualization

- Use Streamlit-native charts or matplotlib
- Charts must reflect real computed data
- Label axes and metrics clearly
- Prefer clarity over visual flair

---

## Code Quality Standards

- Favor readability over cleverness
- Small, named helper functions
- No global mutable state except configuration constants
- Explicit variable names (no single-letter variables)
- Handle edge cases defensively

Avoid:
- premature optimization
- magic numbers
- deeply nested logic
- silent exception swallowing

---

## Autonomy Rules

You MAY:
- refactor code for clarity
- reorganize files for maintainability
- add helper functions
- add UI improvements that preserve intent

You MUST NOT:
- remove features without asking
- change data formats without explicit approval
- introduce new dependencies casually
- invent business logic assumptions

---

## How to Respond to Tasks

When given a task:
1. Clarify assumptions explicitly if needed
2. Produce complete, runnable code
3. Explain important decisions briefly
4. Do not include filler or generic explanations

If something is ambiguous, ask ONE clear question before proceeding.

---

## Success Criteria

A solution is considered successful if:
- The app runs without errors
- The UI works end-to-end
- Data can be viewed and edited
- The app can be deployed without modification
