# Fantasy Baseball Advisor

A command-line tool for fantasy baseball analysis using real MLB stats. Pulls player data via `pybaseball`, analyzes performance trends, and surfaces actionable recommendations for your fantasy roster.

## Setup

**Requirements:** Python 3.11+

1. Clone the repo and enter the project directory:

```bash
git clone <repo-url>
cd fantasy-baseball-advisor
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the example env file and add your config:

```bash
cp config/.env.example .env
```

## Usage

```bash
python -m fantasy_baseball_advisor --help
```

## Project Structure

```
fantasy-baseball-advisor/
├── src/
│   └── fantasy_baseball_advisor/   # Main package
├── tests/                          # Unit tests
├── data/
│   └── cache/                      # Cached stat downloads (gitignored)
├── config/                         # Config files and .env.example
├── requirements.txt
└── README.md
```
