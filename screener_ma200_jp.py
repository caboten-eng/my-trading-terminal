name: Japan MA200 Recovery Screener

on:
  workflow_dispatch:
  schedule:
    - cron: '30 10 * * 1-5'  # 19:30 JST

permissions:
  contents: write

concurrency:
  group: jp-ma200-recovery
  cancel-in-progress: false

jobs:
  screening:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install yfinance pandas numpy openpyxl lxml html5lib

      - name: Run MA200 screener
        run: python screener_ma200_jp.py

      - name: Save results with retry
        shell: bash
        run: |
          set -e
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add results/ma200_jp.json results/ma200_jp.csv
          if git diff --cached --quiet; then
            echo "No change in MA200 results"
            exit 0
          fi
          git commit -m "Update Japan MA200 screening results"
          for attempt in 1 2 3 4 5; do
            echo "Push attempt ${attempt}/5"
            git fetch origin main
            git rebase origin/main
            if git push origin HEAD:main; then
              echo "Successfully published MA200 results"
              exit 0
            fi
            sleep 5
          done
          echo "Push failed after five attempts"
          exit 1
