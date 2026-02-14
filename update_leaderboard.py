import re
import subprocess

# Run scoring on latest submission
result = subprocess.check_output(
    ["python", "scoring_script.py", "submissions/submission.csv", "data/ground_truth.csv"],
    text=True
)

score = float(re.search(r"SCORE=(.*)", result).group(1))

# Append new score to markdown leaderboard
with open("leaderboard.md", "a") as f:
    f.write(f"\n| latest_submissionJ | {score:.4f} |")

# Also print to stdout so it appears in GitHub Actions log
with open("leaderboard.md") as f:
    print(f.read())