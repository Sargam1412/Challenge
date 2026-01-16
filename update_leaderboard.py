import re
import subprocess

# Run scoring on latest submission
result = subprocess.check_output(
    ["python", "scoring_script.py", "submissions/latest.csv"],
    text=True
)

score = float(re.search(r"SCORE=(.*)", result).group(1))

# Append new score to markdown leaderboard
with open("leaderboard.md", "a") as f:
    f.write(f"\n| latest_submission | {score:.4f} |")
