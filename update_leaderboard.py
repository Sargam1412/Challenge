import os
import re
import subprocess

SUBMISSIONS_DIR = "submissions"
LEADERBOARD_FILE = "leaderboard/leaderboard.csv"
TRUTH_FILE = "data/ground_truth.csv"

# Ensure leaderboard exists
if not os.path.exists(LEADERBOARD_FILE):
    with open(LEADERBOARD_FILE, "w") as f:
        f.write("team,score\n")

# Load already submitted teams
submitted = set()
with open(LEADERBOARD_FILE, "r") as f:
    for line in f.readlines()[1:]:
        team = line.split(",")[0]
        submitted.add(team)

# Score each submission
for filename in os.listdir(SUBMISSIONS_DIR):
    if not filename.endswith(".csv"):
        continue

    team = filename.replace(".csv", "")

    if team in submitted:
        print(f"Skipping {team}, already scored")
        continue

    submission_path = os.path.join(SUBMISSIONS_DIR, filename)

    result = subprocess.check_output(
        ["python", "scoring_script.py", submission_path, TRUTH_FILE],
        text=True
    )

    score = float(re.search(r"SCORE=(.*)", result).group(1))

    with open(LEADERBOARD_FILE, "a") as f:
        f.write(f"{team},{score:.4f}\n")

    print(f"Added {team} with score {score:.4f}")
