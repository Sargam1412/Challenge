import pandas as pd

# 1️⃣ Load the authoritative CSV
csv_path = "leaderboard/leaderboard.csv"
md_path = "leaderboard/leaderboard.md"

df = pd.read_csv(csv_path)

# Optional: sort by score descending
df = df.sort_values(by="score", ascending=False)

# 2️⃣ Create Markdown table
md_table = ["| Team | Score |", "|------|-------|"]
for _, row in df.iterrows():
    md_table.append(f"| {row['team']} | {row['score']:.4f} |")

# 3️⃣ Write to leaderboard.md
with open(md_path, "w") as f:
    f.write("\n".join(md_table))

print(f"Rendered {len(df)} entries to {md_path}")
