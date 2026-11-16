import json
from pathlib import Path


source = Path(r"figure/U_Challenge_2_trajectories.jsonl")
output = Path(r"figure/U_Challenge_2_trajectories_recovered.jsonl")

valid_count = 0
invalid_count = 0
first_invalid = None
invalid_examples = []

with open(source, "rb") as fin, \
     open(output, "wb") as fout:

    for line_number, raw_line in enumerate(fin, 1):

        if not raw_line.strip():
            continue

        try:
            text = raw_line.decode("utf-8-sig")
            record = json.loads(text)

            if not isinstance(record, dict):
                raise ValueError("Record is not a JSON object")

            # Preserve the original valid JSONL record.
            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False
                ).encode("utf-8") + b"\n"
            )

            valid_count += 1

        except (UnicodeDecodeError, json.JSONDecodeError,
                ValueError) as e:

            invalid_count += 1

            if first_invalid is None:
                first_invalid = line_number

            if len(invalid_examples) < 10:
                invalid_examples.append(
                    (line_number, str(e))
                )

            continue

print("\n========== RECOVERY REPORT ==========")
print("Valid records:", valid_count)
print("Invalid records:", invalid_count)
print("First invalid line:", first_invalid)
print("Recovered archive:", output)
print("Output size (MB):", round(
    output.stat().st_size / 1024**2, 2
))

for line, error in invalid_examples:
    print(f"Invalid line {line}: {error}")
