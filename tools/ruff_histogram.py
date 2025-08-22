import collections
import re
import sys

counts = collections.Counter()
for line in sys.stdin:
    m = re.search(r"\s([A-Z]{1,3}\d{3})\s", line)
    if m: counts[m.group(1)] += 1
for code, n in counts.most_common(50):
    print(f"{code}\t{n}")
