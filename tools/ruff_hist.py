import collections
import json
import sys
jf = sys.argv[1]
items = json.load(open(jf))
ctr = collections.Counter((v.get('code', 'UNKNOWN') for v in items))
print('rule\tcount')
for rule, cnt in ctr.most_common(50):
    print(f'{rule}\t{cnt}')