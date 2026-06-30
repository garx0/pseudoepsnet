import json
import sys
import os
import random

path = sys.argv[1]
head, tail = os.path.split(path)

rng = random.Random()

j = json.load(open(path, 'r'))

for x in j['flows']:
    rng.seed(x['flow_id'])
    x['hash'] = rng.randint(0, 2**64 - 1)

res = json.dumps(j, indent=4)

with open(os.path.join(head, 'flows.json'), 'w') as f:
    f.write(res)