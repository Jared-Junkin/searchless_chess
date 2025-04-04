import json
from pprint import pprint
import os
filename = "tokenizer.json"
filepath = os.path.join("/workspace/searchless_chess/src/pythia/pythia-160m-deduped/step143000/models--EleutherAI--pythia-160m-deduped/snapshots/3ec8114c365e2a8aee635fa2f5e5fed3dd8f7eec", filename)
with open(filepath) as f:
    d = json.load(f)
    f.close()

    for key in d['model'].keys():
        print(key)