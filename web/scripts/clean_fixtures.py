"""Write Python clean_text outputs for tricky inputs so the JS port can be checked against them."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.protocol import clean_text

cases = [
    "Service dekat kedai ni BAGUS!!! #syok @kawan_baik https://t.co/abc",
    "Straße ΟΔΟΣ ﬁne Ｆｕｌｌｗｉｄｔｈ ½ ™ İstanbul",
    "  tab\there\x1cfile\x85next line　ideo ﻿bom ",
    "www.example.com/path?x=1 then @user.name and @ali123 ok",
    "emoji 😍🔥 #Malaysia2026 ##double # alone",
    "http:/broken and https://ok.my/x,y",
    "", "   ", "@only @mentions", "Tak sangka lah weh 🤣🤣 gila best",
]
rng = random.Random(7)
alphabet = list("abcAB #@_:/.wht ps\t\n ßΣςİﬀ①Ⅻ😀é́　\x85") + ["http://", "www.", "@x"]
for _ in range(400):
    cases.append("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40))))
out = [{"input": c, "expected": clean_text(c)} for c in cases]
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=0))
print(len(out))
