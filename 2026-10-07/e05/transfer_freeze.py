"""E05: combine the source (4B) two-stage screen and the target (8B) native screen into TRANSFER_FREEZE.json.
python transfer_freeze.py SRC_SCREEN_FREEZE.json TGT_SCREEN_FREEZE.json TGT_MAP_DIR OUT.json"""
import hashlib, json, sys
src, tgt, mapd, out = sys.argv[1:5]
S, T = json.load(open(src)), json.load(open(tgt))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
fz = {"source_model": S["model"], "target_model": T["model"], "twostage": S["twostage"], "native": T["twostage"],
      "overlap": sorted(set(S["twostage"]) & set(T["twostage"])), "stop": len(S["twostage"]) < 4,
      "map_freeze_sha256": sha(f"{mapd}/MAP_FREEZE.json"), "source_screen_sha256": sha(src), "target_screen_sha256": sha(tgt),
      "maponly": [], "rscreen": []}
json.dump(fz, open(out, "w"), ensure_ascii=False, indent=1)
print("transfer:", fz["twostage"]); print("native:", fz["native"]); print("overlap:", fz["overlap"], "stop:", fz["stop"])
