"""Synthetic metadata check/replay example; not a biological outcome."""
import json
from invariantcell.io import digest
from invariantcell.scientific_tools_v10 import ToolCall,candidate,verify
records={k:{"record_id":k,"kind":"synthetic","source_sha256":digest({"fixture":k}),"temperature_C":v}for k,v in (("fixture_a",10),("fixture_b",25))}
call=ToolCall("compare_field",("fixture_a","fixture_b"),"temperature_C")
proof=candidate(call,records)
result=verify(proof,call,records,{k:digest(v)for k,v in records.items()})
assert proof["outcome"]=="MISMATCH" and result["valid"]
print(json.dumps({"scope":"SYNTHETIC_ENGINEERING_ONLY","outcome":proof["outcome"],"independent_replay":result},indent=2))
