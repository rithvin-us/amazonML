import json, sys
path = sys.argv[1]; out = sys.argv[2]; maxlen = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
with open(path, encoding='utf-8') as f, open(out, 'w', encoding='utf-8') as o:
    for line in f:
        try: d = json.loads(line)
        except: continue
        t = d.get('type')
        if t not in ('user', 'assistant'): continue
        msg = d.get('message', {})
        c = msg.get('content')
        ts = d.get('timestamp', '')[:19]
        if isinstance(c, str):
            o.write(f"\n[{ts}] {t.upper()}: {c[:maxlen]}\n"); continue
        for part in c or []:
            pt = part.get('type')
            if pt == 'text':
                o.write(f"\n[{ts}] {t.upper()}: {part['text'][:maxlen]}\n")
            elif pt == 'tool_use':
                inp = json.dumps(part.get('input', {}))[:300]
                o.write(f"[{ts}]   TOOL {part.get('name')}: {inp}\n")
            elif pt == 'tool_result':
                cc = part.get('content')
                if isinstance(cc, list):
                    cc = ' '.join(x.get('text', '') for x in cc if isinstance(x, dict))
                o.write(f"[{ts}]   RESULT: {str(cc)[:200]}\n")
