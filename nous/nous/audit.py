"""Audit store (SQLite). requests: one row per gateway reply; events: companion safety / care / fallback events."""
import json, os, sqlite3, threading, time

SCHEMA = """
create table if not exists requests(id text primary key, ts real, client text, mode text, user text, reply text, finish text,
    c2 real, c1_pre real, c1_mean real, c3h_mean real, alarms text, detail text);
create table if not exists events(id integer primary key autoincrement, ts real, device text, kind text, score real, detail text);
create index if not exists ev_dev on events(device, ts);
"""


class Audit:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False); self.db.executescript(SCHEMA); self.lk = threading.Lock()

    def request(self, rid, client, mode, user, reply, finish, pre, done, alarms, detail=None):
        with self.lk:
            self.db.execute("insert or replace into requests values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (rid, time.time(), client, mode, user[:500], reply[:4000], finish, pre.get("C2"), pre.get("C1_pre"),
                             done.get("C1_mean"), done.get("C3H_mean"), json.dumps(alarms), json.dumps(detail or {}, ensure_ascii=False)))
            self.db.commit()

    def event(self, device, kind, score=None, **detail):
        with self.lk:
            self.db.execute("insert into events(ts,device,kind,score,detail) values(?,?,?,?,?)",
                            (time.time(), device, kind, score, json.dumps(detail, ensure_ascii=False)))
            self.db.commit()

    def recent_requests(self, n=50):
        with self.lk:
            cur = self.db.execute("select * from requests order by ts desc limit ?", (n,)); cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows: r["alarms"] = json.loads(r["alarms"]); r["detail"] = json.loads(r["detail"])
        return rows

    def recent_events(self, device=None, kind=None, n=100):
        q, a = "select id,ts,device,kind,score,detail from events where 1=1", []
        if device: q += " and device=?"; a.append(device)
        if kind: q += " and kind=?"; a.append(kind)
        with self.lk:
            rows = self.db.execute(q + " order by ts desc limit ?", (*a, n)).fetchall()
        return [dict(id=i, ts=t, device=d, kind=k, score=s, detail=json.loads(x)) for i, t, d, k, s, x in rows]

    def stats(self, since):
        with self.lk:
            r = self.db.execute("select count(*), sum(finish='content_filter') from requests where ts>=?", (since,)).fetchone()
            e = dict(self.db.execute("select kind, count(*) from events where ts>=? group by kind", (since,)).fetchall())
        return {"requests": r[0] or 0, "blocked": r[1] or 0, "events": e}
