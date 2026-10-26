"""Evidence-first prompt, output parser, and verbatim-quote verification."""
from __future__ import annotations

import re

SYSTEM = (
    "You answer technical-support questions using ONLY the numbered excerpts provided. Never use outside knowledge.\n"
    "Always reply in exactly this format, starting with the line DECISION:\n"
    "DECISION: ANSWER or INSUFFICIENT_EVIDENCE\n"
    "EVIDENCE: copy one or two sentences from the excerpts that support your answer, word for word, in double quotes, with the excerpt id "
    "in square brackets AFTER each quote. Write EVIDENCE: none only if the decision is INSUFFICIENT_EVIDENCE.\n"
    "ANSWER: a short reply (under 120 words) that uses only that evidence. Write ANSWER: none if the decision is INSUFFICIENT_EVIDENCE.\n"
    "If you cannot quote a sentence that answers the question, the decision must be INSUFFICIENT_EVIDENCE.\n\n"
    "Example of a reply when an excerpt contains the answer:\n"
    "DECISION: ANSWER\n"
    "EVIDENCE: \"Set the cache size to 512 and restart the daemon.\" [C00007]\n"
    "ANSWER: Set the cache size to 512 and restart the daemon.\n\n"
    "Example of a reply when the excerpts do not contain the answer:\n"
    "DECISION: INSUFFICIENT_EVIDENCE\n"
    "EVIDENCE: none\n"
    "ANSWER: none")


def build_messages(question, chunks):
    block = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks) or "(no excerpts found)"
    user = f"Excerpts:\n{block}\n\nQuestion:\n{question.strip()}\n\nReply:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


BULLET = chr(0x2022)
CURLY = chr(0x201C) + chr(0x201D) + chr(0x2018) + chr(0x2019)
_FOLD = {0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"'}
_DEC = re.compile(r"DECISION\s*:\s*(ANSWER|INSUFFICIENT[_ ]EVIDENCE)", re.I)
_EVI = re.compile(r"EVIDENCE\s*:\s*(.*?)(?=\n\s*ANSWER\s*:|\Z)", re.I | re.S)
_ANS = re.compile(r"ANSWER\s*:\s*(.*)\Z", re.I | re.S)
_BARE_ABSTAIN = re.compile(r"^\s*[\"'`*]*INSUFFICIENT[_ ]EVIDENCE\b", re.I)
_QUOTE = re.compile(r"(.+?)\s*\[(C\d{5})\]", re.S)
_QUOTED = re.compile(r'"([^"]{15,})"')
_QUOTE_BEFORE = re.compile(r"\[(C\d{5})\]\s*(.+?)(?=\s*\[C\d{5}\]|\Z)", re.S)


def parse(text):
    """-> dict(decision in ANSWER/INSUFFICIENT/UNPARSED, quotes [(text, chunk_id)], answer)."""
    m = _DEC.search(text or "")
    if m:
        dec = "ANSWER" if m.group(1).upper() == "ANSWER" else "INSUFFICIENT"
    else:
        # tolerated fallbacks (small models drop the label): infer the decision from the ANSWER field or a bare abstention
        t = text or ""
        am0 = _ANS.search(t)
        if _BARE_ABSTAIN.match(t) or (am0 and re.match(r"[\s\"'`*]*(none\b|insufficient[_ ]evidence)", am0.group(1), re.I)):
            return dict(decision="INSUFFICIENT", quotes=[], answer="")
        if am0 and am0.group(1).strip():
            dec = "ANSWER"
        else:
            return dict(decision="UNPARSED", quotes=[], answer="")
    ev = _EVI.search(text)
    quotes = []
    if ev and ev.group(1).strip().lower().strip(" .\n") != "none":
        evt = ev.group(1).strip()
        pairs = [(q, cid) for cid, q in _QUOTE_BEFORE.findall(evt)] if re.match(r"[\s\-*" + BULLET + r"]*\[C\d{5}\]", evt) else _QUOTE.findall(evt)
        if not pairs:                      # quotes without any excerpt id: keep them, verify against every shown excerpt
            pairs = [(q, None) for q in _QUOTED.findall(evt)]
        for q, cid in pairs:
            q = q.strip().lstrip("-* " + BULLET).strip().strip("\"' " + CURLY).strip()
            if q:
                quotes.append((q, cid))
    am = _ANS.search(text)
    ans = am.group(1).strip() if am else ""
    if ans.lower().strip(" .") == "none":
        ans = ""
    return dict(decision=dec, quotes=quotes, answer=ans)


def _norm(s):
    return re.sub(r"\s+", " ", s.lower().translate(_FOLD)).strip()


def verify(parsed, shown_chunks):
    """A quote is grounded if it appears verbatim (whitespace/case-insensitive) in the excerpt it cites AND that excerpt was shown;
    a quote given without an id is grounded if it appears verbatim in any excerpt that was shown.
    Returns (n_quotes, n_grounded). Verbatim presence is necessary for grounding, not proof that the quote answers the question."""
    by = {c["id"]: _norm(c["text"]) for c in shown_chunks}
    ok = sum(1 for q, cid in parsed["quotes"]
             if _norm(q) and ((cid in by and _norm(q) in by[cid]) if cid else any(_norm(q) in t for t in by.values())))
    return len(parsed["quotes"]), ok
