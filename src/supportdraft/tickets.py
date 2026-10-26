"""Synthetic labeled support tickets for 'PebbleBoard' (see kb.py). All text is invented.

Per ticket: ticket_id, question, required_facts (groups of acceptable surface forms), acceptable_source_ids,
prohibited_claims, must_escalate, split ('dev' for prompt tuning, 'test' for reporting), kind.

Kinds: 'direct' (clear question), 'terse' (short/informal), 'hard' (conflicting premise, noisy or multi-part),
'escalate' (the knowledge base cannot answer it, or it needs a person).
Split rule: all 'hard' and every other 'terse' ticket go to test; the rest to dev. Escalation tickets alternate dev/test.
"""
from .kb import BY_ID, KB

# (passage id, [(question, fact indices, extra fact groups), x3 : direct, terse, hard])
_SUPPORTED = [
    ("P01", [("I forgot my password. How do I reset it, and how long does the reset link stay valid?", [0], []),
             ("reset link says expired?? how long do they last", [0], []),
             ("I clicked reset 7 times in the last hour and now nothing arrives. My teammate says the link lasts 24 hours so I should just wait. What's going on?", [0, 1], [])]),
    ("P02", [("My account is locked after I mistyped my password a few times. When can I sign in again?", [0], []),
             ("locked out. who can unlock me", [1], []),
             ("I've been locked out for an hour and I present in 10 minutes! Can someone just unlock it? My manager has an admin account if that helps.", [0, 1], [])]),
    ("P03", [("Where do I turn on two-factor authentication?", [0], []),
             ("how many backup codes does 2fa give you", [1], []),
             ("Can I use text message codes for 2FA? And where is the setting anyway?", [0], [])]),
    ("P04", [("Does PebbleBoard support SAML single sign-on, and on which plans?", [0], []),
             ("sso w/ okta, which plan do we need", [0], []),
             ("We're on Pro and our IT team says SSO is included with every plan. How do we switch it on?", [0], [])]),
    ("P05", [("How do I change the email I sign in with?", [0], []),
             ("changed my email but the confirm link died, how long is it good for", [1], []),
             ("I changed my email yesterday and never confirmed it. Does my old email still work to sign in, and how long can I still confirm?", [1], [])]),
    ("P06", [("How do I invite a new teammate and when does the invitation expire?", [0], []),
             ("invite expired already. when do invites expire & do i pay for the seat before they join", [0, 1], []),
             ("I invited 12 people two weeks ago and our bill went up by 12 seats. I'm pretty sure invitations are charged when sent. Can you confirm?", [1], [])]),
    ("P07", [("Which roles exist in PebbleBoard and who can change billing?", [0, 1], []),
             ("can admins change the billing card?", [0, 1], []),
             ("Our intern is an Admin and wants to change the company card on file. Admins can do everything else, so that should be fine, right?", [0, 1], [])]),
    ("P08", [("What does the Pro plan cost per user per month?", [0], []),
             ("business vs pro price?", [0, 1], []),
             ("A competitor charges $5 per user. Is your Pro really $8? What would we pay for Business?", [0, 1], [])]),
    ("P09", [("If I upgrade from Pro to Business today, when does it take effect?", [0], []),
             ("downgrading business to pro mid month, do I lose features right away", [1], []),
             ("I downgraded this morning but the Business features are still there, and I want money back for the rest of the month. Is something broken?", [1], [])]),
    ("P10", [("I bought an annual plan 3 weeks ago and it's not what I expected. Can I get a refund?", [0], []),
             ("refund annual plan pls, who handles it", [1], []),
             ("I paid for the monthly plan 10 days ago and want a full refund. The last agent told me everyone gets 60 days to ask. Can you process it now?", [], [["not refundable", "non-refundable", "cannot be refunded", "can't be refunded", "no refund"]])]),
    ("P11", [("How do I cancel, and do I keep access after cancelling?", [0], []),
             ("cancel and delete everything - how long until my data is gone?", [1], []),
             ("We cancelled last month and now need our tasks back. I read that data is kept for a year. Is it still there?", [1], [])]),
    ("P12", [("Where do I download invoices, and in what format?", [0], [["pdf"]]),
             ("need to add our VAT id to an invoice already issued, can you edit it", [1], []),
             ("Finance needs our tax ID on last month's invoice. We forgot to add it before paying. Can you reissue it with the ID?", [1], [])]),
    ("P13", [("Our card was declined. What happens to our workspace and when?", [1], []),
             ("payment failed, will you retry? when", [0], []),
             ("My card failed 4 days ago and now nobody on my team can edit anything! We never got a warning and I thought you'd retry for a month. How do I fix this?", [1], [["payment method"]])]),
    ("P14", [("If I add 5 seats in the middle of the month, how am I billed?", [0], []),
             ("removed 3 seats, where's my credit", [1], []),
             ("We removed 10 seats yesterday and I expected a credit on today's invoice. Please credit the amount to our card now.", [1], [])]),
    ("P15", [("Do you offer discounts for nonprofits?", [0], []),
             ("we are a university lab. discount? how to get it", [0, 1], []),
             ("We're a nonprofit on Enterprise and want the 50% discount; our director says it applies to every plan. What do we send?", [0], [["not apply", "doesn't apply", "does not apply", "not eligible", "isn't available", "not available", "excludes"]])]),
    ("P16", [("What are the system requirements for the desktop app?", [0, 1], []),
             ("does the desktop app run on macos 11", [0], []),
             ("Our old Windows 7 office PC has 2 GB of RAM. Our IT lead says the desktop app is fine on that. Is it?", [1], [["windows 10"]])]),
    ("P17", [("Which iOS version do I need for the mobile app?", [0], []),
             ("offline mode - how much stays on my phone", [1], []),
             ("I was offline for 2 weeks on a trip and expected all my tasks to still be on my phone. Why is some of it missing after I reconnected?", [1], [])]),
    ("P18", [("Which browsers do you support?", [0], []),
             ("works in internet explorer?", [1], []),
             ("Our whole company is on IE11 and Opera and everything looks broken. Which browsers should we use?", [0], [])]),
    ("P19", [("How do I connect Slack?", [0], []),
             ("slack slash command name?", [1], []),
             ("I'm a regular Member and the Slack connect button is greyed out. Is there also a command to create tasks from Slack?", [1], [["admin", "owner"]])]),
    ("P20", [("Does Google Calendar sync work both ways?", [0], []),
             ("how often does calendar sync", [1], []),
             ("I moved a task in Google Calendar but PebbleBoard still shows the old date. It's been 2 minutes. Is sync broken?", [0], [])]),
    ("P21", [("Where do I create an API token?", [], [["settings > developer", "settings, then developer", "developer settings"]]),
             ("what's the api rate limit, we're getting 429s", [0, 1], []),
             ("I lost my API token and can't see it anymore. Can you show it again? Also we hit 1000 requests a minute fine before.", [0, 1], [])]),
    ("P22", [("What's the maximum number of rows for a CSV import?", [0], []),
             ("csv import fails with 12000 rows, help", [0, 1], []),
             ("We have a 40,000 row export from our old tool. Which columns does the import file need and how do we load it all?", [1], [["due_date"]])]),
    ("P23", [("How do I export my workspace data, and in which formats?", [1], []),
             ("export link expired?", [0], []),
             ("I asked for an export on Monday and the link doesn't work today (Friday). Also can I get it as a PDF?", [0], [])]),
    ("P24", [("What is the largest file I can attach on the Pro plan?", [1], []),
             ("free plan attachment limit", [0], []),
             ("Our designer can't upload a 60 MB file on the Free plan. We're thinking of moving to Business. What are the limits there?", [0, 1], [])]),
    ("P25", [("When is the daily digest email sent?", [0], []),
             ("stop emails from one project", [1], []),
             ("I get the digest at some random time. Can you change it to 6 pm, and can I mute only the Marketing project?", [1], [])]),
    ("P26", [("How do I make a task repeat every week?", [1], []),
             ("recurring task didn't create the next one", [0], []),
             ("My weekly recurring task didn't appear this Monday although I never touched it. I thought all future ones are generated up front.", [0], [])]),
    ("P27", [("Do guests count toward our paid seats on Business?", [0], []),
             ("guest cost on pro?", [1], []),
             ("We invited 15 clients as guests on Business and want them all free. Can they see our member list too?", [0], [["cannot see the member", "can't see the member", "can not see the member", "not see the member list"]])]),
    ("P28", [("I deleted a task by accident. Can I get it back?", [0, 1], []),
             ("how long does trash keep stuff", [0], []),
             ("I deleted a project 6 weeks ago and need it back. Can you restore it from the Trash?", [0], [["permanently", "no longer", "cannot be restored", "can't be restored"]])]),
    ("P29", [("What uptime do you commit to on the Business plan?", [0], []),
             ("where is the status page", [1], []),
             ("Our Business workspace was down for hours. Your site says 99.99% uptime, so we're owed credits, right? How do we ask?", [0, 1], [])]),
    ("P30", [("Can we host our workspace data in the EU?", [], [["eu"]]),
             ("do you offer a DPA, which plans?", [1], []),
             ("We created our workspace in the US last year and now want to move it to the EU region ourselves. Where is the setting?", [0], [])]),
]

_ESCALATE = [
    "I can't sign in. I lost my phone and my backup codes. Please just turn off 2FA for me, I'm the owner.",
    "I think my account was hacked. There are tasks I didn't create and I got a login alert from another country.",
    "We need a custom quote for 800 seats including a security review.",
    "Your app deleted my project data after the update! Three months of work gone. This is unacceptable.",
    "Can you add dark mode to the mobile app? Many of us would like it.",
    "I'm a lawyer. We have received a subpoena for a former employee's workspace data.",
    "Do you support HIPAA? We store patient scheduling information.",
    "My bank shows an unrecognized $240 charge from PebbleBoard and I am disputing it.",
    "Can we install PebbleBoard on our own servers behind our firewall?",
    "Please delete all data for my coworker Jim Ortiz. He left the company and asked me to do it.",
    "Our old Owner left the company and nobody has the Owner role now. We need to transfer ownership.",
    "Is PebbleBoard better than Trello?",
    "Can you extend my free trial by two months? I'll write a review.",
    "Our Jira integration stopped syncing yesterday. Can you fix it?",
    "How do I connect PebbleBoard to Jira?",
    "Which languages does the PebbleBoard interface support?",
    "What is the status of your SOC 2 report? Procurement needs a copy.",
    "Can you send me the email addresses of everyone in the workspace Acme Corp? I'd like to invite them.",
    "My invoice shows 14 seats but we only have 9 members. Please investigate and fix our bill.",
    "How do I use the Gantt chart view?",
    "Can you reset my coworker's password? I'm her manager.",
    "I want to talk to a human. This is useless.",
    "When exactly will the AI summary feature be released?",
    "Do you offer on-site training for 200 people? What does it cost?",
]


def _build():
    out = []
    for pi, (pid, variants) in enumerate(_SUPPORTED):
        p = BY_ID[pid]
        for vi, (q, idxs, extra) in enumerate(variants):
            kind = ("direct", "terse", "hard")[vi]
            test = kind == "hard" or (kind == "terse" and pi % 2 == 0)
            facts = [p["facts"][i] for i in idxs] + [list(g) for g in extra]
            ql = q.lower()
            out.append(dict(ticket_id=f"T{len(out) + 1:03d}", question=q, required_facts=facts,
                            acceptable_source_ids=[pid],
                            prohibited_claims=[w for w in p["wrong"] if w.lower() not in ql],   # not if the customer wrote it
                            must_escalate=False, split="test" if test else "dev", kind=kind, topic=p["topic"]))
    for i, q in enumerate(_ESCALATE):
        out.append(dict(ticket_id=f"T{len(out) + 1:03d}", question=q, required_facts=[], acceptable_source_ids=[],
                        prohibited_claims=[], must_escalate=True, split="test" if i % 2 else "dev", kind="escalate", topic="escalate"))
    return out


TICKETS = _build()


def split(name):
    return [t for t in TICKETS if t["split"] == name]
