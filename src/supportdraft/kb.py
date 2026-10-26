"""Knowledge base for the fictional product 'PebbleBoard' (a team task-tracking app).

Everything here is invented for this project: no real company, product, price or policy. Each passage carries
`facts`: groups of acceptable surface forms (lower case); a draft "states" a fact if any form of the group appears in it.
`wrong`: plausible wrong values that must not appear in a correct draft for that passage.
"""

KB = [
    dict(id="P01", topic="account", title="Password reset",
         text="To reset a password, choose 'Forgot password' on the sign-in page and enter the account email. The reset link is "
              "valid for 30 minutes. Check the spam folder if it does not arrive. A user can request at most 5 reset emails per hour.",
         facts=[["30 minutes", "30-minute", "30 min"], ["5 reset", "five reset", "5 password reset", "5 emails", "5 requests", "5 per hour", "5 times"]],
         wrong=["24 hours", "60 minutes", "1 hour link"]),
    dict(id="P02", topic="account", title="Account lockout",
         text="After 5 failed sign-in attempts an account is locked for 15 minutes. A workspace Admin can unlock it immediately "
              "from Admin > Members > Unlock.",
         facts=[["15 minutes", "15-minute", "15 min"], ["admin"]],
         wrong=["30 minutes locked", "one hour", "24 hours"]),
    dict(id="P03", topic="account", title="Two-factor authentication",
         text="Turn on two-factor authentication under Settings > Security. PebbleBoard supports authenticator apps and issues 10 "
              "backup codes at setup. If both the device and the backup codes are lost, support must verify the owner's identity "
              "before access is restored.",
         facts=[["settings > security", "settings, then security", "security settings"], ["10 backup", "ten backup", "backup codes"]],
         wrong=["sms code", "text message code"]),
    dict(id="P04", topic="account", title="Single sign-on (SSO)",
         text="SAML single sign-on is available on the Business and Enterprise plans only. An Owner configures it under Admin > "
              "Security > SSO. Okta and Azure AD are supported.",
         facts=[["business and enterprise", "business or enterprise", "business plan", "enterprise plan"], ["admin > security", "admin, then security", "okta", "azure"]],
         wrong=["available on all plans", "pro plan includes sso", "google workspace sso"]),
    dict(id="P05", topic="account", title="Changing the account email",
         text="Change the sign-in email under Settings > Profile. A confirmation message is sent to the new address and the link "
              "is valid for 24 hours. The old address keeps working until the new one is confirmed.",
         facts=[["settings > profile", "settings, then profile", "profile settings"], ["24 hours", "24-hour", "24 hrs"]],
         wrong=["30 minutes", "7 days", "48 hours"]),
    dict(id="P06", topic="account", title="Inviting teammates",
         text="Admins invite teammates under Admin > Members > Invite. An invitation expires after 7 days. A seat is counted only "
              "once the invitation is accepted.",
         facts=[["7 days", "seven days", "7-day", "a week"], ["accepted", "accepts", "acceptance"]],
         wrong=["14 days", "30 days", "seat is charged when the invite is sent"]),
    dict(id="P07", topic="account", title="Roles and permissions",
         text="PebbleBoard has four roles: Owner, Admin, Member and Guest. Only an Owner can change billing details or delete the "
              "workspace. Admins manage members and integrations. Guests can only see projects they were invited to.",
         facts=[["owner"], ["billing"]],
         wrong=["admins can change billing", "any admin can delete the workspace"]),
    dict(id="P08", topic="billing", title="Plans and prices",
         text="Free: up to 3 projects and 5 users. Pro: $8 per user per month, or $80 per user per year. Business: $15 per user per "
              "month, or $150 per user per year. Enterprise: custom pricing through the sales team.",
         facts=[["$8"], ["$15"]],
         wrong=["$10 per user", "$12 per user", "$20 per user"]),
    dict(id="P09", topic="billing", title="Upgrading and downgrading",
         text="Plans can be changed at any time under Billing > Plan. An upgrade takes effect immediately and the difference is "
              "prorated. A downgrade takes effect at the end of the current billing period.",
         facts=[["immediately", "right away", "immediate"], ["end of the current billing", "end of your billing", "end of the billing", "end of the current period", "end of the period", "end of your current"]],
         wrong=["downgrades are immediate", "refund the difference"]),
    dict(id="P10", topic="billing", title="Refund policy",
         text="Annual plans can be refunded in full within 30 days of purchase or renewal. Monthly plans are not refundable but can "
              "be cancelled at any time. Refunds are approved by the Billing team; support agents cannot issue them. Requests go to "
              "billing@pebbleboard.example.",
         facts=[["30 days", "30-day", "thirty days"], ["billing team", "billing@pebbleboard.example", "billing@"]],
         wrong=["60 days", "14 days", "monthly plans can be refunded", "90 days refund"]),
    dict(id="P11", topic="billing", title="Cancelling a plan",
         text="Cancel under Billing > Cancel plan. Access continues until the end of the paid period. Workspace data is kept for 90 "
              "days after cancellation and then permanently deleted.",
         facts=[["end of the paid period", "end of your paid", "end of the billing", "end of the current", "end of the period", "end of your current"], ["90 days", "90-day", "ninety days"]],
         wrong=["30 days after cancellation", "deleted immediately", "kept for 1 year"]),
    dict(id="P12", topic="billing", title="Invoices and VAT",
         text="Invoices are available as PDF under Billing > Invoices and are issued on the payment date. Add a VAT or tax ID under "
              "Billing > Details before the invoice is generated; issued invoices cannot be edited.",
         facts=[["billing > invoices", "billing, then invoices", "invoices page", "invoices section"], ["cannot be edited", "can't be edited", "cannot be changed", "can not be edited", "not be edited", "before the invoice"]],
         wrong=["we can edit issued invoices", "invoices are emailed on the 1st"]),
    dict(id="P13", topic="billing", title="Failed payments",
         text="If a payment fails, PebbleBoard retries on day 3, day 5 and day 7. If it still fails after day 7 the workspace becomes "
              "read-only. Update the card under Billing > Payment method to restore editing.",
         facts=[["day 3", "3 days", "third day"], ["read-only", "read only"]],
         wrong=["deleted after 7 days", "suspended immediately", "retries daily"]),
    dict(id="P14", topic="billing", title="Adding and removing seats",
         text="Seats added in the middle of a billing cycle are prorated. Removing seats takes effect at the next renewal, with no "
              "credit for the current period.",
         facts=[["prorated", "pro-rated", "pro rata", "proportion"], ["next renewal", "renewal"]],
         wrong=["credited immediately", "full refund for removed seats"]),
    dict(id="P15", topic="billing", title="Nonprofit and education discount",
         text="Verified nonprofits and educational institutions receive 50% off Pro and Business. Send proof of status to "
              "billing@pebbleboard.example. The discount does not apply to Enterprise.",
         facts=[["50%", "half", "50 percent"], ["proof", "verif"]],
         wrong=["25% off", "free for nonprofits", "70% off"]),
    dict(id="P16", topic="install", title="Desktop app requirements",
         text="The desktop app runs on Windows 10 or later, macOS 12 or later, and Ubuntu 20.04 or later. It needs at least 4 GB of RAM. "
              "Download it from pebbleboard.example/download.",
         facts=[["macos 12", "macos 12 or"], ["4 gb"]],
         wrong=["macos 10", "windows 7", "2 gb of ram", "8 gb"]),
    dict(id="P17", topic="install", title="Mobile apps and offline mode",
         text="The iOS app requires iOS 15 or later and the Android app requires Android 9 or later. Offline mode keeps the last 7 days "
              "of tasks on the device and syncs when the connection returns.",
         facts=[["ios 15"], ["7 days", "seven days", "a week"]],
         wrong=["ios 13", "android 7", "30 days of tasks"]),
    dict(id="P18", topic="install", title="Browser support",
         text="PebbleBoard supports the latest two versions of Chrome, Firefox, Safari and Edge. Internet Explorer is not supported.",
         facts=[["latest two", "last two", "two latest", "two most recent", "2 latest", "latest 2"], ["internet explorer"]],
         wrong=["internet explorer is supported", "ie 11 is supported", "opera"]),
    dict(id="P19", topic="install", title="Slack integration",
         text="Admins connect Slack under Settings > Integrations > Slack. Task updates are posted to the channel chosen at setup, and "
              "the /pebble slash command creates tasks from Slack. Only Admins and Owners can connect the integration.",
         facts=[["settings > integrations", "settings, then integrations", "integrations settings", "integrations page"], ["/pebble"]],
         wrong=["/pebbleboard", "members can connect slack", "microsoft teams integration is included"]),
    dict(id="P20", topic="install", title="Calendar sync",
         text="Google Calendar sync is under Settings > Integrations > Calendar. It is one-way: task due dates appear in the calendar, "
              "but changes made in the calendar do not update tasks. Changes sync about every 15 minutes.",
         facts=[["one-way", "one way"], ["15 minutes", "15 min"]],
         wrong=["two-way", "real time", "outlook calendar sync is included"]),
    dict(id="P21", topic="install", title="API tokens and rate limits",
         text="Create API tokens under Settings > Developer. A token is shown only once, so copy it when created. Each token is limited "
              "to 100 requests per minute; above that the API returns HTTP 429.",
         facts=[["100 requests", "100 req", "100 calls", "100 per minute"], ["429", "shown only once", "only once", "shown once"]],
         wrong=["1000 requests", "60 requests", "tokens can be viewed again"]),
    dict(id="P22", topic="install", title="CSV import",
         text="Import tasks under Projects > Import. A CSV file may contain at most 5,000 rows and needs the columns title, assignee and "
              "due_date. Larger files must be split.",
         facts=[["5,000", "5000"], ["split", "smaller files", "multiple files"]],
         wrong=["10,000 rows", "1,000 rows", "50,000 rows"]),
    dict(id="P23", topic="install", title="Exporting workspace data",
         text="Owners and Admins can export workspace data as CSV or JSON under Settings > Workspace > Export. The download link is "
              "emailed and stays valid for 48 hours.",
         facts=[["48 hours", "48-hour", "48 hrs"], ["csv", "json"]],
         wrong=["24 hours link", "7 days link", "pdf export"]),
    dict(id="P24", topic="install", title="File attachments and storage",
         text="The maximum attachment size is 25 MB on Free and 100 MB on Pro and Business. Storage is 5 GB on Free, 100 GB on Pro and "
              "1 TB on Business.",
         facts=[["25 mb"], ["100 mb"]],
         wrong=["500 mb", "10 mb", "unlimited storage"]),
    dict(id="P25", topic="install", title="Email notifications",
         text="Manage notifications under Settings > Notifications. The daily digest is sent at 8:00 in the user's local time. "
              "Individual projects can be muted.",
         facts=[["8:00", "8 am", "8am", "8 a.m."], ["mute", "muted", "muting"]],
         wrong=["9:00", "midnight utc", "weekly digest"]),
    dict(id="P26", topic="install", title="Recurring tasks",
         text="Set a repeat rule on a task: daily, weekly, monthly or custom. The next occurrence is created only when the current one "
              "is marked complete.",
         facts=[["marked complete", "completed", "complete the", "when the current", "is complete"], ["daily", "weekly", "monthly", "custom"]],
         wrong=["created automatically on schedule", "creates all future tasks in advance"]),
    dict(id="P27", topic="account", title="Guest access",
         text="Guests see only the projects they are invited to and cannot see the member list. The Business plan includes up to 10 free "
              "guests. On Pro, each guest counts as a paid seat.",
         facts=[["10 free", "ten free", "up to 10", "10 guests"], ["paid seat", "counts as a seat", "count as a seat", "counts as a paid", "count as a paid", "billable"]],
         wrong=["guests are always free", "20 free guests", "unlimited guests"]),
    dict(id="P28", topic="policy", title="Deleted data and the Trash",
         text="Deleted tasks stay in the Trash for 30 days and can be restored by any Member who can edit the project. After 30 days "
              "they are permanently removed.",
         facts=[["30 days", "30-day", "thirty days"], ["trash"]],
         wrong=["90 days in the trash", "cannot be restored", "7 days"]),
    dict(id="P29", topic="policy", title="Uptime commitment",
         text="The Business plan carries a 99.9% monthly uptime commitment and the Enterprise plan 99.95%. Live status is at "
              "status.pebbleboard.example. Service credits for missed targets are requested through the Billing team.",
         facts=[["99.9%"], ["status.pebbleboard.example", "status page", "billing team", "service credit"]],
         wrong=["99.99%", "100% uptime", "free plan has an sla"]),
    dict(id="P30", topic="policy", title="Data residency and DPA",
         text="A workspace is hosted in the US or EU region, chosen when the workspace is created; the region cannot be changed "
              "afterwards. A data processing agreement (DPA) is available on Business and Enterprise.",
         facts=[["cannot be changed", "can't be changed", "cannot be moved", "can not be changed", "not be changed", "fixed"], ["dpa", "data processing agreement"]],
         wrong=["region can be changed anytime", "asia region", "dpa is available on all plans"]),
]

BY_ID = {p["id"]: p for p in KB}
PRODUCT = "PebbleBoard"
