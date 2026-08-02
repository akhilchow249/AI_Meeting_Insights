"""
generate_training_data.py
─────────────────────────
Generates synthetic labelled training data for the three NLP classifiers
described in Project-4.pdf, Section 5 (page 9):

  1. Pain Point Classifier      → 2 000+ labelled sentences
  2. Action Item Classifier     → 1 000+ labelled sentences
  3. Decision Classifier        → 1 000+ labelled sentences

Each record is written to JSONL so it can be loaded directly into
HuggingFace datasets or pandas for fine-tuning.

Usage:
  pip install anthropic tqdm
  export ANTHROPIC_API_KEY=sk-...
  python generate_training_data.py
"""

import json, os, random, time
from pathlib import Path
from tqdm import tqdm
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

OUT_DIR = Path("training_data")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SEED EXAMPLES  (gold-label — never used for training, eval only)
# ─────────────────────────────────────────────────────────────

PAIN_POINT_SEED = {
    "positive": [
        "We are blocked on the authentication service — every deploy breaks the token refresh flow and we have had three incidents in two weeks.",
        "The API rate limit is causing nightly failures in the reporting pipeline and we cannot ship until this is resolved.",
        "Our onboarding funnel has a 70 percent drop-off on step three and nobody knows why.",
        "The vendor has not delivered the integration spec and the whole backend team is waiting on it.",
        "We have unclear requirements from the product team and engineering keeps building the wrong thing.",
        "Every time we push to staging the database migration fails and we lose two hours reverting.",
        "The QA team is understaffed — we have one tester for eight developers and defects are shipping to production.",
        "We do not have enough budget to hire the two engineers we need for the deadline.",
        "The legacy codebase has zero test coverage so every refactor breaks something unexpected.",
        "Cross-team communication is so slow that decisions that should take hours are taking weeks.",
    ],
    "negative": [
        "I don't think we should use that approach.",
        "This is a bit complicated for me to understand.",
        "I slightly prefer the second design option.",
        "We could probably do better on performance.",
        "It would be nice to have a dark mode.",
        "I am not a huge fan of the current colour scheme.",
        "Maybe we should revisit that decision next quarter.",
        "The meeting ran a little longer than expected.",
        "Some team members were late joining the call.",
        "I think there might be a slightly better way to structure this.",
    ],
}

ACTION_ITEM_SEED = {
    "positive": [
        "Can you take ownership of the API integration and have it ready by Friday?",
        "Let's make sure someone from the backend team handles the database migration before next week.",
        "Alice, please send the updated spec to the vendor by end of day.",
        "We need Bob to review the pull request and merge it before the release on Thursday.",
        "Can the QA team prepare a regression test plan by Wednesday morning?",
        "Please schedule a follow-up call with the client before the end of the sprint.",
        "Someone needs to document the new authentication flow in Confluence this week.",
        "Mark, can you set up the Grafana dashboard for the new metrics by Monday?",
        "The team agreed that Sarah will own the incident post-mortem and share findings by Friday.",
        "We need to update the roadmap document and share it with stakeholders before next Tuesday.",
    ],
    "negative": [
        "We discussed the authentication flow in detail.",
        "The team reviewed last quarter's performance numbers.",
        "Everyone agreed that the current architecture is solid.",
        "We have been using this approach for two years.",
        "The client called yesterday to check in on progress.",
        "The sprint velocity has been consistent this quarter.",
        "Most of the team prefers the new design direction.",
        "The product demo went well last week.",
        "We talked about the upcoming conference.",
        "The infrastructure team explained how the load balancer works.",
    ],
}

DECISION_SEED = {
    "positive": [
        "We have decided to migrate the monolith to microservices starting Q3.",
        "Going forward we will use Postgres as the primary database for all new services.",
        "The team agrees that we will freeze feature development two weeks before the release.",
        "It has been decided that all API changes must go through a design review before implementation.",
        "We are officially deprecating the v1 API at the end of this quarter.",
        "The team has resolved to adopt trunk-based development and drop long-lived feature branches.",
        "After discussion we have committed to a four-day work week pilot starting next month.",
        "Leadership has approved the budget for two additional senior engineers.",
        "We will not proceed with the third-party vendor and will build the component in-house.",
        "The team has agreed on a zero-tolerance policy for merging code without a passing test suite.",
    ],
    "negative": [
        "We are still evaluating whether to use Kafka or RabbitMQ.",
        "The team is leaning towards the second option but no final call has been made.",
        "Someone mentioned that TypeScript might be worth considering.",
        "It is unclear who will own this project going forward.",
        "We need more information before we can decide.",
        "There was some discussion about potentially changing the release cadence.",
        "Several people raised concerns about the current approach.",
        "The topic of pricing came up briefly during the meeting.",
        "We briefly touched on the option of outsourcing this work.",
        "The manager mentioned they would think about the staffing question.",
    ],
}


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────

PAIN_CATEGORIES = [
    "technical_blocker",
    "resource_constraint",
    "process_inefficiency",
    "external_dependency",
    "unclear_requirements",
    "team_communication",
]

PAIN_POINT_PROMPT = """You are generating TRAINING DATA for a pain-point classifier used in a meeting-transcription AI system.

Generate exactly {n} sentences spoken in a real business meeting.
Category split:
  - {pos} sentences that ARE genuine pain points (label=1).
    A pain point is a structured problem that blocks or slows progress.
    Vary across these categories: {categories}
  - {neg} sentences that are NOT pain points (label=0).
    These can be opinions, preferences, mild frustrations, or neutral observations.
    They must NOT describe a blocker or systemic problem.

Return ONLY a valid JSON array, no preamble, no markdown fences.
Each element: {{"text": "<sentence>", "label": <0 or 1>, "category": "<category or null>"}}

Rules:
- Write in first-person or direct speech as if spoken aloud in a meeting.
- Positive examples must be specific and describe a real consequence or block.
- Negative examples may be slightly negative in tone but must not describe a systemic problem.
- Vary sentence length, formality, and domain (engineering, product, sales, ops).
"""

ACTION_ITEM_PROMPT = """You are generating TRAINING DATA for an action-item classifier used in a meeting-transcription AI system.

Generate exactly {n} sentences spoken in a real business meeting.
Category split:
  - {pos} sentences that ARE action items (label=1).
    An action item assigns a task to a named or implied owner, optionally with a deadline.
  - {neg} sentences that are NOT action items (label=0).
    These are observations, decisions, or discussions — no task assignment.

Return ONLY a valid JSON array, no preamble, no markdown fences.
Each element: {{"text": "<sentence>", "label": <0 or 1>, "owner": "<name or null>", "deadline": "<deadline or null>"}}

Rules:
- Write as spoken English in a real meeting (informal, direct).
- Vary owner types: named person, role ("the QA team"), implied ("someone should").
- Deadlines vary: "by Friday", "before next sprint", "end of day", null.
- Negative examples include decisions, status updates, and general discussion.
"""

DECISION_PROMPT = """You are generating TRAINING DATA for a decision classifier used in a meeting-transcription AI system.

Generate exactly {n} sentences spoken in a real business meeting.
Category split:
  - {pos} sentences that ARE decisions (label=1).
    A decision is a firm commitment or resolved conclusion reached by the group.
    Signals: "we have decided", "going forward", "the team agrees", "it has been resolved".
  - {neg} sentences that are NOT decisions (label=0).
    These include open discussions, options being considered, or deferred choices.

Return ONLY a valid JSON array, no preamble, no markdown fences.
Each element: {{"text": "<sentence>", "label": <0 or 1>, "speaker_signal": "<key phrase or null>"}}

Rules:
- Write as spoken meeting English.
- Positive examples must sound conclusive, not tentative.
- Negative examples should include hedging language ("might", "considering", "not sure yet").
- Vary domains: engineering, product, business strategy, HR, finance.
"""


# ─────────────────────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_batch(prompt: str, n: int, pos: int, neg: int, **kwargs) -> list[dict]:
    """Call Claude to generate one batch of labelled examples."""
    categories = ", ".join(PAIN_CATEGORIES)
    filled = prompt.format(n=n, pos=pos, neg=neg, categories=categories, **kwargs)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": filled}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def collect_dataset(
    name: str,
    prompt_template: str,
    seed_positive: list[str],
    seed_negative: list[str],
    target_total: int = 2000,
    batch_size: int = 40,
    pos_ratio: float = 0.5,
) -> list[dict]:
    """
    Build a dataset by:
      1. Seeding with hand-crafted gold-label examples.
      2. Generating synthetic silver-label examples in batches.
    """
    records = []

    # 1. Seed gold examples
    for s in seed_positive:
        records.append({"text": s, "label": 1, "split": "eval", "source": "gold"})
    for s in seed_negative:
        records.append({"text": s, "label": 0, "split": "eval", "source": "gold"})

    # 2. Synthetic silver examples
    needed = target_total - len(records)
    batches = needed // batch_size + (1 if needed % batch_size else 0)

    print(f"\n[{name}] Generating {needed} synthetic examples in {batches} batches...")

    for i in tqdm(range(batches), desc=name):
        pos_in_batch = int(batch_size * pos_ratio)
        neg_in_batch = batch_size - pos_in_batch
        try:
            batch = generate_batch(
                prompt_template,
                n=batch_size,
                pos=pos_in_batch,
                neg=neg_in_batch,
            )
            for rec in batch:
                rec["split"] = "train"
                rec["source"] = "synthetic"
            records.extend(batch)
        except Exception as e:
            print(f"  Batch {i} failed: {e}")
        time.sleep(0.5)   # rate-limit courtesy delay

    random.shuffle(records)
    return records


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def save_stats(records: list[dict], name: str):
    total   = len(records)
    pos     = sum(1 for r in records if r["label"] == 1)
    neg     = total - pos
    train   = sum(1 for r in records if r.get("split") == "train")
    eval_   = sum(1 for r in records if r.get("split") == "eval")
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Total: {total}  |  Positive: {pos}  |  Negative: {neg}")
    print(f"  Train: {train}  |  Eval: {eval_}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Pain Point Classifier  (target 2 000+) ──────────────
    pain_data = collect_dataset(
        name="Pain Point Classifier",
        prompt_template=PAIN_POINT_PROMPT,
        seed_positive=PAIN_POINT_SEED["positive"],
        seed_negative=PAIN_POINT_SEED["negative"],
        target_total=2000,
        batch_size=40,
        pos_ratio=0.5,
    )
    save_jsonl(pain_data, OUT_DIR / "pain_point_data.jsonl")
    save_stats(pain_data, "Pain Point Classifier")

    # ── 2. Action Item Classifier  (target 1 000+) ─────────────
    action_data = collect_dataset(
        name="Action Item Classifier",
        prompt_template=ACTION_ITEM_PROMPT,
        seed_positive=ACTION_ITEM_SEED["positive"],
        seed_negative=ACTION_ITEM_SEED["negative"],
        target_total=1000,
        batch_size=40,
        pos_ratio=0.5,
    )
    save_jsonl(action_data, OUT_DIR / "action_item_data.jsonl")
    save_stats(action_data, "Action Item Classifier")

    # ── 3. Decision Classifier  (target 1 000+) ────────────────
    decision_data = collect_dataset(
        name="Decision Classifier",
        prompt_template=DECISION_PROMPT,
        seed_positive=DECISION_SEED["positive"],
        seed_negative=DECISION_SEED["negative"],
        target_total=1000,
        batch_size=40,
        pos_ratio=0.5,
    )
    save_jsonl(decision_data, OUT_DIR / "decision_data.jsonl")
    save_stats(decision_data, "Decision Classifier")

    print("\n✓ All datasets written to ./training_data/")
