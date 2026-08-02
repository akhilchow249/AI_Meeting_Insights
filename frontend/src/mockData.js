/* src/mockData.js — realistic demo data for all 5 screens */

export const MOCK_SESSIONS = [
  {
    session_id: 'sess-001',
    filename: 'Q3_Product_Roadmap_Review.mp4',
    created_at: '1719820800',
    duration: '3847',
    num_speakers: 4,
    stage: 'indexing',
    status: 'complete',
    pain_point_count: 7,
    action_item_count: 12,
  },
  {
    session_id: 'sess-002',
    filename: 'Engineering_Sprint_Retro.mp4',
    created_at: '1719734400',
    duration: '2210',
    num_speakers: 6,
    stage: 'indexing',
    status: 'complete',
    pain_point_count: 3,
    action_item_count: 8,
  },
  {
    session_id: 'sess-003',
    filename: 'Investor_Update_June_2024.mp4',
    created_at: '1719648000',
    duration: '5423',
    num_speakers: 3,
    stage: 'nlp_analysis',
    status: 'running',
    pain_point_count: 0,
    action_item_count: 0,
  },
  {
    session_id: 'sess-004',
    filename: 'Design_System_Kickoff.mp4',
    created_at: '1719561600',
    duration: '4102',
    num_speakers: 5,
    stage: 'diarisation',
    status: 'failed',
    pain_point_count: 0,
    action_item_count: 0,
  },
  {
    session_id: 'sess-005',
    filename: 'Sales_Pipeline_Review.mp4',
    created_at: '1719475200',
    duration: '2980',
    num_speakers: 4,
    stage: 'indexing',
    status: 'complete',
    pain_point_count: 11,
    action_item_count: 9,
  },
]

export const SPEAKER_COLORS = ['#f0a000','#00d2c8','#4d9fff','#c97bff','#ff8a65','#69e36b']
export const SPEAKER_NAMES  = ['Alex Chen','Morgan Lee','Jordan Kim','Riley Park','Sam Torres','Taylor Wu']

export const MOCK_TRANSCRIPT = [
  { id: 0, speaker: 0, start: 0,    end: 18,   text: "Alright everyone, let's kick off the Q3 roadmap review. I want to make sure we align on priorities before the board presentation next week." },
  { id: 1, speaker: 1, start: 19,   end: 41,   text: "Thanks. Before we dive in — the mobile redesign has slipped by two sprints. We need to decide today whether we push the launch or descope some features." },
  { id: 2, speaker: 2, start: 42,   end: 67,   text: "From the engineering side, the blocker is the new payment SDK. We only got sandbox access last Thursday, and production creds are still pending from the finance team." },
  { id: 3, speaker: 3, start: 68,   end: 89,   text: "That's on me — I'll have the credentials over by end of day today. But I do want to flag that our current burn rate puts us about fourteen weeks from needing another funding event." },
  { id: 4, speaker: 0, start: 90,   end: 118,  text: "Okay so two separate issues. First: payment SDK blockers we can unblock today. Second: roadmap prioritisation given the runway concern. Let me start a shared doc so we capture decisions in real time." },
  { id: 5, speaker: 1, start: 119,  end: 145,  text: "On the SDK issue — even with creds today, Jordan's team needs at least three days of integration testing before we'd be comfortable promoting to staging. So we're looking at end of next week at the earliest." },
  { id: 6, speaker: 2, start: 146,  end: 172,  text: "Correct. And that's assuming no critical bugs come up in testing. I'd estimate maybe a thirty percent chance we find something that pushes us another week. We should plan for the pessimistic case." },
  { id: 7, speaker: 4, start: 173,  end: 198,  text: "On the product side, our users have been specifically requesting the dark mode feature — the in-app survey last month showed it as the number one feature request with sixty-two percent of respondents." },
  { id: 8, speaker: 0, start: 199,  end: 228,  text: "Good point. So dark mode has clear user demand. What's the engineering effort estimate for that versus the payment integration? I want to weigh them properly before we decide what to descope." },
  { id: 9, speaker: 2, start: 229,  end: 255,  text: "Dark mode is roughly a week of engineering with our current component system. Payment integration is three to four weeks. They're not really comparable in scope — payments is a much larger lift." },
  { id: 10, speaker: 3, start: 256, end: 284,  text: "From a revenue perspective, unblocking payments unlocks the B2B tier which is forty percent of our pipeline. Dark mode is important for retention but doesn't directly convert enterprise deals." },
  { id: 11, speaker: 1, start: 285, end: 312,  text: "So the recommendation is: prioritise payments, push the mobile launch by three weeks, and treat dark mode as a fast follow in Q4. We should communicate this to our waiting-list users today." },
]

export const MOCK_REPORT = {
  summary: `This 64-minute Q3 product roadmap review brought together product, engineering, finance, and design stakeholders to align on launch priorities ahead of the board presentation. The primary output was a decision to delay the mobile launch by three weeks to accommodate payment SDK integration, with dark mode deferred to Q4.`,

  key_decisions: [
    "Mobile launch delayed by 3 weeks — consensus reached",
    "Payment SDK integration prioritised over dark mode",
    "Finance team (Riley) to deliver production credentials by EOD",
    "Dark mode deferred to Q4 as fast-follow",
    "Board presentation deck to reflect revised timeline",
  ],

  pain_points: [
    { text: "Payment SDK production credentials blocked for 5+ days", severity: 'high', speaker: 'Jordan Kim' },
    { text: "Mobile launch already 2 sprints behind schedule", severity: 'high', speaker: 'Morgan Lee' },
    { text: "14-week runway pressure creating prioritisation tension", severity: 'high', speaker: 'Riley Park' },
    { text: "Integration testing estimate has 30% risk of further delay", severity: 'medium', speaker: 'Jordan Kim' },
    { text: "User-facing dark mode request going unaddressed in Q3", severity: 'medium', speaker: 'Sam Torres' },
    { text: "Real-time decision capture process was ad-hoc", severity: 'low', speaker: 'Alex Chen' },
    { text: "No fallback plan documented if payment integration slips again", severity: 'low', speaker: 'Morgan Lee' },
  ],

  action_items: [
    { action: "Deliver payment SDK production credentials", owner: "Riley Park",  due: "Today EOD",    priority: 'critical' },
    { action: "Begin payment SDK integration testing",     owner: "Jordan Kim",   due: "Mon 10 Jun",   priority: 'high'     },
    { action: "Draft updated mobile launch timeline",      owner: "Morgan Lee",   due: "Mon 10 Jun",   priority: 'high'     },
    { action: "Communicate delay to waiting-list users",   owner: "Sam Torres",   due: "Today EOD",    priority: 'high'     },
    { action: "Update board presentation deck",            owner: "Alex Chen",    due: "Wed 12 Jun",   priority: 'medium'   },
    { action: "Document dark mode for Q4 planning",        owner: "Taylor Wu",    due: "Fri 14 Jun",   priority: 'medium'   },
    { action: "Define pessimistic launch contingency plan",owner: "Jordan Kim",   due: "Tue 11 Jun",   priority: 'medium'   },
    { action: "Share shared doc from meeting with team",   owner: "Alex Chen",    due: "Today EOD",    priority: 'low'      },
    { action: "Review in-app survey full results",         owner: "Sam Torres",   due: "Fri 14 Jun",   priority: 'low'      },
  ],

  sentiment_data: [
    { t: '0:00', alex: 0.7, morgan: 0.6, jordan: 0.5, riley: 0.4 },
    { t: '8:00', alex: 0.6, morgan: 0.4, jordan: 0.3, riley: 0.2 },
    { t: '16:00',alex: 0.5, morgan: 0.5, jordan: 0.4, riley: 0.3 },
    { t: '24:00',alex: 0.6, morgan: 0.6, jordan: 0.5, riley: 0.5 },
    { t: '32:00',alex: 0.7, morgan: 0.5, jordan: 0.6, riley: 0.4 },
    { t: '40:00',alex: 0.8, morgan: 0.7, jordan: 0.6, riley: 0.6 },
    { t: '48:00',alex: 0.7, morgan: 0.8, jordan: 0.7, riley: 0.7 },
    { t: '56:00',alex: 0.8, morgan: 0.7, jordan: 0.8, riley: 0.8 },
    { t: '64:00',alex: 0.9, morgan: 0.8, jordan: 0.7, riley: 0.7 },
  ],

  speaker_stats: [
    { speaker: 0, name: 'Alex Chen',    speaking_time: 218, speaking_pct: 34, sentiment: 0.74, word_count: 412, action_items: 2 },
    { speaker: 1, name: 'Morgan Lee',   speaking_time: 156, speaking_pct: 24, sentiment: 0.61, word_count: 297, action_items: 2 },
    { speaker: 2, name: 'Jordan Kim',   speaking_time: 143, speaking_pct: 22, sentiment: 0.58, word_count: 271, action_items: 3 },
    { speaker: 3, name: 'Riley Park',   speaking_time: 87,  speaking_pct: 14, sentiment: 0.52, word_count: 164, action_items: 1 },
    { speaker: 4, name: 'Sam Torres',   speaking_time: 41,  speaking_pct: 6,  sentiment: 0.71, word_count: 78,  action_items: 2 },
  ],

  topics: ["Product Roadmap", "Payment SDK Integration", "Mobile Launch Timeline", "Dark Mode Feature", "Q3 Priorities", "Runway Concerns"],

  follow_ups: [
    "How does the payment SDK delay affect the B2B enterprise pipeline?",
    "What is the contingency plan if the 3-week delay extends further?",
    "When should dark mode be officially added to the Q4 roadmap?",
  ],
}

export const MOCK_PIPELINE_STAGES = [
  { key: 'video_ingestion',    label: 'Video Ingestion',      icon: '📥', desc: 'Validate format, extract metadata' },
  { key: 'audio_extraction',   label: 'Audio Extraction',     icon: '🎵', desc: 'Extract, denoise, VAD segmentation' },
  { key: 'transcription',      label: 'Speech-to-Text',       icon: '🗣️', desc: 'Whisper ASR with word timestamps' },
  { key: 'diarisation',        label: 'Speaker Diarisation',  icon: '👥', desc: 'pyannote speaker identification' },
  { key: 'nlp_analysis',       label: 'NLP Analysis',         icon: '🧠', desc: 'Pain points, actions, topics, sentiment' },
  { key: 'genai_report',       label: 'AI Report Generation', icon: '✨', desc: 'Claude 7-section meeting report' },
  { key: 'indexing',           label: 'Indexing',             icon: '🔍', desc: 'Full-text search index update' },
]

export const MOCK_METRICS = {
  latency: [
    { stage: 'Video Ingestion',    p50: 1200, p95: 3800  },
    { stage: 'Audio Extraction',   p50: 8400, p95: 22000 },
    { stage: 'Speech-to-Text',     p50: 42000,p95: 98000 },
    { stage: 'Diarisation',        p50: 18000,p95: 67000 },
    { stage: 'NLP Analysis',       p50: 9200, p95: 24000 },
    { stage: 'AI Report',          p50: 4100, p95: 8800  },
    { stage: 'Indexing',           p50: 620,  p95: 1400  },
  ],
  confidence: [
    { bucket: '0.50', count: 2  },
    { bucket: '0.60', count: 5  },
    { bucket: '0.65', count: 9  },
    { bucket: '0.70', count: 18 },
    { bucket: '0.75', count: 34 },
    { bucket: '0.80', count: 61 },
    { bucket: '0.85', count: 88 },
    { bucket: '0.90', count: 72 },
    { bucket: '0.95', count: 43 },
    { bucket: '1.00', count: 12 },
  ],
  failureRates: [
    { stage: 'Video Ingestion',  failures: 3,  total: 148 },
    { stage: 'Audio Extraction', failures: 7,  total: 145 },
    { stage: 'Speech-to-Text',   failures: 2,  total: 138 },
    { stage: 'Diarisation',      failures: 12, total: 136 },
    { stage: 'NLP Analysis',     failures: 4,  total: 124 },
    { stage: 'AI Report',        failures: 1,  total: 120 },
    { stage: 'Indexing',         failures: 0,  total: 119 },
  ],
  queueDepth: [
    { time: '00:00', depth: 0 }, { time: '04:00', depth: 1 },
    { time: '08:00', depth: 8 }, { time: '10:00', depth: 14 },
    { time: '12:00', depth: 11 }, { time: '14:00', depth: 7 },
    { time: '16:00', depth: 9 }, { time: '18:00', depth: 5 },
    { time: '20:00', depth: 2 }, { time: '22:00', depth: 1 },
  ],
}
