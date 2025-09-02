# Agentic AI + MongoDB for Media Optimization

## Problem Statement

Media companies face two hard problems simultaneously:

1. **Conversion** – turning casual readers into registered users (via register walls).
2. **Retention** – keeping readers engaged over time, so they come back tomorrow.

Traditionally, editors or product managers configure A/B tests by hand:

* One experiment at a time.
* Rigidly defined arms (variants).
* Weeks of waiting for results.

This manual setup is **too slow** and **too limited**. It cannot keep up with the dynamics of real readers and trending content.

---

## Why Agentic AI

With **agentic AI**:

* The system **spawns its own experiments** dynamically when it sees opportunities (e.g. “Politics category is losing engagement, let’s test new homepage rankings there”).
* It **proposes and tests new variants (arms)** automatically (new register-wall copy, new homepage ordering strategies).
* It **retires losers and promotes winners** without human intervention.
* It **learns per user** — not just one “global best”. Each user can see the strategy that fits them best.

This is far beyond manual A/B testing: it’s a **self-optimizing newsroom**.

---

## Why MongoDB

MongoDB is the perfect foundation here:

* **Flexible schema**: stores experiments, arms, assignments, and raw events side by side.
* **Time-series capabilities**: daily aggregates (`campaign_ts`) power dashboards.
* **Rich queries**: quickly segment by user cohorts, categories, or article metadata.
* **Scalable**: can handle both demo scale and production scale.

MongoDB acts as the **memory of the system** — every experiment, every user decision, every success or failure is logged.

---

## How the Solution Works

### 1. Data flows in

* User engagement events (views, scrolls, reads, register clicks) stream into Mongo.
* Article metadata (`news` collection) provides popularity signals.

### 2. The bandit agent makes decisions

* When a user clicks an article → service decides if a **register-wall** is shown.
* When a user loads the homepage → service decides which **homepage ordering strategy** to apply and delivers the sorted list of 16 articles.
* Decisions are logged as **assignments** tied to experiment + arm.

### 3. Experiments and arms

* **Experiment** = a test scope (e.g. “Homepage ordering for Politics readers”).
* **Arms** = the concrete strategies inside it (time\_order, popular\_first, interest\_boosted).
* Experiments keep a list of `arm_ids` for transparency; arms are separate docs for easy updates.

### 4. Rewards and success windows

* **Conversion success** = user clicks the register CTA within 5 minutes of being shown the wall.
* **Momentum success** = user’s momentum metric (`ema7_over_ema28`) rises within 48 hours of seeing a homepage strategy.

Assignments are resolved after these windows and update each arm’s statistics.

### 5. The Planner (Agentic AI)

* Monitors results in Mongo.
* **Spawns new experiments** when performance drifts.
* **Clones successful arms** into new contexts.
* **Retires/merges underperformers**.
* **Adjusts reward weights** between conversion and retention.
  All decisions are recorded with `origin: llm_autospawn`.

---

## How Success Is Visible

For the demo:

* **Aggregate charts per arm** show learning in real time:

  * Assignment counts.
  * Conversion rates.
  * Momentum success rates.
* You can tell the story:

  * “Here, the AI launched three homepage experiments.”
  * “This strategy started weak, but improved and got more traffic.”
  * “This register-wall copy failed, and the system retired it automatically.”

Everything is **autonomous**: no manual configuration, just continuous optimization.

---

## The Key Takeaway

With agentic AI and MongoDB:

* The newsroom optimizes itself.
* Conversion and retention are improved simultaneously.
* The system adapts to readers, categories, and trends in real time.
* The demo clearly shows: **no humans set up these tests — the AI did.**

---

# 🎤 Speaking Notes for Demo

### Slide 1 — Problem Statement

* *“Media companies have two battles: getting people to register, and keeping them coming back.
  Right now, those are handled with slow, manual A/B tests. One test at a time, weeks to learn anything. Too slow.”*

---

### Slide 2 — Why Agentic AI

* *“Instead of manually configuring tests, our system does this automatically.
  It invents experiments, tries multiple options at once, kills off bad ones, and adapts per user.
  This is more than A/B testing — it’s a newsroom that optimizes itself.”*

---

### Slide 3 — Why MongoDB

* *“MongoDB is our memory.
  It stores every experiment, every arm, every user decision, every outcome.
  It’s schema-flexible for experiments, great for time series, and powerful for real-time queries.”*

---

### Slide 4 — How It Works (flow diagram)

* *“When a user clicks an article, the system decides: show a register wall or not.
  When a user opens the homepage, the system chooses the best ordering strategy and sends back a ready-made list of articles.
  Every decision is logged, every outcome is measured.”*

---

### Slide 5 — Experiments vs Arms

* *“Think of experiments as the test container.
  Example: Homepage ordering.
  Inside each experiment, we have arms — specific strategies like newest first, popular first, or interest-boosted.
  Experiments can run in parallel, and the AI spawns new ones as needed.”*

---

### Slide 6 — Rewards & Success

* *“For register walls: success is a register CTA click within 5 minutes.
  For homepage ordering: success is if the user’s momentum goes up within 48 hours.
  Assignments are resolved automatically, and arms update their stats.”*

---

### Slide 7 — Planner Autonomy

* *“Here’s the magic: the Planner, powered by LLM, monitors performance.
  It spawns new experiments when it sees drift, clones successful strategies into new categories, retires losers, merges duplicates, even shifts weights between conversion and retention.
  All autonomous. No human needed.”*

---

### Slide 8 — Demo Dashboard

* *“This chart shows assignments, conversion rates, and momentum success rates per arm.
  Notice how the system gives more users to winning strategies over time, and disables bad ones.
  Every experiment here was invented by the AI itself.”*

---

### Slide 9 — Key Takeaway

* *“With agentic AI and MongoDB, the newsroom optimizes itself.
  It learns what works for each user, in each category, in real time.
  We don’t set up the tests. The AI does. And that’s the breakthrough.”*

---
