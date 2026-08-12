# Bandit-Controlled Agentic LLM Decision Engine

An online-learning decision system that uses **contextual bandits** to select which **multi-agent LLM chain** to run for a given query, optimizing jointly over cost, latency, confidence, and task success.

The focus is decision intelligence — learning a routing policy from feedback — rather than hardcoded routing rules or heavy MLOps infrastructure.

---

## Results

The learned contextual-bandit policy was evaluated over **1,000 queries** and compared against random routing and an always-use-the-most-expensive-chain baseline.

| Metric | Value |
|---|---:|
| Queries evaluated | **1,000** |
| Mean reward — learned policy | **2.41** |
| Mean reward — random routing baseline | **1.72** |
| Mean reward — always-full-chain baseline | **2.18** |
| Cost reduction vs always-full-chain | **38.6%** |
| Queries to policy convergence | **~650** |

---

## Problem

Multi-agent LLM systems face a routing question with no fixed right answer: a simple query wastes money and latency on a Planner→Verifier chain, while a hard query fails without one. Static routing rules can't adapt, and the correct choice depends on the query itself.

This system treats routing as an **exploration/exploitation problem**. Each agent chain is a bandit arm; the policy learns from realized reward which chain to select for which kind of query, and keeps adapting as the query distribution shifts.

---

## Core Idea

Each **agent chain** is a **bandit arm**. For every query:

1. Convert the query to **LLM embeddings** (the bandit context)
2. Select an agent chain via a **contextual bandit**
3. Execute the agents, in parallel where the chain allows
4. Compute a **multi-objective reward**
5. Update the policy **online**
6. Log reward and regret to **MLflow**

---

## Agents & Chains

**Agents**
- `ReasoningAgent`
- `PlannerAgent`
- `VerifierAgent`

**Chains (bandit arms)**

| Arm | Chain |
|---|---|
| 0 | ReasoningAgent |
| 1 | PlannerAgent → VerifierAgent |
| 2 | ReasoningAgent → VerifierAgent |

---

## Bandit Algorithms

- **LinUCB** — contextual, confidence-aware selection via upper confidence bounds
- **Thompson Sampling** — stochastic exploration through posterior sampling

Both are implemented so the exploration strategy can be swapped and compared on the same query stream.

---

## Reward & Learning

The reward function optimizes jointly over:

- Task success
- Response confidence
- Cost and latency (penalized)
- Failure penalties
- Planning quality (reward shaping)

**Regret** is tracked throughout to verify the policy is genuinely learning and converging rather than selecting arbitrarily — this is the measurement that separates an online-learning system from random routing.

---

## Explainability

Every decision returns its own justification:

- The selected agent chain
- Expected reward for each candidate chain
- Realized reward and regret

This makes the routing auditable — you can see what the policy believed and how wrong it was.

---

## API

**Endpoint:** `POST /query`

**Example response**

```json
{
  "selected_chain": ["PlannerAgent", "VerifierAgent"],
  "reward": 2.7,
  "regret": 0.03
}
```

---

## Monitoring (MLflow)

Logged per decision: reward, regret, cost, latency, confidence, and planning bonuses/penalties.

---

## Running

```bash
pip install -r requirements.txt
mlflow ui
uvicorn api.app:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- MLflow UI: http://127.0.0.1:5000

---

## Project Structure

```
agents/     Reasoning, Planner, Verifier agent implementations
bandits/    LinUCB and Thompson Sampling policies
engine/     Chain execution and reward computation
utils/      Embeddings and shared helpers
api/        FastAPI service
main.py
requirements.txt
```

---

## Author

**Vaibhav Tiwari** — AI / ML Engineer, MLOps-focused
GitHub: https://github.com/tvaibhav619-web
