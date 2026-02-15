# EconAgent Quick Start

This project simulates macro-economic dynamics with language-model-driven agents. The current workflow assumes **remote LLM APIs only** (no bundled local models) and adds support for resuming partially completed simulations.

## Main Simulation Script

**`simulate.py`** is the primary entry point for running simulations. Run it first to generate data; outputs go to `<save_path>/data/<run_name>/` and `<save_path>/figs/<run_name>/`. Then use the scripts in `scripts/` for analysis (Shapley attribution, faithfulness, risk features, plotting).

## 1. Configure the Simulation

1. Edit `config.yaml`:
   - Under `simulation`, set:
     - `run_name`: label for the output run (used to name the `data/` and `figs/` directories).
     - `save_path`: root folder where results will be written.
   - Under `llm`, ensure remote API settings are correct:
     - `provider`: leave as `openai_proxy` (or your compatible proxy name).
     - `api_base`: endpoint URL for the proxy/API gateway.
     - `model`: remote model identifier (e.g. `gpt-4o`).
     - `api_key_env`: name of the environment variable holding the key (e.g. `OPENAI_API_KEY`).
     - Optionally set `api_key` directly in the file (not recommended for shared repos).

2. Provide the API key at runtime:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   or leave the key inline in `config.yaml` (only for isolated/local use).

## 2. Run a Simulation

Use the main entry point `simulate.py`:

```bash
python simulate.py \
  --policy_model baseline \
  --num_agents 10 \
  --episode_length 50
```

Key flags:
- `policy_model`: `gpt` (LLM-driven agents) or `complex` (hand-coded baseline).
- `num_agents`, `episode_length`: override defaults from `config.yaml` if desired.
- Additional optional flags (for GPT runs):
  - `dialog_len`: number of recent prompt/response turns retained per agent (default 3).
  - `max_price_inflation`, `max_wage_inflation`: forwarded to consumption component scaling.
  - `dense_log_frequency`: controls how often environment snapshots (including price/inflation metrics) are written; defaults to `1` (every step).

Outputs are written to:
```
<save_path>/data/<run_name>/...
<save_path>/figs/<run_name>/...
```
The data directory contains per-step pickles for actions, observations, environment state, dialog queues, dense logs, and a `run_state.pkl` checkpoint.

## 3. Resume After Interruption

If a run stops mid-way, you can continue from a chosen step using the per-step snapshots:

```bash
python simulate.py \
  --policy_model gpt \
  --num_agents 100 \
  --episode_length 240 \
  --resume_step 180 \
  --resume_dir <save_path>/data/<run_name>
```

Guidelines:
- `resume_step` should match the latest completed step (e.g. resume from 180 to continue with step 181).
- `resume_dir` defaults to `<save_path>/data/<run_name>`; supply it explicitly when resuming from archival locations or alternative directories.
- The script restores environment state, dialog history, accumulated cost, and error counters from the corresponding pickles and `run_state.pkl`.
- Dialog snapshots are required for GPT runs. If missing, resumption will fail with a clear error.

## 4. Troubleshooting

- **API Key errors**: ensure the environment variable named in `config.yaml:llm.api_key_env` is exported, or set `llm.api_key` directly.
- **Model errors or empty responses**: verify `llm.api_base` and `llm.model` match the proxy’s supported models.
- **Resume failures**: confirm that the per-step `env_*.pkl`, `obs_*.pkl`, dialog pickles, and `run_state.pkl` exist in the run directory.

## 5. Legacy Notes

Earlier versions shipped with local Qwen inference helpers and no resume logic. Those paths are deprecated—this README reflects the streamlined remote-API workflow introduced in 2025-02.
