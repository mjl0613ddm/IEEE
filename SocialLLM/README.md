# SocialLLM

LLM-driven social media polarization simulation: agents post, view, and interact; beliefs update from feedback. Supports counterfactual runs and Shapley value attribution.

## Project structure

```
SocialLLM/
├── config/
│   ├── config.yaml        # Main simulation config (agents, steps, LLM config path)
│   └── api_example.yaml   # Template for API key & model (copy to api.yaml)
├── main.py                # Entry point
├── agents.py              # Agent logic
├── model.py               # Model / environment
├── llm_decision.py        # LLM calls
├── counterfactual.py      # Counterfactual simulation
├── utils.py
├── requirements.txt
└── scripts/               # Shapley, faithfulness, risk, plotting
```

## Install

We recommend using a separate conda environment for SocialLLM to avoid dependency conflicts with other simulations:

```bash
cd SocialLLM
conda create -n socialllm python=3.10
conda activate socialllm
pip install -r requirements.txt
```

## Configuration

1. **API config** (required for LLM runs)  
   Copy the example and set your key (do not commit `api.yaml`):

   ```bash
   cp config/api_example.yaml config/api.yaml
   # Edit config/api.yaml: set api_key, model_name, base_url
   ```

   `config/api.yaml` is in `.gitignore`. Use environment variables or a local file only.

2. **Simulation config**  
   Edit `config/config.yaml` for `num_agents`, `num_steps`, `seed`, and other parameters. The `llm.config_path` should point to `config/api.yaml` after you create it.

## Usage

### Single run

```bash
python main.py --config config/config.yaml --output results/simulation_1
```

### Counterfactual run

```bash
python main.py \
  --mode counterfactual \
  --config config/config.yaml \
  --actions_file results/simulation_1/actions.json \
  --random_states_file results/simulation_1/random_states.json \
  --masked_agent 0 \
  --masked_timestep 5 \
  --output results/counterfactual
```

### Batch runs

Batch scripts use a single API config: `config/api.yaml`. Set your model and key there, then run:

```bash
bash scripts/batch_run_simulation/batch_run_simulation.sh
```

Adjust `MODEL_NAMES` in the script to control output subdirectory names (e.g. `results/run1/`, `results/run2/`).

## Simulation steps (per timestep)

1. **Post**: Each agent uses the LLM to decide whether to post.
2. **View**: Each agent views a random subset of current posts.
3. **Interact**: Each agent uses the LLM to like/dislike/ignore viewed posts.
4. **Belief update**: Beliefs update from feedback (e.g. like → move toward post; dislike → move away).

## Agent metrics

- **belief_value** ∈ [-1, 1]: stance  
- **post_preference**, **interaction_preference**: tendency to post / interact  
- **belief_update_sensitivity**: how much beliefs change from feedback  

## Outputs

- `actions.json`: action history  
- `random_states.json`: random states (reproducibility)  
- `results.json`: final beliefs and polarization risk  

## Analysis

Use the scripts under `scripts/` for Shapley attribution, faithfulness experiments, risk features, and plotting (see each subfolder’s README).
