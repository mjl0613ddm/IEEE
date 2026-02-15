# Simulation System Attribution

LLM-driven economic and social simulation environments for **attribution** (e.g. Shapley value) and **faithfulness** analysis. This repository includes three simulation backends; each can be run and analyzed independently.

## Simulations

| Simulation | Description | Main Entry |
|------------|-------------|------------|
| [EconAgent](EconAgent/) | Macro-economic dynamics with LLM-driven agents (labor, consumption, tax) | `simulate.py` |
| [SocialLLM](SocialLLM/) | Social media polarization with LLM-driven agents | `main.py` |
| [TwinMarket](TwinMarket/) | Stock market simulation with multi-agent trading and social features | `simulation.py` |

Each simulation has its own **dependencies** and **configuration**. See the README in each folder for setup and usage.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Simulation_System_Attribution
   ```

2. **Configure a simulation**  
   API keys and secrets are **not** stored in the repo. For each simulation you use:
   - Copy the provided example config (e.g. `config/api_example.yaml` → `config/api.yaml`, or edit `config.yaml`).
   - Set your API key via the config file or environment variable (see each simulation’s README).

3. **Run the main script**  
   Run the simulation’s main entry (see table above) to generate data.

4. **Run analysis**  
   Use the scripts in each simulation’s `scripts/` folder for Shapley attribution, faithfulness, risk features, and plotting.

## Data and Outputs

**Simulation data and results are not included.** Run the simulations yourself to generate data. Output directories (e.g. `datas/`, `data/`, `results/`, `logs/`) are in `.gitignore`.

## Quick Links

- [EconAgent](EconAgent/README.md) — `config.yaml`, then `python simulate.py` or `bash batch_run_simulations.sh`
- [SocialLLM](SocialLLM/README.md) — Configure `config/`, run `main.py`
- [TwinMarket](TwinMarket/README.md) — Copy `config/api_example.yaml` → `config/api.yaml`, then `bash scripts/run.sh`
