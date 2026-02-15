import fire
import os
import json
import csv
import random
import re

import ai_economist.foundation as foundation
import numpy as np
import yaml
from time import time
from collections import deque
import simulate_utils as sim_utils
from simulate_utils import *
import pickle as pkl
from dateutil.relativedelta import relativedelta
from datetime import datetime

# 支持从环境变量读取配置文件路径（用于并行任务）
import os
config_file_path = os.getenv('ECONAGENT_CONFIG_FILE', 'config.yaml')
with open(config_file_path, "r") as f:
    run_configuration = yaml.safe_load(f)
env_config = run_configuration.get('env')
simulation_config = run_configuration.get('simulation', {})
llm_config = run_configuration.get('llm', {})

_configured_save_path = simulation_config.get('save_path')
_configured_run_name = simulation_config.get('run_name')

sim_utils.configure_simulation(
    save_path_override=_configured_save_path,
    llm_overrides=llm_config,
)

save_path = sim_utils.save_path


def _load_resume_state(run_dir, resume_step, policy_model, dialog_len):
    run_dir = os.path.abspath(run_dir)
    env_path = os.path.join(run_dir, 'env', f'env_{resume_step}.pkl')
    obs_path = os.path.join(run_dir, 'obs', f'obs_{resume_step}.pkl')

    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Missing environment snapshot: {env_path}")
    if not os.path.exists(obs_path):
        raise FileNotFoundError(f"Missing observation snapshot: {obs_path}")

    with open(env_path, 'rb') as f:
        env = pkl.load(f)
    with open(obs_path, 'rb') as f:
        obs = pkl.load(f)

    dialog_queue = None
    dialog4ref_queue = None
    if policy_model == 'gpt':
        dialog_path = os.path.join(run_dir, 'dialog_pickles', f'dialog_{resume_step}.pkl')
        dialog4ref_path = os.path.join(run_dir, 'dialog4ref_pickles', f'dialog4ref_{resume_step}.pkl')
        if not os.path.exists(dialog_path) or not os.path.exists(dialog4ref_path):
            raise FileNotFoundError("Missing dialog history required for resuming LLM-controlled agents.")
        with open(dialog_path, 'rb') as f:
            raw_dialog_queue = pkl.load(f)
        with open(dialog4ref_path, 'rb') as f:
            raw_dialog4ref_queue = pkl.load(f)
        dialog_queue = [deque(d, maxlen=dialog_len) for d in raw_dialog_queue]
        dialog4ref_queue = [deque(d, maxlen=7) for d in raw_dialog4ref_queue]

    state_path = os.path.join(run_dir, 'run_state.pkl')
    resume_state = {}
    if os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            resume_state = pkl.load(f) or {}

    return env, obs, dialog_queue, dialog4ref_queue, resume_state

def gpt_actions(env, obs, dialog_queue, dialog4ref_queue, gpt_path, gpt_error, total_cost):
    import sys
    if not os.path.exists(gpt_path):
        os.makedirs(gpt_path)
    curr_rates = obs['p']['PeriodicBracketTax-curr_rates']
    current_datetime = world_start_time + relativedelta(months=env.world.timestep)
    current_time = current_datetime.strftime('%Y.%m')
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        skill = this_agent.state['skill']
        wealth = this_agent.inventory['Coin']
        consumption = this_agent.consumption['Coin']
        interest_rate = env.world.interest_rate[-1]
        price = env.world.price[-1]
        tax_paid = obs['p'][f'p{idx}']['PeriodicBracketTax-tax_paid']
        lump_sum = obs['p'][f'p{idx}']['PeriodicBracketTax-lump_sum']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        name = this_agent.endogenous['name']
        age = this_agent.endogenous['age']
        city = this_agent.endogenous['city']
        job = this_agent.endogenous['job']
        offer = this_agent.endogenous['offer']
        actions = env.dense_log['actions']
        states = env.dense_log['states']
        problem_prompt = f'''
                    You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.
                    Now it's {current_time}.
                '''
        if current_datetime >= datetime(2004, 1, 1):
        # if env.world.timestep >= 50:
            problem_prompt += (
                " In response to the large-scale outbreak of influenza A(H1N1) in the United States, "
                "the federal government has declared a national emergency since January 2004."
            )
        if job == 'Unemployment':
            job_prompt = f'''
                        In the previous month, you became unemployed and had no income. Now, you are invited to work as a(an) {offer} with monthly salary of ${skill*max_l:.2f}.
                    '''
        else:
            if skill >= states[-1][str(idx)]['skill']:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is increased compared to the last month due to the inflation of labor market.
                        '''
            else:
                job_prompt = f'''
                            In the previous month, you worked as a(an) {job}. If you continue working this month, your expected income will be ${skill*max_l:.2f}, which is decreased compared to the last month due to the deflation of labor market.
                        '''
        if (consumption <= 0) and (len(actions) > 0) and (actions[-1].get('SimpleConsumption', 0) > 0):
            consumption_prompt = f'''
                        Besides, you had no consumption due to shortage of goods.
                    '''
        else:
            consumption_prompt = f'''
                        Besides, your consumption was ${consumption:.2f}.
                    '''
        if env._components_dict['PeriodicBracketTax'].tax_model == 'us-federal-single-filer-2018-scaled':
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, the government sets the brackets: {format_numbers(brackets)} and their corresponding rates: {format_numbers(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        else:
            tax_prompt = f'''Your tax deduction amounted to ${tax_paid:.2f}. However, as part of the government's redistribution program, you received a credit of ${lump_sum:.2f}.
                            In this month, according to the optimal taxation theory, Saez Tax, the brackets are not changed: {format_numbers(brackets)} but the government has updated corresponding rates: {format_percentages(curr_rates)}. Income earned within each bracket is taxed only at that bracket's rate.'''
        if env.world.timestep == 0:
            price_prompt = f'''Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}.'''
        else:
            if price >= env.world.price[-2]:
                price_prompt = f'''Meanwhile, inflation has led to a price increase in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
            else:
                price_prompt = f'''Meanwhile, deflation has led to a price decrease in the consumption market, with the average price of essential goods now at ${price:.2f}.'''
        
        # Inject inflation information to selected agents only (step 35 onwards)
        inflation_info_prompt = ""
        if env.world.timestep >= 50:
            # Option 1: Select specific agent IDs (e.g., first 3 agents: 0, 1, 2)
            # selected_agents = []  # Modify this list to choose which agents receive the info
            # if idx in selected_agents:
            #     inflation_info_prompt = " MARKET ALERT: Economic forecasts indicate rising inflation is likely over the coming months. Experts suggest prices may increase significantly (5-7% expected), and many are recommending purchasing essential goods sooner rather than later, as prices are likely to continue rising. Supply chain disruptions and increased demand are anticipated to drive up costs."
            
            # Option 2: Random selection (uncomment to use instead)
            if np.random.random() < 0.3:  # 30% of agents receive the info
                inflation_info_prompt = " MARKET ALERT: Economic forecasts indicate rising inflation is likely over the coming months. Experts suggest prices may increase significantly (5-7% expected), and many are recommending purchasing essential goods sooner rather than later, as prices are likely to continue rising. Supply chain disruptions and increased demand are anticipated to drive up costs."
            
            # Option 3: Select agents by wealth/income (e.g., top 30% by wealth)
            # agent_wealths = [env.get_agent(str(i)).inventory['Coin'] for i in range(env.num_agents)]
            # wealth_threshold = np.percentile(agent_wealths, 70)  # Top 30%
            # if wealth >= wealth_threshold:
            #     inflation_info_prompt = " MARKET ALERT: Economic forecasts indicate rising inflation is likely over the coming months. Experts suggest prices may increase significantly (5-7% expected), and many are recommending purchasing essential goods sooner rather than later, as prices are likely to continue rising. Supply chain disruptions and increased demand are anticipated to drive up costs."
        
        job_prompt = prettify_document(job_prompt)
        obs_prompt = f'''
                        {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt} {inflation_info_prompt}
                        Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. 
                        With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?
                        Please share your decisions in a JSON format. The format should have two keys: 'work' (a value between 0 and 1 with intervals of 0.02, indicating the willingness or propensity to work) and 'consumption' (a value between 0 and 1 with intervals of 0.02, indicating the proportion of all your savings and income you intend to spend on essential goods). Respond with only the JSON object, without markdown fences or explanations.
                    '''
        obs_prompt = prettify_document(obs_prompt)
        dialog_queue[idx].append({'role': 'user', 'content': obs_prompt})
        dialog4ref_queue[idx].append({'role': 'user', 'content': obs_prompt})
    
    def action_check(actions):
        if len(actions) != 2:
            return False
        else:
            return (actions[0] >= 0) & (actions[0] <= 1) & (actions[1] >= 0) & (actions[1] <= 1)
    
    def parse_json_response(content):
        """Parse JSON from LLM response, handling markdown code blocks."""
        if not content or not isinstance(content, str):
            raise ValueError("Invalid content type")
        
        # Save original content for fallback
        original_content = content
        
        # Remove markdown code blocks if present - use simple string replacement first
        content = content.strip()
        
        # Remove ```json and ``` markers - try multiple approaches
        # First, try to remove complete markdown code blocks
        content = re.sub(r'```json\s*\n?', '', content, flags=re.IGNORECASE)
        content = re.sub(r'```\s*\n?', '', content)
        content = re.sub(r'\n?```\s*', '', content)
        # Also handle cases where there's no newline
        content = content.replace('```json', '').replace('```', '')
        content = content.strip()
        
        # Try to extract JSON object if there's extra text
        # Find the first { and match to the corresponding }
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(content):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    content = content[start_idx:i+1]
                    break
        
        # If we didn't find a complete JSON object, try alternative methods
        if start_idx == -1 or brace_count != 0:
            # If content already starts with {, try to parse it as-is (might be incomplete)
            if content.startswith('{'):
                # Try to find the last } if content might be truncated
                last_brace = content.rfind('}')
                if last_brace > 0:
                    content = content[:last_brace+1]
                else:
                    # Content starts with { but has no closing }, try to extract values and complete it
                    # Extract work and consumption values using regex
                    work_match = re.search(r'"work"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
                    consumption_match = re.search(r'"consumption"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
                    
                    if work_match or consumption_match:
                        work_val = work_match.group(1) if work_match else "1.0"
                        consumption_val = consumption_match.group(1) if consumption_match else "0.5"
                        content = f'{{"work": {work_val}, "consumption": {consumption_val}}}'
            else:
                # Try to find JSON anywhere in the content using regex
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                else:
                    # Last resort: try to find any { ... } pattern
                    json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(0)
        
        # Clean up any remaining whitespace/newlines
        content = content.strip()
        
        # Final validation: ensure we have something that looks like JSON
        if not content or not content.startswith('{'):
            # Try to extract work and consumption values from original content
            work_match = re.search(r'"work"\s*:\s*([0-9.]+)', original_content, re.IGNORECASE)
            consumption_match = re.search(r'"consumption"\s*:\s*([0-9.]+)', original_content, re.IGNORECASE)
            
            if work_match or consumption_match:
                work_val = work_match.group(1) if work_match else "1.0"
                consumption_val = consumption_match.group(1) if consumption_match else "0.5"
                content = f'{{"work": {work_val}, "consumption": {consumption_val}}}'
            else:
                raise ValueError(f"No valid JSON object found. Original content preview: {original_content[:200]}")
        
        # If JSON is incomplete (missing closing brace), try to complete it
        if content.startswith('{') and not content.endswith('}'):
            # Extract values and reconstruct
            work_match = re.search(r'"work"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
            consumption_match = re.search(r'"consumption"\s*:\s*([0-9.]+)', content, re.IGNORECASE)
            
            if work_match or consumption_match:
                work_val = work_match.group(1) if work_match else "1.0"
                consumption_val = consumption_match.group(1) if consumption_match else "0.5"
                content = f'{{"work": {work_val}, "consumption": {consumption_val}}}'
        
        # Parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to fix common issues
            # Remove any trailing commas
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Last resort: try eval for Python dict format (less safe but sometimes works)
                try:
                    # Only use eval if it looks like a Python dict
                    if content.startswith('{') and content.endswith('}'):
                        return eval(content)
                    else:
                        raise ValueError(f"Content does not look like JSON or dict: {content[:100]}")
                except (SyntaxError, ValueError) as eval_error:
                    raise json.JSONDecodeError(
                        f"Failed to parse JSON. Original error: {e}, Eval error: {eval_error}",
                        content, 0
                    )
    
    if env.world.timestep%3 == 0 and env.world.timestep > 0:
        results, cost = get_multiple_completion([list(dialogs)[:2] + list(dialog4ref)[-3:-1] + list(dialogs)[-1:] for dialogs, dialog4ref in zip(dialog_queue, dialog4ref_queue)], max_tokens=200)
        total_cost += cost
    else:
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog_queue], max_tokens=200)
        total_cost += cost
    actions = {}
    for idx in range(env.num_agents):
        content = results[idx]
        try:
            parsed_json = parse_json_response(content)
            extracted_actions = list(parsed_json.values())
            if not action_check(extracted_actions):
                # Log invalid values for debugging
                if env.world.timestep % 10 == 0:  # Only log occasionally to avoid spam
                    print(f"  Agent {idx}: Invalid action values {extracted_actions}, using default [1, 0.5]")
                extracted_actions = [1, 0.5]
                gpt_error += 1
        except Exception as e:
            extracted_actions = [1, 0.5]
            gpt_error += 1
        extracted_actions[0] = int(np.random.uniform() <= extracted_actions[0])
        extracted_actions[1] /= 0.02
        actions[str(idx)] = extracted_actions
        dialog_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
        dialog4ref_queue[idx].append({'role': 'assistant', 'content': f'{content}'})
    actions['p'] = [0]
    for idx, agent_dialog in enumerate(dialog_queue):
        with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
            for dialog in list(agent_dialog)[-2:]:
                f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
        
    if (env.world.timestep+1)%3 == 0:
        reflection_prompt = '''Given the previous quarter's economic environment, reflect on the labor, consumption, and financial markets, as well as their dynamics. What conclusions have you drawn?
        Your answer must be less than 200 words!'''
        reflection_prompt = prettify_document(reflection_prompt)
        for idx in range(env.num_agents):
            # dialog_queue[idx].append({'role': 'user', 'content': reflection_prompt})
            dialog4ref_queue[idx].append({'role': 'user', 'content': reflection_prompt})
        results, cost = get_multiple_completion([list(dialogs) for dialogs in dialog4ref_queue], temperature=0, max_tokens=200)
        total_cost += cost
        for idx in range(env.num_agents):
            content = results[idx]
            # dialog_queue[idx].append({'role': 'assistant', 'content': content})
            dialog4ref_queue[idx].append({'role': 'assistant', 'content': content})
        
        for idx, agent_dialog in enumerate(dialog4ref_queue):
             with open(f'''{gpt_path}/{env.get_agent(str(idx)).endogenous['name']}''', 'a') as f:
                for dialog in list(agent_dialog)[-2:]:
                    f.write(f'''>>>>>>>>>{dialog['role']}: {dialog['content']}\n''')
    return actions, gpt_error, total_cost

def _convert_to_json_serializable(obj):
    """Convert numpy types and other non-JSON-serializable objects to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_json_serializable(item) for item in obj)
    else:
        return obj

def _calculate_gini(wealths):
    """Calculate Gini coefficient for wealth distribution."""
    if len(wealths) == 0 or np.sum(wealths) == 0:
        return 0.0
    wealths = np.array(wealths)
    n = len(wealths)
    sorted_wealths = np.sort(wealths)
    cumsum = np.cumsum(sorted_wealths)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_wealths)) / (n * np.sum(wealths)) - (n + 1) / n

def _extract_agent_metrics(env, timestep):
    """Extract key metrics for each agent from the environment at a given timestep.
    
    Returns a list of dictionaries, one per agent, with metrics like:
    - agent_id, timestep, wealth, income, consumption, skill, etc.
    """
    metrics = []
    for idx in range(env.num_agents):
        agent = env.get_agent(str(idx))
        agent_metrics = {
            'agent_id': str(idx),
            'timestep': timestep,
            'wealth': float(agent.inventory.get('Coin', 0)),
            'income': float(agent.income.get('Coin', 0)),
            'consumption': float(agent.consumption.get('Coin', 0)),
            'skill': float(agent.state.get('skill', 0)),
        }
        # Add endogenous attributes if available
        if hasattr(agent, 'endogenous'):
            for key, value in agent.endogenous.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    agent_metrics[f'endogenous_{key}'] = value
        
        metrics.append(agent_metrics)
    return metrics

def _extract_world_metrics(env, timestep):
    """Extract comprehensive world-level metrics including inflation, unemployment, Gini, etc.
    
    Returns a dictionary with macro-economic indicators:
    - timestep, price, interest_rate
    - price_inflation_rate: 价格通胀率（相对于上一期）
    - unemployment_rate: 失业率
    - gini_wealth: 财富基尼系数
    - gini_income: 收入基尼系数
    - total_wealth: 总财富
    - total_income: 总收入
    - total_consumption: 总消费
    - avg_wealth: 平均财富
    - avg_income: 平均收入
    """
    world_metrics = {
        'timestep': timestep,
    }
    
    # Price and interest rate
    if hasattr(env, 'world') and hasattr(env.world, 'price'):
        prices = env.world.price
        if len(prices) > 0:
            world_metrics['price'] = float(prices[-1])
            # Calculate price inflation rate (compared to previous period)
            if len(prices) > 1:
                price_inflation = (prices[-1] - prices[-2]) / (prices[-2] + 1e-8)
                world_metrics['price_inflation_rate'] = float(price_inflation)
            else:
                world_metrics['price_inflation_rate'] = 0.0
        else:
            world_metrics['price'] = None
            world_metrics['price_inflation_rate'] = None
    
    if hasattr(env, 'world') and hasattr(env.world, 'interest_rate'):
        interest_rates = env.world.interest_rate
        if len(interest_rates) > 0:
            world_metrics['interest_rate'] = float(interest_rates[-1])
        else:
            world_metrics['interest_rate'] = None
    
    # Extract agent-level data for aggregate metrics
    wealths = []
    incomes = []
    consumptions = []
    work_decisions = []
    
    for idx in range(env.num_agents):
        agent = env.get_agent(str(idx))
        wealth = float(agent.inventory.get('Coin', 0))
        income = float(agent.income.get('Coin', 0))
        consumption = float(agent.consumption.get('Coin', 0))
        
        wealths.append(wealth)
        incomes.append(income)
        consumptions.append(consumption)

        # Prefer using the labor hours/job flag recorded by SimpleLabor; fall back to income proxy.
        endogenous_state = {}
        if hasattr(agent, "state"):
            endogenous_state = agent.state.get("endogenous", {}) if isinstance(agent.state, dict) else {}
        job_label = endogenous_state.get("job")
        labor_hours = endogenous_state.get("Labor")

        if job_label is not None:
            is_working = job_label != "Unemployment"
        elif labor_hours is not None:
            is_working = labor_hours >= 1
        else:
            is_working = income > 0

        work_decisions.append(1 if is_working else 0)
    
    # Aggregate metrics
    world_metrics['total_wealth'] = float(np.sum(wealths))
    world_metrics['total_income'] = float(np.sum(incomes))
    world_metrics['total_consumption'] = float(np.sum(consumptions))
    world_metrics['avg_wealth'] = float(np.mean(wealths)) if len(wealths) > 0 else 0.0
    world_metrics['avg_income'] = float(np.mean(incomes)) if len(incomes) > 0 else 0.0
    world_metrics['avg_consumption'] = float(np.mean(consumptions)) if len(consumptions) > 0 else 0.0
    
    # Inequality metrics (Gini coefficients)
    world_metrics['gini_wealth'] = float(_calculate_gini(wealths))
    world_metrics['gini_income'] = float(_calculate_gini(incomes))
    
    # Unemployment rate (approximate: agents with zero income)
    if len(work_decisions) > 0:
        unemployment_rate = 1.0 - (np.sum(work_decisions) / len(work_decisions))
        world_metrics['unemployment_rate'] = float(unemployment_rate)
    else:
        world_metrics['unemployment_rate'] = 0.0
    
    # Additional metrics from dense_log if available
    if hasattr(env, 'dense_log') and env.dense_log:
        # Try to extract more detailed metrics from dense_log
        if 'states' in env.dense_log and len(env.dense_log['states']) > 0:
            # Could extract more state information here
            pass
    
    return world_metrics

def baseline_actions(env, obs, work_decision=1, consumption_ratio=0.7, use_probabilistic_work=False, seed=None):
    """
    Baseline action function that returns constant actions for all agents.
    Used as counterfactual baseline for Shapley value calculation.
    
    Args:
        env: Environment instance
        obs: Current observations
        work_decision: Work decision parameter. Interpretation depends on use_probabilistic_work:
                      - If use_probabilistic_work=False (default): 
                        Should be 0 or 1. Values in (0,1) are clipped to nearest integer (>=0.5->1, <0.5->0).
                      - If use_probabilistic_work=True:
                        Can be in [0, 1] representing work probability. Each agent independently
                        decides to work with probability = work_decision.
        consumption_ratio: Fixed consumption ratio (0-1) for all agents. Default: 0.5
                          This will be converted to the consumption index (0-50)
        use_probabilistic_work: If True, work_decision is interpreted as probability (0-1).
                               If False (default), work_decision is deterministically converted to 0 or 1.
        seed: Random seed for reproducibility when use_probabilistic_work=True. 
              If None, uses current global random state.
    
    Returns:
        actions: Dictionary with format {str(idx): [l, c], 'p': [0]}
                 where l is work decision (0 or 1) and c is consumption index (0-50)
                 Note: c is an index (0-50), actual consumption ratio = c * 0.02
    """
    consumption_ratio = np.clip(consumption_ratio, 0.0, 1.0)
    
    actions = {}
    # Convert consumption ratio to consumption index (0-50)
    # consumption_ratio is in [0, 1], consumption index is in [0, 50]
    consumption_idx = int(np.clip(consumption_ratio / 0.02, 0, 50))
    
    # Determine work action for each agent
    if use_probabilistic_work and 0 < work_decision < 1:
        # Probabilistic: each agent decides independently based on work_decision probability
        work_probability = np.clip(work_decision, 0.0, 1.0)
        
        # Create a deterministic RNG for reproducibility if seed is provided
        if seed is not None:
            # Use timestep as part of seed to ensure different timesteps have different randomness
            timestep = getattr(env.world, 'timestep', 0)
            rng = np.random.RandomState(seed + timestep * 10000)
        else:
            rng = np.random
        
        for idx in range(env.num_agents):
            # Use agent index in seed for reproducibility
            if seed is not None:
                agent_rng = np.random.RandomState(seed + timestep * 10000 + idx * 100)
                agent_work = int(agent_rng.uniform() <= work_probability)
            else:
                agent_work = int(rng.uniform() <= work_probability)
            actions[str(idx)] = [agent_work, consumption_idx]
    else:
        # Deterministic: convert to 0 or 1
        # >= 0.5 -> 1 (work), < 0.5 -> 0 (don't work)
        work_decision_clipped = np.clip(work_decision, 0.0, 1.0)
        work_action = 1 if work_decision_clipped >= 0.5 else 0
        for idx in range(env.num_agents):
            actions[str(idx)] = [work_action, consumption_idx]
    
    actions['p'] = [0]
    return actions

def complex_actions(env, obs, beta=0.1, gamma=0.1, h=1):

    def consumption_len(price, wealth, curr_income, last_income, interest_rate):
        c = (price/(1e-8+wealth+curr_income))**beta
        c = min(max(c//0.02, 0), 50)
        return c
    def consumption_cats(price, wealth, curr_income, last_income, interest_rate):
        h1 = h / (1 + interest_rate)
        g = curr_income/(last_income+1e-8) - 1
        d = wealth/(last_income+1e-8) - h1
        c = 1 + (d - h1*g)/(1 + g + 1e-8)
        c = min(max(c*curr_income/(wealth+curr_income+1e-8)//0.02, 0), 50)
        return c
    def work_income_wealth(price, wealth, curr_income, last_income, expected_income, interest_rate):
        return int(np.random.uniform() < (curr_income/(wealth*(1 + interest_rate)+1e-8))**gamma)
    
    consumption_funs = [consumption_len, consumption_cats]
    work_funs = [work_income_wealth]

    actions = {}
    for idx in range(env.num_agents):
        this_agent = env.get_agent(str(idx))
        price = env.world.price[-1]
        wealth = this_agent.inventory['Coin']
        max_l = env._components_dict['SimpleLabor'].num_labor_hours
        max_income = max_l * this_agent.state['skill']
        last_income = this_agent.income['Coin']
        expected_income = max_l * this_agent.state['expected skill']
        interest_rate = env.world.interest_rate[-1]
        if 'consumption_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['consumption_fun_idx'] = np.random.choice(range(len(consumption_funs)))
        if 'work_fun_idx' not in this_agent.endogenous:
            this_agent.endogenous['work_fun_idx'] = np.random.choice(range(len(work_funs)))
        work_fun = work_funs[this_agent.endogenous['work_fun_idx']]
        l = work_fun(price, wealth, max_income, last_income, expected_income, interest_rate)
        curr_income = l * max_income
        consumption_fun = consumption_funs[this_agent.endogenous['consumption_fun_idx']]
        c = consumption_fun(price, wealth, curr_income, last_income, interest_rate)
        actions[str(idx)] = [l, c]
    actions['p'] = [0]
    return actions

def main(policy_model='gpt', num_agents=100, episode_length=240, dialog_len=3,
         beta=0.1, gamma=0.1, h=1, max_price_inflation=0.1, max_wage_inflation=0.05,
         resume_step=0, resume_dir=None, baseline_work=1, baseline_consumption=0.7, seed=None):
    resume_step = int(resume_step or 0)
    if resume_step < 0:
        raise ValueError("resume_step must be >= 0")

    # Load config (config overrides function defaults)
    if seed is None:
        seed = simulation_config.get('seed')
    if 'policy_model' in simulation_config:
        policy_model = simulation_config['policy_model']
    if 'num_agents' in simulation_config:
        num_agents = simulation_config['num_agents']
    if 'episode_length' in simulation_config:
        episode_length = simulation_config['episode_length']
    
    baseline_config = simulation_config.get('baseline', {})
    if 'work' in baseline_config:
        baseline_work = baseline_config['work']
    if 'consumption' in baseline_config:
        baseline_consumption = baseline_config['consumption']
    
    print(f"Simulation: {policy_model}, agents={num_agents}, length={episode_length}, seed={seed}")
    
    if policy_model == 'baseline':
        use_probabilistic = 0 < baseline_work < 1
        print(f"  Baseline: work={baseline_work} ({'probabilistic' if use_probabilistic else 'deterministic'}), consumption={baseline_consumption}")
    
    env_config['n_agents'] = num_agents
    env_config['episode_length'] = episode_length
    
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)
        env_config['seed'] = seed

    dialog_queue = None
    dialog4ref_queue = None
    gpt_error = 0
    total_cost = 0.0

    if policy_model == 'gpt':
        env_config['flatten_masks'] = False
        env_config['flatten_observations'] = False
        env_config['components'][0]['SimpleLabor']['scale_obs'] = False
        env_config['components'][1]['PeriodicBracketTax']['scale_obs'] = False
        env_config['components'][3]['SimpleSaving']['scale_obs'] = False
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        action_fn = gpt_actions
    elif policy_model == 'complex':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        action_fn = None
    elif policy_model == 'baseline':
        env_config['components'][2]['SimpleConsumption']['max_price_inflation'] = max_price_inflation
        env_config['components'][2]['SimpleConsumption']['max_wage_inflation'] = max_wage_inflation
        action_fn = None
    else:
        raise ValueError(f"Unsupported policy_model '{policy_model}'")

    default_run_base = policy_model
    if policy_model == 'complex':
        default_run_base = f'{policy_model}-{beta}-{gamma}-{h}-{max_price_inflation}-{max_wage_inflation}'
    if policy_model == 'gpt':
        default_run_base = f'{policy_model}-{dialog_len}-noperception-reflection-1'
    if policy_model == 'baseline':
        default_run_base = f'{policy_model}-work{baseline_work}-cons{baseline_consumption}'

    default_run_name = f'{default_run_base}-{num_agents}agents-{episode_length}months'
    run_label = _configured_run_name or default_run_name

    if resume_dir:
        data_dir = os.path.abspath(resume_dir)
        run_label = os.path.basename(os.path.normpath(data_dir))
    else:
        data_dir = os.path.join(save_path, 'datas', run_label)
    # figs_dir 已移除，不再创建figs文件夹

    os.makedirs(data_dir, exist_ok=True)

    actions_dir = os.path.join(data_dir, 'actions')
    obs_dir = os.path.join(data_dir, 'obs')
    env_dir = os.path.join(data_dir, 'env')
    dialog_pkl_dir = os.path.join(data_dir, 'dialog_pickles')
    dialog4ref_pkl_dir = os.path.join(data_dir, 'dialog4ref_pickles')
    dense_log_dir = os.path.join(data_dir, 'dense_logs')
    dialogs_dir = os.path.join(data_dir, 'dialogs')
    
    # JSON/CSV export directories
    actions_json_dir = os.path.join(data_dir, 'actions_json')
    metrics_csv_dir = os.path.join(data_dir, 'metrics_csv')

    os.makedirs(actions_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(dialog_pkl_dir, exist_ok=True)
    os.makedirs(dialog4ref_pkl_dir, exist_ok=True)
    os.makedirs(dense_log_dir, exist_ok=True)
    os.makedirs(dialogs_dir, exist_ok=True)
    os.makedirs(actions_json_dir, exist_ok=True)
    os.makedirs(metrics_csv_dir, exist_ok=True)
    
    # Initialize CSV files for metrics (write headers on first step)
    agent_metrics_csv_path = os.path.join(metrics_csv_dir, 'agent_metrics.csv')
    world_metrics_csv_path = os.path.join(metrics_csv_dir, 'world_metrics.csv')
    agent_metrics_file_exists = os.path.exists(agent_metrics_csv_path)
    world_metrics_file_exists = os.path.exists(world_metrics_csv_path)

    env = None
    obs = None
    start_episode = 0

    if resume_step > 0:
        env, obs, dialog_queue, dialog4ref_queue, resume_state = _load_resume_state(
            data_dir, resume_step, policy_model, dialog_len
        )
        start_episode = resume_step
        gpt_error = resume_state.get('gpt_error', gpt_error)
        total_cost = resume_state.get('total_cost', total_cost)
        print(f"Resuming from step {resume_step} using data directory: {data_dir}")
    else:
        import sys
        sys.stdout.flush()
        env = foundation.make_env_instance(**env_config)
        sys.stdout.flush()
        try:
            obs = env.reset()
        except Exception as e:
            raise

    if env is None or obs is None:
        raise RuntimeError("Failed to initialise environment state.")

    if env.num_agents != num_agents:
        raise ValueError(f"Environment reports {env.num_agents} agents, but num_agents={num_agents}.")

    if resume_step >= env.episode_length:
        print(f"resume_step {resume_step} >= episode length {env.episode_length}; nothing to run.")
        return

    print("Initializing dialog queues...")
    sys.stdout.flush()
    if policy_model == 'gpt':
        if dialog_queue is None:
            dialog_queue = [deque(maxlen=dialog_len) for _ in range(env.num_agents)]
        else:
            dialog_queue = [deque(list(d), maxlen=dialog_len) for d in dialog_queue]
        if dialog4ref_queue is None:
            dialog4ref_queue = [deque(maxlen=7) for _ in range(env.num_agents)]
        else:
            dialog4ref_queue = [deque(list(d), maxlen=7) for d in dialog4ref_queue]
        if len(dialog_queue) != env.num_agents or len(dialog4ref_queue) != env.num_agents:
            raise ValueError("Dialog queues do not align with number of agents.")
        print(f"Dialog queues initialized: {len(dialog_queue)} queues")
    else:
        dialog_queue = []
        dialog4ref_queue = []
    sys.stdout.flush()

    t = time()
    run_state_path = os.path.join(data_dir, 'run_state.pkl')
    print(f"Starting simulation loop: {start_episode} to {env.episode_length}")
    sys.stdout.flush()

    for epi in range(start_episode, env.episode_length):
        if policy_model == 'gpt':
            actions, gpt_error, total_cost = action_fn(
                env,
                obs,
                dialog_queue,
                dialog4ref_queue,
                dialogs_dir,
                gpt_error,
                total_cost,
            )
        elif policy_model == 'complex':
            actions = complex_actions(env, obs, beta=beta, gamma=gamma, h=h)
        elif policy_model == 'baseline':
            # If baseline_work is in (0, 1), use probabilistic work (0.8 means 80% probability to work)
            # If baseline_work is exactly 0 or 1, use deterministic (always 0 or always 1)
            use_probabilistic = 0 < baseline_work < 1
            actions = baseline_actions(env, obs, work_decision=baseline_work, 
                                     consumption_ratio=baseline_consumption,
                                     use_probabilistic_work=use_probabilistic,
                                     seed=seed)
        else:
            actions = {}

        obs, rew, done, info = env.step(actions)

        if (epi + 1) % 3 == 0:
            print(f'step {epi+1} done, cost {time()-t:.1f}s')
            if policy_model == 'gpt':
                print(f'#errors: {gpt_error}, cost ${total_cost:.1f} so far')
            t = time()

        if (epi + 1) % 1 == 0 or epi + 1 == env.episode_length:
            # Save pickle files (original format)
            with open(os.path.join(actions_dir, f'actions_{epi+1}.pkl'), 'wb') as f:
                pkl.dump(actions, f)
            with open(os.path.join(obs_dir, f'obs_{epi+1}.pkl'), 'wb') as f:
                pkl.dump(obs, f)
            with open(os.path.join(env_dir, f'env_{epi+1}.pkl'), 'wb') as f:
                pkl.dump(env, f)
            if policy_model == 'gpt':
                with open(os.path.join(dialog_pkl_dir, f'dialog_{epi+1}.pkl'), 'wb') as f:
                    pkl.dump(dialog_queue, f)
                with open(os.path.join(dialog4ref_pkl_dir, f'dialog4ref_{epi+1}.pkl'), 'wb') as f:
                    pkl.dump(dialog4ref_queue, f)
            with open(os.path.join(dense_log_dir, f'dense_log_{epi+1}.pkl'), 'wb') as f:
                pkl.dump(env.dense_log, f)
            
            # Save JSON/CSV files (for easier visualization and analysis)
            # 1. Save actions as JSON
            actions_json = _convert_to_json_serializable(actions)
            with open(os.path.join(actions_json_dir, f'actions_{epi+1}.json'), 'w') as f:
                json.dump(actions_json, f, indent=2)
            
            # 2. Extract and save agent metrics as CSV
            agent_metrics = _extract_agent_metrics(env, epi + 1)
            if agent_metrics:
                fieldnames = list(agent_metrics[0].keys())
                with open(agent_metrics_csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not agent_metrics_file_exists:
                        writer.writeheader()
                        agent_metrics_file_exists = True
                    writer.writerows(agent_metrics)
            
            # 3. Extract and save world metrics as CSV (with comprehensive macro indicators)
            world_metrics = _extract_world_metrics(env, epi + 1)
            if world_metrics:
                fieldnames = list(world_metrics.keys())
                with open(world_metrics_csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not world_metrics_file_exists:
                        writer.writeheader()
                        world_metrics_file_exists = True
                    writer.writerow(world_metrics)

            run_state = {
                "step": epi + 1,
                "policy_model": policy_model,
                "num_agents": env.num_agents,
                "dialog_len": dialog_len if policy_model == 'gpt' else None,
                "gpt_error": gpt_error,
                "total_cost": total_cost,
            }
            with open(run_state_path, 'wb') as f:
                pkl.dump(run_state, f)

    with open(os.path.join(dense_log_dir, 'dense_log.pkl'), 'wb') as f:
        pkl.dump(env.dense_log, f)
    
    # Create a consolidated actions JSON file for all timesteps (useful for Shapley value calculation)
    print("Creating consolidated actions JSON file...")
    all_actions = {}
    for step in range(1, env.episode_length + 1):
        actions_json_path = os.path.join(actions_json_dir, f'actions_{step}.json')
        if os.path.exists(actions_json_path):
            with open(actions_json_path, 'r') as f:
                all_actions[f'step_{step}'] = json.load(f)
    
    consolidated_actions_path = os.path.join(actions_json_dir, 'all_actions.json')
    with open(consolidated_actions_path, 'w') as f:
        json.dump(all_actions, f, indent=2)
    print(f"Consolidated actions saved to: {consolidated_actions_path}")

    if policy_model == 'gpt':
        print(f'#gpt errors: {gpt_error}')

if __name__ == "__main__":
    fire.Fire(main)
