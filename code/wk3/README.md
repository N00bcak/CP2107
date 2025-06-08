## Setup

See [installation.md](installation.md). It's worth going through this again since some dependencies have changed since homework 1. You also need to make sure to run `pip install -e .` in the hw2 folder.

## Running on Google Cloud
Starting with HW2, we will be providing some infrastructure to run experiments on Google Cloud compute. There are some very important caveats:

- **Do not leave your instance running.** The provided infrastructure tries to prevent this, but it will still be easy to accidentally leave your instance running and burn through all of your credits. You are responsible for making sure you use your credits wisely.
- **Only use this for big hyperparameter sweeps.** Definitely don't use Google Cloud for debugging; only launch a job once you are 100% sure your code works. Even then, single jobs will probably run faster on your local machine (yes, even if you don't have a GPU). The only reason to use Google Cloud is if you want to run multiple jobs in parallel.

For more instructions, see [google_cloud/README.md](google_cloud/README.md).

## Complete the code

There are TODOs in these files:

- `cs285/scripts/run_hw2.py`
- `cs285/agents/pg_agent.py`
- `cs285/networks/policies.py`
- `cs285/networks/critics.py`
- `cs285/infrastructure/utils.py`

See the [Assignment PDF](hw2.pdf) for more info.

### Changes
The changes are enumerated below, broken down by component.
Note that we adopt the convention of using [code/wk/cs285](../../code/wk3/cs285)
as the root directory for the code, and point to specific lines by using the regex
`^([^:]+):L(\d+)(?:-\d+)$`

## Trajectory Collection

| Sub-Component | Changes | Location |
|---------------|---------|----------|
| Action Sampling | Now calls the policy network to sample actions | [infrastructure/utils.py:L32-36](../../code/wk3/cs285/infrastructure/utils.py) |
| Environment Interaction | We now perform environment interaction steps | [infrastructure/utils.py:L38-42](../../code/wk3/cs285/infrastructure/utils.py) |
| Environment Doneness | We now check if the environment is done | [infrastructure/utils.py:L44-50](../../code/wk3/cs285/infrastructure/utils.py) |

## State-Value Estimation

| Sub-Component | Changes | Location |
|---------------|---------|----------|
| Q-Value Calculation | We define at a high-level how to calculate state-values (Q-values) | [agents/pg_agent.py:L112-130](../../code/wk3/cs285/agents/pg_agent.py) |
| Episodic Return Calculation | We can estimate Q-values by calculating the episodic return | [agents/pg_agent.py:L197-210](../../code/wk3/cs285/agents/pg_agent.py) |
| Reward-to-Go Calculation | We can estimate Q-values by calculating the reward-to-go | [agents/pg_agent.py:L219-227](../../code/wk3/cs285/agents/pg_agent.py) |

## Advantage Estimation

| Sub-Component | Changes | Location |
|---------------|---------|----------|
| Advantage Calculation | We define at a high-level how to calculate advantages | [agents/pg_agent.py:L146-L151](../../code/wk3/cs285/agents/pg_agent.py) |
| Q-Value As Advantage | We used the Q-values as advantages[^1] | [agents/pg_agent.py:L146-151](../../code/wk3/cs285/agents/pg_agent.py) |
| Advantage Normalization | We allow normalizing the advantages to have zero mean and unit variance | [agents/pg_agent.py:L180-184](../../code/wk3/cs285/agents/pg_agent.py) |

## Policy

| Sub-Component | Changes | Location |
|---------------|---------|----------|
| Forward Pass & Action Sampling | We implemented the forward pass and action sampling of the policy network[^2] | [networks/policies.py:L61-64](../../code/wk3/cs285/networks/policies.py)<br>[networks/policies.py:L74-94](../../code/wk3/cs285/networks/policies.py) |

## Update Loop

| Sub-Component | Changes | Location |
|---------------|---------|----------|
| Loss Calculation | We implemented the loss calculation for the policy gradient algorithm | [agents/pg_agent.py:L118-144](../../code/wk3/cs285/agents/pg_agent.py) |
| Backpropagation | We implemented the backpropagation and optimizer steps for the policy gradient algorithm | [agents/pg_agent.py:L118-144](../../code/wk3/cs285/agents/pg_agent.py) |

## Others
| Sub-Component | Changes | Location |
|---------------|---------|----------|
| Data Logging | We changed the naming convention for data logging | [scripts/run_hw2.py:L184-193](../../code/wk3/cs285/scripts/run_hw2.py) |


[^1]: In practice we would use a neural network baseline, and subtract the baselines from the rewards-to-go.

[^2]: While I would personally have deferred the sampling to `get_action` so I could get a more convenient backpropagation, the template implied I should have sampled in the forward pass.