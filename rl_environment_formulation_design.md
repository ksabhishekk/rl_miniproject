# Architectural Design & Reinforcement Learning Formulation
## Comparative Analysis of RL Algorithms for Shilling Attack Mitigation in Recommendation Systems

---

## 1. Executive Summary & Design Principles

This project investigates reinforcement learning (RL) policies to dynamically defend Collaborative Filtering (CF) recommendation engines against shilling (profile injection) attacks using the **MovieLens 100K** benchmark.

### Core Design Philosophy: Avoiding the "Disguised Classifier" Trap
A frequent design error in applying RL to security is reducing the problem to per-sample binary classification ($y \in \{0, 1\}$ with binary rewards $+1/-1$). That is not true reinforcement learning—it fails to capture sequential effects, system-level feedback, and trade-offs.

To make this a genuine RL formulation:
1. **System-State Feedback**: The agent does not simply inspect a user profile in isolation. It observes the profile **in relation to the evolving recommendation engine state** (target item predicted rank shift, neighbor graph centrality, catalog distortion).
2. **Action Impact on Future States**: When the agent modifies a profile's influence ($w_v \in [0, 1]$), it directly alters the user-user similarity graph and predicted rating distributions for subsequent queries.
3. **Reward Rooted in System Utility**: Rewards are derived from **Recommendation Fidelity** (Precision@K / RMSE on genuine validation users) and **Attack Suppression** (suppression of target item rank inflation), rather than purely checking ground-truth profile labels.

---

## 2. Recommendation Engine Architecture

### Choice: User-Based Collaborative Filtering (UBCF) with Dynamic Profile Weighting
User-Based CF is mathematically transparent and directly responds to profile-level actions:

$$\hat{r}_{u, i} = \bar{r}_u + \frac{\sum_{v \in \mathcal{N}_k(u)} w_v \cdot \text{sim}(u, v) \cdot (r_{v, i} - \bar{r}_v)}{\sum_{v \in \mathcal{N}_k(u)} w_v \cdot |\text{sim}(u, v)| + \epsilon}$$

Where:
- $\text{sim}(u, v)$ is the Pearson Correlation Coefficient or Adjusted Cosine Similarity between user $u$ and profile $v$.
- $\mathcal{N}_k(u)$ is the top-$k$ nearest neighbors of user $u$.
- $w_v \in \{1.0, 0.25, 0.0\}$ is the **defense weight assigned by the RL agent** to profile $v$.

---

## 3. Shilling Attack Profiles (MovieLens 100K)

Let $I$ be the set of all items ($|I| = 1682$), $I_T$ be the target items (low-to-medium popularity items being artificially promoted), $I_S$ be selected filler items, and $I_\emptyset$ be unrated items. Attack profiles assign maximum rating $r_{\max} = 5.0$ to $i \in I_T$.

We simulate three classic attack models:

| Attack Model | Target Item ($I_T$) | Selected Items ($I_S$) | Unrated Items ($I_\emptyset$) | Shilling Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Random Attack** | $r = 5.0$ | Random sample of items rated uniformly $r \sim \mathcal{U}(1, 5)$ | Null | High variance, low similarity to genuine user patterns. |
| **Average Attack** | $r = 5.0$ | Random items rated close to item sample mean $\mathcal{N}(\bar{r}_i, \sigma_i^2)$ | Null | Mimics average consensus; harder to detect via rating deviation alone. |
| **Bandwagon Attack** | $r = 5.0$ | Popular blockbuster items rated $r = 5.0$ + small random filler | Null | High similarity to large genuine user clusters; strong push attack power. |

---

## 4. Formal Gymnasium Environment Specification

### 4.1 Episode & Step Dynamics
- **Episode Duration ($T$)**: An audit window of $T = 100$ profiles (e.g., $70\%$ genuine, $30\%$ malicious injected under chosen attack strategy).
- **Step $t$**: At step $t$, profile $P_t$ arrives or is reviewed from the audit queue.
- **Action Execution**: The agent selects an action $a_t \in \{0, 1, 2\}$, mapping to profile weight $w_t$.
- **State Transition**: The recommendation matrix incorporates $w_t$, metrics update, and the next candidate profile $P_{t+1}$ and updated system state are presented.

### 4.2 State Space $\mathcal{S}$

Each state $s_t$ is a 5-dimensional feature vector combining **Profile Statistical Indicators** and **System Impact Indicators**:

$$s_t = [f_{\text{RDMA}}, f_{\text{DegSim}}, f_{\text{ActRatio}}, f_{\text{TargetDist}}, f_{\text{SystemRankShift}}]$$

1. **Rating Deviation from Mean Agreement (RDMA)**:
   $$\text{RDMA}(v) = \frac{1}{|I_v|} \sum_{i \in I_v} \frac{|r_{v, i} - \bar{r}_i|}{N_i}$$
   Measures deviation from global item consensus, normalized by item popularity $N_i$.
2. **Degree of Similarity with Popular Items (DegSim)**:
   $$\text{DegSim}(v) = \frac{1}{|I_{\text{pop}}|} \sum_{i \in I_v \cap I_{\text{pop}}} \frac{|r_{v, i} - \bar{r}_i|}{\max(r) - \min(r)}$$
   Detects bandwagon effects where a profile matches popular item clusters.
3. **Activity Ratio ($f_{\text{ActRatio}}$)**:
   $$f_{\text{ActRatio}}(v) = \min\left(1.0, \frac{|I_v|}{\text{mean}(|I_{\text{all}}|) \times 2}\right)$$
   Normalized profile length/activity volume.
4. **Target Affinity Indicator ($f_{\text{TargetDist}}$)**:
   Fraction of monitored target items rated with $r = 5.0$ by profile $v$.
5. **System Target Inflation Metric ($f_{\text{SystemRankShift}}$)**:
   Current normalized shift in target item predicted score/ranking across the genuine user base.

#### Discretization for Tabular Q-Learning:
To enable fair comparison across Tabular Q-Learning, DQN, and PPO:
- **DQN and PPO**: Receive the continuous vector $s_t \in [0, 1]^5$.
- **Q-Learning**: Each feature is discretized into $B = 3$ bins (Low, Medium, High) using empirical quantile binning:
  $$|\mathcal{S}_{\text{discrete}}| = 3^5 = 243 \text{ discrete states}$$
  This is compact enough for tabular convergence within $10^4 - 10^5$ steps, while preserving the exact same underlying dynamics.

### 4.3 Action Space $\mathcal{A}$

A discrete action space $|\mathcal{A}| = 3$:
- **Action 0 (`ACCEPT`, $w=1.0$)**: Trust profile completely. Full inclusion in collaborative filtering.
- **Action 1 (`DOWNWEIGHT`, $w=0.25$)**: Soft mitigation. Profile included with dampened influence.
- **Action 2 (`BLOCK`, $w=0.0$)**: Quarantine/Zero trust. Profile excluded from similarity calculations.

### 4.4 Reward Function $\mathcal{R}(s_t, a_t, s_{t+1})$

The reward directly balances **Attack Impact Suppression** against **Genuine Recommendation Quality Preservation**:

$$R_t = R_{\text{Mitigation}} + R_{\text{Quality}} - C_{\text{Intervention}}$$

Where:
1. **Attack Mitigation Reward ($R_{\text{Mitigation}}$)**:
   - When the profile is an attack profile:
     - `BLOCK` ($a=2$): $+2.0$ (successfully eliminated attack influence)
     - `DOWNWEIGHT` ($a=1$): $+0.8$ (partially mitigated)
     - `ACCEPT` ($a=0$): $-2.5$ (attack successfully penetrated system)
2. **Quality Preservation Reward ($R_{\text{Quality}}$)**:
   - When the profile is genuine:
     - `ACCEPT` ($a=0$): $+1.0$ (preserved genuine rating data)
     - `DOWNWEIGHT` ($a=1$): $-0.5$ (degraded genuine data quality)
     - `BLOCK` ($a=2$): $-2.0$ (false positive penalty; harms catalog coverage)
3. **System Ranking Delta Component**:
   $$\Delta \text{RankInflation} = \text{Rank}_{\text{target}}(t) - \text{Rank}_{\text{target}}(t-1)$$
   A positive bonus $+0.5$ if target item ranking drops towards true baseline; penalty $-0.5$ if target item moves higher in recommendation lists.

---

## 5. Comparative Algorithms Formulation

```
                                  +------------------------------+
                                  |   Gymnasium Environment      |
                                  |   - MovieLens 100K Matrix    |
                                  |   - Target Items & Attacks   |
                                  +--------------+---------------+
                                                 | State: s_t (5-dim)
                   +-----------------------------+-----------------------------+
                   |                             |                             |
                   v                             v                             v
      +------------------------+    +------------------------+   +------------------------+
      |      Q-Learning        |    |          DQN           |   |          PPO           |
      |   (Tabular, NumPy)     |    |  (Stable-Baselines3)   |   |  (Stable-Baselines3)   |
      +------------------------+    +------------------------+   +------------------------+
      | State: Discretized 243 |    | State: Continuous R^5  |   | State: Continuous R^5  |
      | Update: Bellman Eq     |    | Q-Network + ReplayBuf  |   | Actor-Critic + Clip    |
      +------------------------+    +------------------------+   +------------------------+
                   |                             |                             |
                   +-----------------------------+-----------------------------+
                                                 | Action: a_t in {0, 1, 2}
                                                 v
                                  +------------------------------+
                                  | Defense Action Applied       |
                                  | w_t in {1.0, 0.25, 0.0}      |
                                  | CF Recommendations Evaluated |
                                  +------------------------------+
```

### 1. Tabular Q-Learning (NumPy)
- **State Space**: Discretized index $i_s \in \{0, \dots, 242\}$.
- **Update Rule**:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
- **Exploration**: $\epsilon$-greedy with decay schedule $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 \cdot d^t)$.

### 2. Deep Q-Network (DQN via Stable-Baselines3)
- **State Space**: Continuous $s \in [0, 1]^5$.
- **Architecture**: Multi-Layer Perceptron (MLP) with 2 hidden layers (64, 64), ReLU activations.
- **Key Features**: Experience Replay buffer ($10^4$ transitions), target network with periodic sync, Huber loss.

### 3. Proximal Policy Optimization (PPO via Stable-Baselines3)
- **State Space**: Continuous $s \in [0, 1]^5$.
- **Architecture**: Actor-Critic network (Shared/Separate MLP $[64, 64]$).
- **Key Features**: Clipped surrogate objective ($\epsilon_{\text{clip}} = 0.2$), Generalized Advantage Estimation (GAE).

---

## 6. Evaluation Framework & Baseline Comparison

### 6.1 Baselines for Benchmarking
1. **Undefended CF**: Baseline collaborative filtering with all ratings accepted ($w=1.0$).
2. **Heuristic Rule-Based Filter**: Statistical threshold detector (e.g. threshold on RDMA $> \theta$).
3. **Random Agent**: Uniform random action selection.

### 6.2 Key Performance Indicators (KPIs)

1. **Attack Mitigation Metrics**:
   - **Prediction Shift (PS)**: Average change in target item predicted rating before vs after attack with defense:
     $$\text{PS}(I_T) = \frac{1}{|U_{\text{eval}}|} \sum_{u \in U_{\text{eval}}} (\hat{r}_{\text{defended}}(u, i) - \hat{r}_{\text{baseline}}(u, i))$$
   - **Hit Ratio @ 10 on Target Items**: Frequency with which target item appears in Top-10 recommendations for genuine users.
2. **Recommendation Quality (Utility Preservation)**:
   - **Root Mean Squared Error (RMSE)** on genuine test ratings.
   - **Precision@10 and Recall@10** on genuine user recommendations.
3. **RL Operational Metrics**:
   - Cumulative Episode Reward across training steps.
   - Sample efficiency / Convergence speed.
   - Wall-clock training and inference latency.
