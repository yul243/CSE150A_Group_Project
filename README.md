# CSE150A_Group_Project

## Dataset
We are using the following dataset for our project:

- [Blackjack Hands Dataset](https://www.kaggle.com/datasets/dennisho/blackjack-hands)

The dataset is too large to be uploaded to GitHub.  
You need to download it from the link.

## PEAS description
*   **Performance Measure:**
    *   Maximize long-term winnings (or minimize losses) in Blackjack. More specifically, in this simplified version, accurately predict win/loss/push based on observed variables. Also, predict the action with the highest likelihood.
*   **Environment:**
    *   Single-player Blackjack game against a dealer.
    *   The "world" consists of:
        *   A finite deck of cards (typically multiple decks shuffled together).
        *   Blackjack rules (dealer hits on soft 17, etc.).
        *   The state of the game: dealer's up card, player's hand (or hand value), actions taken, and the game outcome.
        *   The agent operates in a *static* environment as the rules don't changes.
        *   The environment is *partially observable* as the agent doesn't see the dealer's hole card or the entire deck.
        *   The environment is *stochastic* as the cards dealt are random.
        *   The environment is *sequential* as past actions affect future states.
*   **Actuators:**
    *   Actions: Hit (H), Stand (S), Double Down (D), Split (P), Surrender (R), Insurance (I), No Insurance (N).
*   **Sensors:**
    *   Dealer's up card (`dealer_up`).
    *   Player's final hand value (`player_final_value`).
    *   Prior actions of player's action.
    *   Game outcome (win, loss, push).

## Type of Agent
This Blackjack AI agent is primarily a **Utility-Based Agent**, the reasons are:

*   **Utility-Based:** The ultimate goal is to maximize the *utility* of the agent, which is measured as long-term winnings (or minimize losses). It aims to choose actions that lead to the highest expected utility. Although the actual "utility" is learned through the probability of win given the action.
*   **Probabilistic Agent:** The agent explicitly reasons about probabilities (the CPTs in the Bayesian Network) to make decisions.


## Probabilistic Modeling and the Agent's Setup

*   **Bayesian Network Structure:**
    *   A directed acyclic graph (DAG) where nodes represent variables (e.g., `dealer_up`, `player_final_value`, `win`, `actions`) and edges represent probabilistic dependencies between these variables.
    *   The structure of the network encodes assumptions about conditional independence. For example, the agent assumes that `player_final_value` is directly influenced by `dealer_up`.
*   **Conditional Probability Tables (CPTs):**
    *   Each node in the network has a CPT that quantifies the probability of each possible value of the variable given the values of its parent nodes.
    *   The CPTs are learned from the dataset, effectively capturing the statistical relationships between the variables.
*   **Inference:**
    *   When the agent observes the environment (e.g., sees the `dealer_up` card), it uses *probabilistic inference* to update its beliefs about the other variables in the network, including the probability of winning. This is done using the `predict_proba` function.
    *   The agent then chooses the action that maximizes its expected utility (which is related to the probability of winning). 


## Training the Model

The training process for this Blackjack AI agent involves the following steps:

1.  **Data Acquisition and Preprocessing:**
    *   The agent uses `kagglehub` to download the Blackjack dataset (used hardcode dataset path at last to improve convenience).
    *   The dataset is preprocessed using `pandas`. This involves:
        *   Selecting relevant features (e.g., `dealer_up`, `player_final_value`, `actions_taken`, `win`).
        *   Discretizing continuous features like `player_final_value` and `dealer_up` into categorical bins.
        *   Converting the `actions_taken` column into one-hot encoded columns representing individual actions.
        *   Converting the `win` column into categorical data ('win', 'lose', 'push').
        *   Removing rows with missing values (`NaN`).

2.  **Bayesian Network Structure Definition:**
    *   The structure of the Bayesian Network is defined manually by creating `BayesianNode` objects and specifying parent-child relationships. This encodes the prior knowledge about the dependencies between the variables.  For example, we define that `player_final_value` is dependent on `dealer_up`.

3.  **CPT Learning:**
    *   The `compute_cpt()` method is called for each `BayesianNode`. This calculates the Conditional Probability Table (CPT) based on the preprocessed data.
    *   For nodes without parents, the CPT is simply the marginal probability distribution of that variable.
    *   For nodes with parents, the CPT is calculated by grouping the data by the parent variables and the node's variable and calculating the conditional probabilities.

## Evaluating the Model
1.  **Data Splitting:**
    *   Divide the dataset into training and testing sets (e.g., 80% for training, 20% for testing).  This ensures that the model is evaluated on data it hasn't seen during training. This needs to be added.

2.  **Prediction:**
    *   For each data point in the testing set, use the `predict_win_probability()` function to predict the probability of winning given the observed `dealer_up` and `player_final_value`.

3.  **Evaluation Metrics:**
    *   **Accuracy:**  The percentage of correctly classified win/loss/push outcomes.

### Conclusion of the First Model

*   **Basic Understanding of Relationships:** The model can capture basic statistical relationships between `dealer_up`, `player_final_value`, and `win`.  For example, it will likely learn that a high dealer up card reduces the probability of winning.
*   **Limited Generalization:** The model likely has poor generalization performance, especially for rare or unseen combinations of conditions. This is due to the simple learning approach and the lack of smoothing. (This might be the problem we are currently facing.)
*   **Overfitting Potential:** Without a separate test set, it's impossible to assess whether the model is overfitting the training data.

The first model serves primarily as a proof-of-concept and a starting point, we are still facing some issues with the out-of-context data, and our evaluation part currently has relatively poor performance as lots of the test data are unseen in the training process so that it cannot generate probability dependencies.

### Potential Improvements

Here are several ways to improve the model:

1.  **Implement Proper Training and Evaluation:**  The most important improvement is to split the data into training and testing sets and to use appropriate evaluation metrics to assess performance.

2.  **Smoothing:** Add smoothing techniques (e.g., Laplace smoothing) to the CPTs to avoid zero probabilities and improve generalization.

3.  **Feature Engineering:**
    *   Create more informative features. For example:
        *   Combine `dealer_up` and `player_initial_hand` into a single feature that represents the "initial situation".
        *   Include the running count and cards remaining in the shoe as features (if available).
        *   Consider the player's initial hand *before* discretization.

4.  **More Complex Network Structure:**
    *   Experiment with adding more edges to represent more complex relationships. For example:
        *   `run_count` -> `actions`
        *   `cards_remaining` -> `actions`
        *   Interactions between variables (e.g., `dealer_up` + `player_initial_hand` -> `actions`)