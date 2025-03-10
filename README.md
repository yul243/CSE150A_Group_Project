# CSE150A_Group_Project

## Dataset
We are using the following dataset for our project:

- [Blackjack Hands Dataset](https://www.kaggle.com/datasets/dennisho/blackjack-hands)

The dataset is too large to be uploaded to GitHub.  
You need to download it from the link. (50,000,000 test size.)

## PEAS description
*   **Performance Measure:**
    *   Maximize long-term winnings (or minimize losses) in Blackjack, predict the action with the highest likelihood.
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

*   **Utility-Based:** The ultimate goal is to maximize the *utility* of the agent, which is measured as long-term winnings (or minimize losses). It aims to choose actions that lead to the highest expected utility. However, the actual "utility" is learned through the probability of winning, given the action.
*   **Probabilistic Agent:** The agent explicitly reasons about probabilities (the CPTs in the Bayesian Network) to make decisions.


## Probabilistic Modeling and the Agent's Setup




## Training the Model



## Evaluating the Model


### Conclusion of the Reinforcement Learning Model



### Potential Improvements
