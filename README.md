# Classification with Decision Tree and Random Forest

## Project Description

This project is a from-scratch implementation of **Decision Tree** and **Random Forest** classifiers in Python. These models are used for a binary classification task: predicting which passengers survived the Titanic disaster. The analysis is based on the popular `titanic.csv` dataset.

---

## Features

-   **Decision Tree Implementation:**
    -   Tree construction based on recursive data partitioning.
    -   Selection of the best split using the **Gini impurity** criterion to maximize information gain.
    -   Ability to control the maximum depth of the tree to prevent overfitting.
-   **Random Forest Implementation:**
    -   Builds multiple decision trees to create a more stable and accurate model.
    -   Uses the **bagging** technique (sampling with replacement) to create diverse data subsets for each tree.
    -   Aggregates results from individual trees through majority voting.
-   **Data Preprocessing:**
    -   Loading and cleaning data from the `titanic.csv` file.
    -   Feature selection and conversion of categorical data to numerical values.
    -   Splitting the dataset into training and testing sets.

---

## Project Structure

-   **`main.py`**: The main file that initializes, trains, and evaluates the Decision Tree and Random Forest models.
-   **`decision_tree.py`**: Contains the `DecisionTree` class, which builds a single decision tree.
-   **`random_forest.py`**: Contains the `RandomForest` class, which manages the creation and prediction process using multiple trees.
-   **`node.py`**: Defines the `Node` class, representing a single node in the tree. It is responsible for finding the best split and recursively building the tree.
-   **`load_data.py`**: Responsible for loading data from `titanic.csv`, preprocessing it, and splitting it into training and test sets.
-   **`titanic.csv`**: The dataset containing information about the Titanic passengers.
-   **`requirements.txt`**: A file with a list of libraries required to run the project.

---

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Skan404/Classification-Random-Forest.git
    cd Classification-Random-Forest
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To run the project and see the performance of both classifiers, execute the following command in the main project directory:

```bash
python main.py
