# Knowledge Graph Extraction and Visualization

This project extracts key concepts and their relationships from a given text document and visualizes them as a network graph. It leverages a large language model to identify terms and their relationships, calculates contextual proximity between terms, and visualizes the results using a graph.

![Knowledge Graph Preview](./assets/GRAPH%20pic.png)

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
- [Functions and Flow](#functions)
  - [main.py](#mainpy)
  - [utils.py](#utilspy)
- [Output](#output)
- [Flow](#flow)

## Introduction

The script processes a text document to create a knowledge graph, where nodes represent key concepts, and edges represent relationships between these concepts. The graph is visualized using NetworkX and PyVis.

## Project Structure

- `main.py`: The main script that orchestrates the data processing, extraction, and visualization.
- `utils.py`: A utility module containing functions to support data transformation, graph creation, and contextual proximity calculation.
- `assets/`: Directory containing input text files and output data files.
- `requirements.txt`: List of dependencies required to run the script.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Vishnu-add/Knowledge_graph_using_llms.git
    cd Knowledge_graph_using_llms
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Script

1. Place your input text file in the `assets/` directory. Ensure the file name is specified in the `input_file_name` variable in `main.py`.

2. Run the script:
    ```bash
    python main.py
    ```

3. The output will be saved in the `assets/data_output/` directory and the visualization will be available as `knowledge_graph.html`.

## Functions

### main.py

1. **Variable Initialization:**
    - Define input and output file paths.
    - Initialize the large language model.

2. **load_text()**
    - Loads the text document, cleans it, and splits it into chunks.
    - Returns the text chunks.

3. **create_df_for_all_chunks(pages)**
    - Converts text chunks into a DataFrame.
    - Returns the DataFrame.

4. **extract_concepts(df)**
    - Extracts key concepts and their relationships from the DataFrame using the language model.
    - Saves the resulting graphs and chunks to CSV files.
    - Returns the DataFrame containing the graph.

5. **calculate_contextual_proximity(dfg1)**
    - Calculates contextual proximity between terms in the graph.
    - Merges the original and proximity graphs.
    - Returns the merged DataFrame.

6. **colors2Community(communities)**
    - Assigns colors to different communities in the graph.
    - Returns a DataFrame with node colors and groups.

7. **calculate_networkX_graph(dfg)**
    - Constructs a NetworkX graph from the DataFrame.
    - Identifies communities for node coloring.
    - Returns the NetworkX graph.

8. **visualize_graph(G)**
    - Visualizes the NetworkX graph using PyVis.
    - Saves the visualization as an HTML file.

### utils.py

1. **graphPrompt(input, metadata, llm)**
    - Uses the language model to extract ontology terms and their relationships from a text chunk.
    - Returns a list of dictionaries with the extracted relationships.

2. **documents2Dataframe(documents)**
    - Converts a list of document chunks into a DataFrame.
    - Returns the DataFrame.

3. **df2Graph(dataframe)**
    - Converts a DataFrame of text chunks into a list of graph entities.
    - Returns the list of extracted ontology terms and relationships.

4. **graph2Df(nodes_list)**
    - Converts a list of graph nodes into a DataFrame, cleaning and standardizing the data.
    - Returns the DataFrame.

5. **contextual_proximity(df)**
    - Calculates the contextual proximity between terms in a DataFrame.
    - Returns the DataFrame with contextual proximity edges and nodes.

## Output

- **Data Files:** 
  - Extracted graph and chunk data are saved as CSV files in the `assets/data_output/` directory.
- **Visualization:**
  - The knowledge graph is visualized and saved as `knowledge_graph.html` in the project root directory.


## Flow

1. **Initialize Variables and Model:**
    - Set input and output file paths.
    - Initialize the language model `llm`.

2. **Load Text Document:**
    - `pages = load_text()`
      - Calls the `load_text()` function which:
        - Loads the text document specified in the `input_file_name`.
        - Cleans unnecessary line breaks and splits the document into chunks using `RecursiveCharacterTextSplitter`.
        - Returns the list of text chunks (`pages`).

3. **Create DataFrame for Text Chunks:**
    - `df = create_df_for_all_chunks(pages)`
      - Calls the `create_df_for_all_chunks(pages)` function which:
        - Converts the list of text chunks into a DataFrame.
        - Returns the DataFrame (`df`).

4. **Extract Concepts:**
    - `dfg1 = extract_concepts(df)`
      - Calls the `extract_concepts(df)` function which:
        - Extracts key concepts and their relationships using the language model through the `graphPrompt` function in `utils.py`.
        - Saves the resulting graphs and chunks to CSV files.
        - Returns the DataFrame containing the graph (`dfg1`).

5. **Calculate Contextual Proximity:**
    - `dfg = calculate_contextual_proximity(dfg1)`
      - Calls the `calculate_contextual_proximity(dfg1)` function which:
        - Calculates the contextual proximity between terms in the graph using the `contextual_proximity` function in `utils.py`.
        - Merges the original graph with the proximity graph.
        - Returns the merged DataFrame (`dfg`).

6. **Calculate NetworkX Graph:**
    - `G = calculate_networkX_graph(dfg)`
      - Calls the `calculate_networkX_graph(dfg)` function which:
        - Constructs a NetworkX graph from the DataFrame.
        - Identifies communities for node coloring using the `colors2Community` function in `main.py`.
        - Returns the NetworkX graph (`G`).

7. **Visualize Graph:**
    - `visualize_graph(G)`
      - Calls the `visualize_graph(G)` function which:
        - Visualizes the NetworkX graph using PyVis.
        - Saves the visualization as an HTML file.


**COLAB NOTEBOOK LINK** :

[KNOWLEDGE_GRAPH_FOR_SHAIKSPHERE](https://colab.research.google.com/drive/10Wha9JfOBHekJ7BTZdofu6gsOijBkqAU?usp=sharing)