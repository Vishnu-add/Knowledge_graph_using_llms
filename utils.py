import json
import pandas as pd
import numpy as np
import uuid
import json

def graphPrompt(input, metadata, llm):
    """
    Generate a list of ontology terms and their relationships from the given context using a large language model.

    Args:
        input (str): The text chunk from which to extract terms and relationships.
        metadata (dict): Metadata containing the chunk ID.
        llm (Ollama): The language model instance used to generate the ontology.

    Returns:
        list: A list of dictionaries, each containing 'chunk_id', 'node_1', 'node_2', and 'edge'.
    """
    chunk_id = metadata.get('chunk_id', None)
    SYS_PROMPT = ("You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include person (agent), location, organization, date, duration, \n"
            "\tcondition, concept, object, entity  etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them like the follwing. NEVER change the value of the chunk_ID as defined in this prompt: \n"
        "[\n"
        "   {\n"
        '       "chunk_id": "CHUNK_ID_GOES_HERE",\n'
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )
    SYS_PROMPT = SYS_PROMPT.replace('CHUNK_ID_GOES_HERE', chunk_id)

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    complt_prompt = SYS_PROMPT + USER_PROMPT
    response = llm.invoke(complt_prompt)

    aux1 = response
    # Find the index of the first open bracket '['
    start_index = aux1.find('[')
    # Find the index of the last closed bracket ']'
    end_index = aux1.rfind(']')
    # Slice the string from start_index to extract the JSON part and fix an unexpected problem with insertes escapes (WHY ?)
    json_string = aux1[start_index:end_index+1]
    # json_string = json_string.replace('\\\\\_', '_')
    # json_string = json_string.replace('\\\\_', '_')
    # json_string = json_string.replace('\\\_', '_')
    # json_string = json_string.replace('\\_', '_')
    # json_string = json_string.replace('\_', '_')
    json_string.lstrip() # eliminate eventual leading blank spaces
#####################################################
    print("json-string:\n" + json_string)
#####################################################
    try:
        result = json.loads(json_string)
        result = [dict(item) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    print("§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§")

    return result

def documents2Dataframe(documents) -> pd.DataFrame:
    """
    Convert a list of document chunks into a DataFrame.

    Args:
        documents (list): List of document chunks.

    Returns:
        DataFrame: DataFrame containing the text, metadata, and chunk IDs.
    """
    rows = []
    for chunk in documents:
        row = {
            "text": chunk.page_content,
            **chunk.metadata,
            "chunk_id": uuid.uuid4().hex,
        }
        rows = rows + [row]

    df = pd.DataFrame(rows)
    return df

def df2Graph(dataframe: pd.DataFrame) -> list:
    """
    Convert a DataFrame of text chunks into a list of graph entities using the graphPrompt function.

    Args:
        dataframe (DataFrame): DataFrame containing text chunks and metadata.

    Returns:
        list: List of extracted ontology terms and their relationships.
    """
    # dataframe.reset_index(inplace=True)
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}), axis=1
    )
    # invalid json results in NaN
    results = results.dropna()
    results = results.reset_index(drop=True)

    ## Flatten the list of lists to one single list of entities.
    concept_list = np.concatenate(results).ravel().tolist()
    return concept_list

def graph2Df(nodes_list) -> pd.DataFrame:
    """
    Convert a list of graph nodes into a DataFrame, cleaning and standardizing the data.

    Args:
        nodes_list (list): List of graph nodes.

    Returns:
        DataFrame: DataFrame containing the graph edges and nodes.
    """
    ## Remove all NaN entities
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(lambda x: x.lower())
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(lambda x: x.lower())
    return graph_dataframe

def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the contextual proximity between terms in a DataFrame.

    Args:
        df (DataFrame): DataFrame containing the initial graph edges and nodes.

    Returns:
        DataFrame: DataFrame containing the contextual proximity edges and nodes.
    """
    ## Melt the dataframe into a list of nodes
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)

    # Self join with chunk id as the key will create a link between terms occuring in the same text chunk.
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))

    # drop self loops
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)

    ## Group and count edges.
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    
    # Drop edges with 1 count
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2