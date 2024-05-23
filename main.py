from langchain_community.llms import Ollama
import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
import uuid
from utils import *
import networkx as nx
import seaborn as sns
from pyvis.network import Network
# from IPython.core.display import display, HTML

# ## Variables
## Input data directory
input_file_name = "Scene_1_rousillon.txt"

data_dir = "assets/"+input_file_name
inputdirectory = Path(f"./{data_dir}")

## This is where the output csv files will be written
outputdirectory = Path(f"assets/data_output")

output_graph_file_name = f"graph_{input_file_name[:-4]}.csv"
output_graph_file_with_path = outputdirectory/output_graph_file_name

output_chunks_file_name = f"chunks_{input_file_name[:-4]}.csv"
output_chunks_file_with_path = outputdirectory/output_chunks_file_name

output_context_prox_file_name = f"graph_contex_prox_{input_file_name[:-4]}.csv"
output_context_prox_file_with_path = outputdirectory/output_context_prox_file_name


llm = Ollama(model = "llama3")

def load_text():
    """
    Load text data from a file, clean it, split into chunks, and return the chunks.
    
    Returns:
        List[Document]: A list of document chunks.
    """
    loader = TextLoader("assets/Scene_1_rousillon.txt")
    Document = loader.load()
    # clean unnecessary line breaks
    Document[0].page_content = Document[0].page_content.replace("\n", " ")

    # Rename the source name to only file name
    for i in range(len(Document)):
        Document[i].metadata['source'] = Document[i].metadata['source'].split('/')[-1]

    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    pages = splitter.split_documents(Document)
    print("Number of chunks = ", len(pages))
    print(pages[5].page_content)
    return pages

def create_df_for_all_chunks(pages):
    """
    Create a DataFrame for all document chunks.
    
    Args:
        pages (List[Document]): A list of document chunks.
    
    Returns:
        DataFrame: A DataFrame containing the chunks.
    """
    df = documents2Dataframe(pages)
    return df

# ## Extract Concepts
def extract_concepts(df):
    """
    Extract concepts from the document chunks and generate a graph DataFrame.
    
    Args:
        df (DataFrame): DataFrame containing document chunks.
    
    Returns:
        DataFrame: A DataFrame containing the graph edges and nodes.
    """
    regenerate = True  # toggle to True if the time-consuming (re-)generation of the knowlege extraction is required
    if regenerate:
        concepts_list = df2Graph(df)
        dfg1 = graph2Df(concepts_list)

        if not os.path.exists(outputdirectory):
            os.makedirs(outputdirectory)

        dfg1.to_csv(output_graph_file_with_path, sep=";", index=False)
        df.to_csv(output_chunks_file_with_path, sep=";", index=False)
    else:
        dfg1 = pd.read_csv(output_graph_file_with_path, sep=";")
        dfg1.replace("", np.nan, inplace=True)
        dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
        dfg1['count'] = 4
    return dfg1

def calculate_contextual_proximity(dfg1):
    """
    Calculate contextual proximity of concepts in the graph and merge with original graph DataFrame.
    
    Args:
        dfg1 (DataFrame): DataFrame containing the graph edges and nodes.
    
    Returns:
        DataFrame: A merged DataFrame containing original and contextually proximate edges and nodes.
    """
    dfg2 = contextual_proximity(dfg1)
    dfg2.to_csv(output_context_prox_file_with_path, sep=";", index=False)
    dfg2.tail()#

    # ### Merge both the dataframes
    dfg = pd.concat([dfg1, dfg2], axis=0)
    dfg = (
        dfg.groupby(["node_1", "node_2"])
        .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
        .reset_index()
    )
    return dfg

# ### Create a dataframe for community colors
def colors2Community(communities) -> pd.DataFrame:
    """
    Assign colors to communities in the graph.
    
    Args:
        communities (List[List[str]]): List of communities, where each community is a list of nodes.
    
    Returns:
        DataFrame: A DataFrame mapping nodes to their respective community colors and groups.
    """
    ## Define a color palette
    palette = "hls"
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

# ## Calculate the NetworkX Graph
def calculate_networkX_graph(dfg):
    """
    Calculate a NetworkX graph from the DataFrame of edges and nodes.
    
    Args:
        dfg (DataFrame): DataFrame containing the graph edges and nodes.
    
    Returns:
        Graph: A NetworkX graph object.
    """
    nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
    nodes.shape

    G = nx.Graph()

    # Add nodes to the graph
    for node in nodes:
        G.add_node(
            str(node)
        )
    
    # Add edges to the graph
    for index, row in dfg.iterrows():
        G.add_edge(
            str(row["node_1"]),
            str(row["node_2"]),
            title=row["edge"],
            weight=row['count']/4
        )

    # Calculate communities for coloring the nodes
    communities_generator = nx.community.girvan_newman(G)
    top_level_communities = next(communities_generator)
    next_level_communities = next(communities_generator)
    communities = sorted(map(sorted, next_level_communities))
    print("Number of Communities = ", len(communities))
    print(communities)

    # Add  colors to communities and make another dataframe
    colors = colors2Community(communities)

    # Add colors to the graph
    for index, row in colors.iterrows():
        G.nodes[row['node']]['group'] = row['group']
        G.nodes[row['node']]['color'] = row['color']
        G.nodes[row['node']]['size'] = G.degree[row['node']]
    
    return G

def visualize_graph(G):
    """
    Visualize the NetworkX graph using PyVis.
    
    Args:
        G (Graph): A NetworkX graph object.
    """
    net = Network(
        notebook=True,
        # bgcolor="#1a1a1a",
        cdn_resources="remote",
        height="800px",
        width="100%",
        select_menu=True,
        # font_color="#cccccc",
        filter_menu=False,
    )
    net.from_nx(G)
    net.force_atlas_2based(central_gravity=0.015, gravity=-31)

    net.show_buttons(filter_=['physics'])
    net.show("knowledge_graph.html")


# Main execution sequence
pages = load_text()
df = create_df_for_all_chunks(pages)
dfg1 = extract_concepts(df)
dfg = calculate_contextual_proximity(dfg1)
G = calculate_networkX_graph(dfg)
visualize_graph(G)