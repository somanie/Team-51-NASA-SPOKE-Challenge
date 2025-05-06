# create_kg_utils_pickle.py
import os
import pickle
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define your KG version
KG_VERSION = "v0.0.1"  # Replace with your version


# Function to get base directory from environment variable or config file
def get_kg_base_directory():
    """
    Get knowledge graph BASE directory from environment variable or config file
    Returns the directory path or None if not configured
    """
    # First check environment variable
    base_dir = os.environ.get('KG_BASE_DIR')

    # If not found, check for config file
    if not base_dir:
        config_file = os.path.join(os.getcwd(), 'kg_config.txt')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                base_dir = f.read().strip()

    return base_dir


# Update this in your kg_utils.py file
def setup_directories():
    """Setup knowledge graph directory structure"""
    # Try to get configured base directory
    custom_base_dir = get_kg_base_directory()

    if custom_base_dir:
        # Use the configured base directory
        base_dir = custom_base_dir
    else:
        # Default: Use parent directory of current working directory
        notebook_dir = os.getcwd()
        parent_dir = os.path.dirname(notebook_dir)
        base_dir = parent_dir

    # Create a "knowledge_graph" directory inside the base directory
    kg_dir = os.path.join(base_dir, "knowledge_graph")
    os.makedirs(kg_dir, exist_ok=True)

    # Create version directory inside the knowledge graph directory
    version_dir = os.path.join(kg_dir, KG_VERSION)
    os.makedirs(version_dir, exist_ok=True)

    nodes_dir = os.path.join(version_dir, "nodes")
    rels_dir = os.path.join(version_dir, "rels")
    os.makedirs(nodes_dir, exist_ok=True)
    os.makedirs(rels_dir, exist_ok=True)

    # Print the directories for confirmation
    print(f"Base Directory: {base_dir}")
    print(f"Knowledge Graph Directory: {kg_dir}")
    print(f"Version Directory: {version_dir}")
    print(f"Nodes Directory: {nodes_dir}")
    print(f"Relationships Directory: {rels_dir}")

    return {"base": base_dir, "kg": kg_dir, "version": version_dir, "nodes": nodes_dir, "rels": rels_dir}


def save_dataframe(df, path, index=False):
    """Save dataframe to CSV"""
    df.to_csv(path, index=index)
    logger.info(f"Saved {len(df)} rows to {path}")


def calculate_statistics(data1, data2=None, method="default"):
    """Calculate statistics between two datasets using the specified method"""
    if method == "t-test":
        import scipy.stats as stats
        # Perform t-test between two arrays
        t_stat, p_value = stats.ttest_ind(data1, data2)
        return p_value

    elif method == "fold_change":
        # Calculate fold change between two arrays (mean ratio)
        import numpy as np
        return np.mean(data1) / np.mean(data2)

    else:
        # Default simple statistics on single dataset
        import numpy as np
        if data2 is not None:
            logger.warning(f"Second dataset provided but ignored for method: {method}")

        return {
            "count": len(data1),
            "mean": np.mean(data1),
            "median": np.median(data1),
            "std": np.std(data1)
        }

    return stats


# Create utils dictionary
utils = {
    "setup_directories": setup_directories,
    "save_dataframe": save_dataframe,
    "calculate_statistics": calculate_statistics,
    "KG_VERSION": KG_VERSION,
    "logger": logger,
    "get_kg_base_directory": get_kg_base_directory
}

# Save to pickle file
with open('kg_utils.pkl', 'wb') as f:
    pickle.dump(utils, f)

print("Successfully created kg_utils.pkl")