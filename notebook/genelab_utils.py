import os
import shutil
import glob

from io import StringIO
import json
import hashlib
from urllib.parse import quote
import re
import time
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import requests
from dotenv import load_dotenv

API_ROOT = "https://visualization.osdr.nasa.gov/biodata/api/v2/"
DATASET_URL = f"{API_ROOT}dataset/"
DATASET_PATH = "../data"  # data download directory


def setup_environment():
    load_dotenv("../.env", override=True)

    NEO4J_DATA = os.getenv("NEO4J_DATA")
    if not NEO4J_DATA:
        raise Exception("NEO4J_DATA is not set in the .env file!")

    node_dir = os.path.join(NEO4J_DATA, "nodes")
    rel_dir = os.path.join(NEO4J_DATA, "relationships")

    # Create KG directories
    os.makedirs(node_dir, exist_ok=True)
    os.makedirs(rel_dir, exist_ok=True)
    os.makedirs("../data", exist_ok=True)

    print(f"Environment setup for KG version: {os.getenv('KG_VERSION')}")

    return node_dir, rel_dir


def validate_kg_metadata():
    NEO4J_METADATA = os.getenv("NEO4J_METADATA")
    if not NEO4J_METADATA:
        raise Exception("NEO4J_METADATA is not set in the .env file!")

    node_dir = os.path.join(NEO4J_METADATA, "nodes")
    rel_dir = os.path.join(NEO4J_METADATA, "relationships")
    dirs = [node_dir, rel_dir]

    # The following columns are required for a metadata file in that order
    required = ["property", "type", "description", "example"]
    required_cols = ",".join(required)

    error = False
    for d in dirs:
        for fn in glob.glob(os.path.join(d, "*.csv")):
            try:
                cols = pd.read_csv(fn).columns.tolist()
            except Exception as e:
                print(f"Warning: could not parse {fn!r}: {e}")
                continue

            actual_cols = ",".join(cols)
            if required_cols != actual_cols:
                print(
                    f"ERROR: The columns in {fn!r}: {actual_cols} don't match the required columns: {required_cols}"
                )
                error = True

            # TODO
            # Check types
            # Check that the description and example are not empty
            # Check filename syntax
            # Check if node names match the names in the relationship files

    if error:
        raise Exception("ERROR: Invalid metadata files!")

    print("Metadata files passed the check!")


def get_processed_datasets():
    metadata = get_info()
    metadata = filter_by_gl_processed(metadata)
    metadata = add_sample_counts(metadata)
    return metadata


def get_info():
    url = (
        f"{API_ROOT}/query/metadata/"
        "?"
        "study.characteristics.organism.term accession number"
        "&"
        "investigation.study assays.study assay technology type"
        "&"
        "investigation.study assays.study assay measurement type"
        "&"
        "study.characteristics.material type"
        "&"
        "study.characteristics.material type.term accession number"
        "&"
        "file.category"
        "&"
        "file.subcategory"
        # Note: the new API formats the header as a single line, but if you needed to break it up like before,
        # we have support for backwards compatibility:
        # "&" "format.header.multi" # you'd use this to break up the header into two lines for the "legacy" format
        # "&" "format.header.mark" # you'd use this to prepend "#" to header lines for the "legacy" format
    )
    metadata = pd.read_csv(quote(url, safe=":/=?&"), na_filter=False)

    # Simplify column names
    metadata.rename(
        columns={
            "id.accession": "identifier",
            "investigation.study assays.study assay measurement type": "measurement",
            "investigation.study assays.study assay technology type": "technology",
            "id.assay name": "assay_name",
            "study.characteristics.material type": "material",
        },
        inplace=True,
    )

    # Assign taxonomy id
    metadata["taxonomy"] = metadata["study.characteristics.organism.term accession number"].apply(
        lambda s: (
            s.split("/")[-1]
            if s.startswith("http://purl.bioontology.org/ontology/NCBITAXON/")
            else ""
        )
    )

    # Sort by identifier
    metadata = metadata.sort_values(
        by="identifier", key=lambda col: col.str.extract(r"-(\d+)$")[0].astype(int)
    ).reset_index(drop=True)

    return metadata


def filter_by_gl_processed(metadata):
    metadata["file.category"] = (
        metadata["file.category"]
        .fillna("")
        .apply(lambda s: (s if re.search(r"^genelab processed", s, flags=re.IGNORECASE) else False))
    )
    metadata["file.subcategory"] = (
        metadata["file.subcategory"]
        .fillna("")
        .apply(lambda s: (s if re.search(r"^processed data", s, flags=re.IGNORECASE) else False))
    )

    metadata = metadata.drop_duplicates().rename(  # there may be multiple "Genelab Processed" lines per sample, etc
        columns={
            "file.category": "file.GL-processed",
            "file.subcategory": "file.non-GL-processed",
        }
    )

    return metadata[
        (metadata["file.GL-processed"] != False) | (metadata["file.non-GL-processed"] != False)
    ].copy()


def add_sample_counts(metadata):
    per_sample_counts = metadata.drop(columns="id.sample name").value_counts().reset_index()
    per_sample_counts.insert(
        2, "id.sample count", per_sample_counts.pop("count")
    )  # just moving it from the last position to where it belongs
    return per_sample_counts.sort_values(by=["identifier", "assay_name"])


def filter_by_technology_type(metadata, technology_types):
    return metadata[
        metadata["technology"].str.lower().isin([x.lower() for x in technology_types])
    ].copy()


def filter_by_organism(metadata, taxids):
    metadata = metadata[metadata["taxonomy"].isin(taxids)].copy()
    #metadata["organism"] = metadata["taxonomy"].map(lambda x: taxids.get(x))
    metadata["organism"] = metadata["taxonomy"].map(taxids)
    return metadata


def download_data_files(assays, file_types, filters, reset=False):
    if reset:
        shutil.rmtree(DATASET_PATH)

    os.makedirs(DATASET_PATH, exist_ok=True)

    file_list = []

    for _, row in assays.iterrows():
        identifier = row["identifier"]
        technology = row["technology"]

        file_type = file_types.get(technology, None)
        filter_func = filters.get(file_type, None)

        if file_type:
            url = os.path.join(DATASET_URL, identifier, "files")
            try:
                response = requests.get(url, allow_redirects=True, timeout=10)
                response.raise_for_status()

                # Get filename and URL for each dataset and download it
                datafile_info = get_file_info(response.json(), file_type)
                for info in datafile_info:
                    filename = info["filename"]
                    file_url = info["url"]

                    success = download_data_file(file_url, filename, filter_func, DATASET_PATH)
                    time.sleep(0.1)
                    if not success:
                        continue

                    # Save info about the downloaded file
                    file_info = row.copy()
                    file_info["filename"] = filename
                    file_info["url"] = file_url
                    file_list.append(file_info)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {str(e)}")

    return pd.DataFrame(file_list)


def get_file_info(data, file_type):
    rows = []

    for identifier, content in data.items():
        files = content.get("files", {})
        for file_name, file_details in files.items():
            # These types of files are obsolete
            if "ERCCnorm" in file_name:
                continue
            if file_type in file_name:
                rows.append(
                    {
                        "identifier": identifier,
                        "filename": file_name,
                        "url": file_details.get("URL"),
                    }
                )
    return rows


def download_data_file(url, filename, filter_func, dataset_path):
    file_path = os.path.join(dataset_path, filename)

    if not os.path.exists(file_path):
        try:
            response = requests.get(url, allow_redirects=True, timeout=10)
            response.raise_for_status()

            print(f"Downloading: {filename}")

            # Load CSV content into DataFrame
            data = pd.read_csv(StringIO(response.text), low_memory=False)

            # Reduce the size of the data file by applying a filter function
            if filter_func is not None:
                filtered_data = filter_func(data)

            if filtered_data.empty:
                print(f"Skipping file: {filename}. No data after filtering.")
                return False

            # Save the filtered DataFrame
            filtered_data.to_csv(file_path, index=False)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}: {str(e)}")
    else:
        print(f"File already exist: {filename}")

    return True


def get_metadata(manifest):
    study_list = []
    for _, row in manifest.iterrows():
        identifier = row["identifier"]
        taxonomy = row["taxonomy"]
        organism = row["organism"]
        study_list.extend(extract_metadata(identifier, taxonomy, organism))
        time.sleep(0.1)

    return pd.DataFrame(study_list)


def to_iso_date(date_str) -> str:
    """
    Parse a date string in almost any common format
    and return an ISO‐formatted date "YYYY-MM-DD".
    Returns None if parsing fails.
    """
    try:
        dt = parse(date_str, fuzzy=True)
        return dt.date().isoformat()
    except (ValueError, TypeError):
        return ""


def to_list(x):
    """Always return a list.
    - If x is already a list, return it.
    - If x is None, return [].
    - Otherwise, wrap x in [x].
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def extract_metadata(accession, taxonomy, organism):
    """
    Fetches the JSON for the given OSDR dataset accession (e.g. "OSD-47")
    and extracts:
      - mission_name
      - mission_start_date
      - mission_end_date
      - flight_program
      - space_program
      - project_type
      - project_title

    Returns a dict with those keys (values will be None if the field is missing).
    """
    url = f"https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/{accession}/?format=json"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Top-level object for this accession
    ds = data.get(accession, {})

    # Metadata block
    meta = ds.get("metadata", {})

    # Progam info
    flight_program = meta.get("flight program", "")
    space_program = meta.get("space program", "")
    project_type = meta.get("project type", "")
    project_title = meta.get("project title", "")
    # Join project titles if there are multiple
    project_title = ", ".join(to_list(project_title))

    # Mission info
    mission = meta.get("mission", {})
    raw_name = mission.get("name", "")
    raw_start_date = mission.get("start date", "")
    raw_end_date = mission.get("end date", "")

    # One study may map to multiple missions
    mission_name = to_list(raw_name)
    mission_start_date = to_list(raw_start_date)
    mission_end_date = to_list(raw_end_date)

    study_list = []
    for name, start_date, end_date in zip(mission_name, mission_start_date, mission_end_date):
        study = {
            "identifier": accession,
            "project_type": project_type,
            "project_title": project_title,
            "taxonomy": taxonomy,
            "organism": organism,
            "flight_program": flight_program,
            "space_program": space_program,
            "mission_id": name.replace(" ", "-"),
            "name": name,
            "start_date": to_iso_date(start_date),
            "end_date": to_iso_date(end_date),
        }
        study_list.append(study)

    return study_list


def extract_gene_info(manifest):
    gene_list = []
    for _, row in manifest.iterrows():
        file_path = os.path.join(DATASET_PATH, row["filename"])
        if os.path.exists(file_path):
            df = pd.read_csv(
                file_path,
                usecols=["ENTREZID", "GENENAME"],
                dtype=str,
                keep_default_na=False,
            )
            df["organism"] = row["organism"]
            df["taxonomy"] = str(row["taxonomy"])
            gene_list.append(df)

    mgenes = pd.concat(gene_list, ignore_index=True)
    mgenes.drop_duplicates(subset="ENTREZID", inplace=True)

    # Match names of properties in metagraph
    mgenes = mgenes[["ENTREZID", "GENENAME", "organism", "taxonomy"]]
    mgenes.rename(columns={"ENTREZID": "identifier", "GENENAME": "name"}, inplace=True)

    # Remove version number
    mgenes["identifier"] = mgenes.identifier.apply(lambda x: x.split(".")[0])

    return mgenes


def extract_assay_info(manifest, variables):
    manifest["factors"] = manifest.apply(
        get_factor_data,
        args=(DATASET_PATH, variables),
        axis=1,
    )
    # Put each pair of factors into a separate row and split it into 3 columns
    assays = manifest.explode("factors")
    assays["factors"], assays["factors_1"], assays["factors_2"] = zip(*assays["factors"])

    return assays


def get_factor_data(row, dataset_path, variables):
    file_path = os.path.join(dataset_path, row["filename"])
    data = pd.read_csv(file_path, low_memory=False, nrows=1)
    cols = data.columns
    end_points = variables.get(row["measurement"], "")
    if end_points == "":
        print(
            f"Error: no relevant columns found in file {row['filename']} for {variables.keys()} measurements"
        )
        return []
    factors = [get_factors(col) for col in cols if col.startswith(end_points)]

    return factors


def get_factors(column_name):
    factors_string = column_name.split("_", 1)[1]
    parts = column_name.split(")v(")
    part1 = parts[0].split("(")[1]
    part2 = parts[1]
    factors_1 = [x.strip(" ()") for x in part1.split("&")]
    factors_2 = [x.strip(" ()") for x in part2.split("&")]

    return factors_string, factors_1, factors_2


def extract_materials(assays):
    # Extract cell or tissue types from the factors lists and materials column
    factors_1 = assays["factors_1"].to_list()
    unique_factors_1 = [item for sublist in factors_1 for item in sublist]
    factors_2 = assays["factors_2"].to_list()
    unique_factors_2 = [item for sublist in factors_2 for item in sublist]
    materials = assays["material"].to_list()

    unique_factors = sorted(set(unique_factors_1 + unique_factors_2 + materials))
    materials = pd.DataFrame(unique_factors, columns=["material"])
    return materials


def assign_material_to_assays(assays, mapped_materials):
    material_dict = {
        row["material"]: (row["material"], row["material_name"], row["material_id"])
        for _, row in mapped_materials.iterrows()
    }
    assays[
        [
            "material_1",
            "material_2",
            "material_name_1",
            "material_name_2",
            "material_id_1",
            "material_id_2",
        ]
    ] = assays.apply(assign_material, material_dict=material_dict, axis=1)
    return assays


def assign_material(row, material_dict):
    # Assign material to each factor. If a factor doesn't contain a material,
    # use the default material specified for the assay
    default = material_dict.get(row["material"], (row["material"], "", ""))
    mat1 = next((m for f in row["factors_1"] if (m := material_dict.get(f))), default)
    mat2 = next((m for f in row["factors_2"] if (m := material_dict.get(f))), default)
    return pd.Series([mat1[0], mat2[0], mat1[1], mat2[1], mat1[2], mat2[2]])


def add_assay_identifiers(assays):
    # pre‑compute each row’s MD5 hash of its JSON representation
    hashes = [
        hashlib.md5(json.dumps(r, sort_keys=True).encode()).hexdigest()
        for r in assays.to_dict("records")
    ]
    return assays.assign(
        study_id=assays["identifier"],
        assay_hash=hashes,
        identifier=lambda df: df["study_id"].str.cat(df["assay_hash"], sep="-"),
    )


def extract_transcription_data(assays: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    For each transcription‑profiling assay in `assays`, read its file once,
    extract ENTREZID/log2fc/adj_p.value columns, filter by threshold, and
    return a DataFrame of edges with columns ['from', 'to', 'log2fc', 'adj_p_value'].
    """
    rows = []
    cols = ["from", "to", "log2fc", "adj_p_value"]

    # Filter to transcription profiling and group by filename
    tp = assays[assays["measurement"] == "transcription profiling"]
    for filename, grp in tp.groupby("filename"):
        # Print study_id when loading a new file
        print(f"processing: {grp['study_id'].iat[0]}")
        df = pd.read_csv(os.path.join(DATASET_PATH, filename), low_memory=False)

        for _, row in grp.iterrows():
            f = row["factors"]
            log2fc_col = f"Log2fc_{f}"
            adj_col = f"Adj.p.value_{f}"

            # Skip if expected columns are missing
            if not {log2fc_col, adj_col}.issubset(df.columns):
                continue

            # Select columns
            sub = df[["ENTREZID", log2fc_col, adj_col]].dropna()

            # Filter by p‑value threshold
            sub = sub[sub[adj_col] <= threshold]
            if sub.empty:
                print(f"No statistically significant data for {row.study_id}: {log2fc_col}")
                continue

            # Rename to Neo4j convention
            sub = sub.rename(
                columns={"ENTREZID": "to", log2fc_col: "log2fc", adj_col: "adj_p_value"}
            )
            sub["from"] = row["identifier"]  # direct assignment of "from"
            sub["to"] = sub["to"].astype(int)

            # Keep only the four columns in order
            sub = sub[cols]
            rows.append(sub)

    # Concatenate or return empty
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=cols)


def extract_methylation_data(assays: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    For each DNA‑methylation‑profiling assay in `assays`, read its file once,
    extract ENTREZID/methylation_diff/q_value and region columns, filter by
    q‑value threshold, and return a DataFrame with the combined results.
    """
    rows = []
    cols = [
        "ENTREZID",
        "methylation_diff",
        "q_value",
        "chr",
        "start",
        "end",
        "dist.to.feature",
        "in_promoter",
        "in_exon",
        "in_intron",
        "assay_id",
        "methylation_id",
    ]

    # Filter by DNA methylation profiling and group by file
    dm = assays[assays["measurement"] == "DNA methylation profiling"]
    for filename, grp in dm.groupby("filename"):
        # print study_id when loading each file
        print(f"processing: {grp['study_id'].iat[0]}")
        df = pd.read_csv(os.path.join(DATASET_PATH, filename), low_memory=False)

        for _, row in grp.iterrows():
            f = row["factors"]
            diff_col = f"meth.diff_{f}"
            qv_col = f"qvalue_{f}"

            # skip if expected columns are missing
            if not {diff_col, qv_col}.issubset(df.columns):
                continue

            # select & drop NA
            sub = df[
                [
                    "ENTREZID",
                    diff_col,
                    qv_col,
                    "chr",
                    "start",
                    "end",
                    "dist.to.feature",
                    "prom",
                    "exon",
                    "intron",
                ]
            ].dropna()

            # filter by q‑value threshold
            sub = sub[sub[qv_col] <= threshold]
            if sub.empty:
                print(f"No significant data for {row['study_id']}: {diff_col}")
                continue

            # rename to standard column names
            sub = sub.rename(
                columns={
                    diff_col: "methylation_diff",
                    qv_col: "q_value",
                    "prom": "in_promoter",
                    "exon": "in_exon",
                    "intron": "in_intron",
                }
            )

            # map 0/1 → 'false'/'true'
            neo4j_bool = {1: "true", 0: "false"}
            sub["in_promoter"] = sub["in_promoter"].map(neo4j_bool)
            sub["in_exon"] = sub["in_exon"].map(neo4j_bool)
            sub["in_intron"] = sub["in_intron"].map(neo4j_bool)

            # add assay_id and methylation_id
            sub["assay_id"] = row["identifier"]
            sub["methylation_id"] = (
                sub["chr"] + ":" + sub["start"].astype(str) + "-" + sub["end"].astype(str)
            )

            # keep columns in the desired order
            rows.append(sub[cols])

    # concatenate or return empty frame with proper columns
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=cols)


def list_to_string(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).all():
            df[col] = df[col].apply(lambda x: "|".join(map(str, x)))

    return df


def save_dataframe_to_kg(df, node_or_rel_name, node_or_rel_directory):
    # Remove previous versions of the node or relationship file
    for file_path in glob.glob(os.path.join(node_or_rel_directory, f"{node_or_rel_name}_*.csv")):
        os.remove(file_path)

    # Convert any column of type list to a "|" separated string (required for Neo4j import)
    df = list_to_string(df)

    # Remove duplicate nodes or relationships
    if "identifier" in df.columns and "nodes" in node_or_rel_directory:
        df = df.drop_duplicates(subset="identifier")
    elif {"from", "to"}.issubset(df.columns) and "relationships" in node_or_rel_directory:
        df = df.drop_duplicates(subset=["from", "to"])
    else:
        raise ValueError(
            f"Invalid node or relationship file {list(df.columns)} or directory {node_or_rel_directory}. See https://github.com/sbl-sdsc/kg-import for details."
        )

    update_date = datetime.today().strftime("%Y-%d-%m")
    df.to_csv(
        os.path.join(node_or_rel_directory, f"{node_or_rel_name}_{update_date}.csv"),
        index=False,
    )

    return df
