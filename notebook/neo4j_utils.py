# neo4j_custom_functions.py

from py2neo import Graph


def connect_to_neo4j(uri="bolt://localhost:7687",
                     username="neo4j",
                     password="password",
                     database="neo4j"):
    """Connect to Neo4j database"""
    try:
        graph = Graph(uri, user=username, password=password, name=database)
        print(f"Connected to Neo4j database: {database}")
        return graph
    except Exception as e:
        print(f"Error connecting to Neo4j: {str(e)}")
        return None


def create_neo4j_custom_function(graph, function_name, function_body):
    """Create a single custom Neo4j function"""
    function_query = f"""
    CREATE OR REPLACE FUNCTION {function_name}
    AS {function_body}
    """
    try:
        graph.run(function_query)
        print(f"Custom function '{function_name}' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating function '{function_name}': {str(e)}")
        return False


def create_custom_functions(graph, function_definitions):
    """Create multiple custom functions from a list of definitions"""
    success_count = 0
    for func_def in function_definitions:
        if create_neo4j_custom_function(graph, func_def["name"], func_def["body"]):
            success_count += 1

    print(f"Created {success_count} out of {len(function_definitions)} custom functions")
    return success_count