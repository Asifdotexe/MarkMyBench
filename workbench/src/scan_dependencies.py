import os
import nbformat
import re
import pkg_resources
import argparse

def get_package_version(package_name: str) -> str:
    """
    Get the version of a given package using `pkg_resources`.

    :param package_name: The name of the package to retrieve the version for.
    :type package_name: str
    :return: The version of the package if found, otherwise None.
    :rtype: str or None
    
    :raises pkg_resources.DistributionNotFound: If the package is not found in the environment.
    """
    try:
        package_version = pkg_resources.get_distribution(package_name).version
        return package_version
    except pkg_resources.DistributionNotFound:
        return None


def scan_dependencies_in_directory(directory_path: str) -> dict:
    """
    Scan a directory for Python files (.py) and Jupyter notebooks (.ipynb) and extract 
    package dependencies (import statements).

    :param directory_path: The directory to scan for files.
    :type directory_path: str
    :return: A dictionary where the keys are package names and the values are their versions.
    :rtype: dict
    
    This function will look for both Python scripts and Jupyter notebooks in the specified directory.
    It will extract import statements from both file types and fetch the package versions if available.
    """
    dependencies = {}

    # Scan Python files
    for root_directory, _, file_names in os.walk(directory_path):
        for file_name in file_names:
            if file_name.endswith(".py"):
                with open(os.path.join(root_directory, file_name), "r", encoding="utf-8") as python_file:
                    python_file_content = python_file.read()
                    import_statements = re.findall(r"^\s*(import|from) ([\w\.]+)", python_file_content, re.MULTILINE)
                    for import_statement in import_statements:
                        package_name = import_statement[1].split('.')[0]  # Get top-level package
                        if package_name not in dependencies:
                            package_version = get_package_version(package_name)
                            dependencies[package_name] = package_version

            # Scan Jupyter notebooks
            elif file_name.endswith(".ipynb"):
                with open(os.path.join(root_directory, file_name), "r", encoding="utf-8") as notebook_file:
                    notebook = nbformat.read(notebook_file, as_version=4)
                    for cell in notebook.cells:
                        if cell.cell_type == "code":
                            import_statements = re.findall(r"^\s*(import|from) ([\w\.]+)", cell.source, re.MULTILINE)
                            for import_statement in import_statements:
                                package_name = import_statement[1].split('.')[0]
                                if package_name not in dependencies:
                                    package_version = get_package_version(package_name)
                                    dependencies[package_name] = package_version

    return dependencies


def create_requirements_file(dependencies: dict, output_file: str = "requirements.txt") -> None:
    """
    Create a `requirements.txt` file with the list of dependencies and their versions.

    :param dependencies: A dictionary of dependencies (package names and versions).
    :type dependencies: dict
    :param output_file: The name of the output file (default is "requirements.txt").
    :type output_file: str
    
    This function writes the dependencies to a file, in the format `package==version`.
    If no version is available for a package, it writes `package` without the version.
    """
    with open(output_file, "w", encoding="utf-8") as requirements_file:
        for package_name, package_version in dependencies.items():
            if package_version:
                requirements_file.write(f"{package_name}=={package_version}\n")
            else:
                requirements_file.write(f"{package_name}\n")  # If no version, write the package name only
    print(f"Requirements file created: {output_file}")


def main() -> None:
    """
    Main function to parse command-line arguments and generate a requirements.txt file 
    for the scanned dependencies in the specified directory.

    Command-line arguments:
        - directory (str): The directory to scan for dependencies (Python files and Jupyter notebooks).
        - output (str): Output file for the requirements (default is "requirements.txt").
    
    :param directory: Directory to scan for dependencies (Python files and Jupyter notebooks).
    :type directory: str
    :param output: Output file for the requirements (default: "requirements.txt").
    :type output: str
    
    Example usage:
        python scan_requirements.py /path/to/project
        python scan_requirements.py /path/to/project --output custom_requirements.txt
    """
    # Set up argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Scan project for dependencies and create a requirements.txt file.")
    parser.add_argument("directory_path", help="Directory to scan for dependencies (Python files and Jupyter notebooks).")
    parser.add_argument("-o", "--output", help="Output file for the requirements (default: requirements.txt).", default="requirements.txt")
    
    args = parser.parse_args()

    # Scan the directory for dependencies
    dependencies = scan_dependencies_in_directory(args.directory_path)

    # Create the requirements.txt file
    create_requirements_file(dependencies, args.output)


if __name__ == "__main__":
    main()
