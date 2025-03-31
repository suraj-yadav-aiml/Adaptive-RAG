import os
from pathlib import Path

class ProjectStructure:
    """Class to create a structured project directory with necessary files."""

    def __init__(self, project_name: str) -> None:
        """
        Initializes the ProjectStructure instance.

        Args:
            project_name (str): The package name for the project.
        """
        self.project_name = project_name
        self.paths_to_create = self._generate_paths()

    def _generate_paths(self) -> list:
        """Generates a list of project files and directories."""
        return [
            # Top-level files
            "app.py",
            "requirements.txt",
            ".gitignore",

            # GitHub action file
            ".github/workflows/main.yml",

            # src directory and its contents
            f"src/__init__.py",
            f"src/{self.project_name}/__init__.py", 
            f"src/{self.project_name}/main.py",     

            # Sub-packages within the project package name
            f"src/{self.project_name}/ui/__init__.py",
            f"src/{self.project_name}/ui/uiconfigfile.py",
            f"src/{self.project_name}/ui/uiconfigfile.ini",
            f"src/{self.project_name}/ui/streamlit/__init__.py",
            f"src/{self.project_name}/ui/streamlit/loadui.py",
            f"src/{self.project_name}/ui/streamlit/display_result.py",
            f"src/{self.project_name}/tools/__init__.py",
            f"src/{self.project_name}/state/__init__.py",
            f"src/{self.project_name}/edges/__init__.py",
            f"src/{self.project_name}/nodes/__init__.py",
            f"src/{self.project_name}/llm/__init__.py",
            f"src/{self.project_name}/graph/__init__.py",
            f"src/{self.project_name}/retriever/__init__.py",
        ]

    def create_structure(self) -> None:
        """Creates directories and files for the project."""
        for path_str in self.paths_to_create:
            path = Path(path_str)

            try:
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)

                # Create the file if it does not exist
                if not path.exists():
                    path.touch()
                    print(f"Created file: {path}")
                else:
                    print(f"File already exists: {path}")

            except Exception as e:
                print(f"Error creating {path}: {e}")

if __name__ == "__main__":
    project = ProjectStructure("my_cool_app")
    project.create_structure()
