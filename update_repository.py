import subprocess
import os
import getpass

def commit_and_push_changes():
    try:
        # Get the current user's username
        username = getpass.getuser()

        # Define the repository directory based on the username
        if username == "andre":
            repo_dir = r"C:\Users\andre\iCloudDrive\Uni\Tercero\Proyecto\Codigo_vigas"
        elif username == "Mel":
            repo_dir = r"C:\Users\melan\OneDrive\Documentos\Experimental2_Sismómetro\TareaResistencia_AnálisisVigas\Vigas-besto-proyecto"
        else:
            raise Exception(f"Unknown user: {username}")
        # Check if the directory is a Git repository
        if not os.path.isdir(os.path.join(repo_dir, ".git")):
            print(f"The directory {repo_dir} is not a Git repository. Initializing...")
            subprocess.run(["git", "init"], check=True, cwd=repo_dir)
            print(f"Initialized empty Git repository in {repo_dir}")

        # Pull the latest changes from the remote repository
        subprocess.run(["git", "pull"], check=True, cwd=repo_dir)
        
        # Stage all changes
        subprocess.run(["git", "add", "."], check=True, cwd=repo_dir)
        
        # Commit the changes
        subprocess.run(["git", "commit", "-m", "Automated update by {username}"], check=True, cwd=repo_dir)
        
        # Push the changes to the remote repository
        subprocess.run(["git", "push"], check=True, cwd=repo_dir)
        
        print("Changes pulled, committed, and pushed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    commit_and_push_changes()