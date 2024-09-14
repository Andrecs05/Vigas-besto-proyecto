import subprocess

def commit_and_push_changes():
    try:
        # Define the repository directory
        repo_dir = r"C:\Users\andre\iCloudDrive\Uni\Tercero\Proyecto\Codigo_vigas"

        # Stage all changes
        subprocess.run(["git", "add", "."], check=True, cwd=repo_dir)
        
        # Commit the changes
        subprocess.run(["git", "commit", "-m", "Automated update"], check=True, cwd=repo_dir)
        
        # Push the changes to the remote repository
        subprocess.run(["git", "push"], check=True, cwd=repo_dir)
        
        print("Changes committed and pushed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    commit_and_push_changes()