import os
from pathlib import Path

# --- CONFIGURATION ---
# Output file name
OUTPUT_FILE = "codebase_anchor.txt"

# Directories to completely skip (to avoid bloating tokens or leaking binaries)
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "runs",  # TensorBoard logs
    ".vscode",
    ".idea",
    "exports"
}

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    #".json",
    ".sh",
    ".toml"
}

# Files to explicitly ignore
IGNORE_FILES = {
    OUTPUT_FILE,
    "pack_codebase.py"
}
# ---------------------

def build_codebase_anchor():
    root_dir = Path.cwd()
    output_path = root_dir / OUTPUT_FILE
    
    print(f"🧬 Scanning codebase from: {root_dir}")
    print(f"🚫 Ignoring directories: {', '.join(IGNORE_DIRS)}")
    
    files_packed = 0
    
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"# 🔍 CODEBASE ANCHOR: {root_dir.name}\n")
        out.write("# Generated automatically for context parsing.\n\n")
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(root_dir):
            # Modify dirs in-place to avoid walking down ignored paths
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
            
            current_path = Path(root)
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                    
                file_path = current_path / file
                if file_path.suffix.lower() in INCLUDE_EXTENSIONS:
                    # Get relative path from project root
                    rel_path = file_path.relative_to(root_dir)
                    
                    print(f"📦 Packing: {rel_path}")
                    
                    out.write(f"='='='='='='='='='='='='='='='='='='='='='='='='='\n")
                    out.write(f"FILE: {rel_path}\n")
                    out.write(f"='='='='='='='='='='='='='='='='='='='='='='='='='\n")
                    out.write(f"```{file_path.suffix[1:]}\n")
                    
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f"// Error reading file: {str(e)}\n")
                        
                    out.write("\n```\n\n")
                    files_packed += 1
                    
    print(f"\n🚀 Success! Packed {files_packed} files into -> {OUTPUT_FILE}")
    print("You can now open this file, copy its content, and drop it here.")

if __name__ == "__main__":
    build_codebase_anchor()