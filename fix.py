import json

# Load the original notebook
with open("Text to ASL Video Generation\project_final_ha33_ashtikma_text_to_asl_video.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove top-level 'widgets' metadata
nb["metadata"].pop("widgets", None)

# Also clean 'widgets' from any cell metadata if present
for cell in nb.get("cells", []):
    if "metadata" in cell and "widgets" in cell["metadata"]:
        cell["metadata"].pop("widgets")

# Overwrite the original notebook
with open("Text to ASL Video Generation\project_final_ha33_ashtikma_text_to_asl_video.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
