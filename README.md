# WebCred Dataset — Annotations for Online Source Reliability

A curated dataset of 500 websites annotated for the accuracy of their content (labels: Correct, Incorrect, Partially Correct), plus simple visualization and evaluation tooling.

**What's included**
- `website_dataset.json`: the dataset (each entry includes `id`, `url`, `topic`, `label`, and `reasoning`).
- `visualize.html`: lightweight interactive charts for exploring label and topic distributions.
- `eval.py`: example evaluation harness that queries a web-enabled LLM and saves outputs to `model_behavior_outputs.json`.
- `model_behavior_outputs.json`: example outputs from a model evaluation run.

## Dataset Structure

Each item in `website_dataset.json` contains:
- `id` (string): unique identifier
- `url` (string): source URL
- `topic` (string): subject/domain
- `label` (string): one of `Correct`, `Incorrect`, `Partially Correct`
- `reasoning` (string): human explanation for the assigned label

## Visualizing the data

The easiest way is to serve the repository and open `visualize.html` in a browser (most browsers block local file access for JSON):

```bash
python -m http.server 8000
# then open http://localhost:8000/visualize.html
```

## Running the evaluation (`eval.py`)

`eval.py` is an example script that:
- loads `website_dataset.json`
- constructs prompts for each item
- calls an LLM (via NVIDIA/Tavily tool bindings) to fetch web evidence and generate a response
- writes results to `model_behavior_outputs.json`

Prerequisites and notes:
- The script expects two API keys as environment variables: `NVIDIA_API_KEY` and `TAVILY_API_KEY`.
- Install required Python packages used in `eval.py` before running (the script uses `langchain_core`, `langchain_nvidia_ai_endpoints`, and `langchain_tavily` bindings).
- Run the script with:

```bash
python eval.py
```

Outputs from a run are saved to `model_behavior_outputs.json` in the repository root.

## License

This work is licensed under [MIT License](LICENSE).