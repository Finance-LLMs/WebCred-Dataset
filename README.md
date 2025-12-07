# Website-dataset

A curated dataset of 200 websites with labels indicating the accuracy of their content (Correct, Incorrect, or Partially Correct) across various topics.

## Dataset Structure

Each entry in `Website_dataset.json` contains:
- `id`: Unique identifier
- `url`: Website URL
- `topic`: Subject/domain category
- `label`: Accuracy classification (Correct/Incorrect/Partially Correct)
- `reasoning`: Explanation for the label

## Visualization

Open `visualize.html` in a web browser to view interactive pie charts showing:
- **Label Distribution**: Breakdown of Correct, Incorrect, and Partially Correct entries
- **Topic Distribution**: Range of domains/topics covered in the dataset

To view the visualization, you can:
1. Clone this repository
2. Open `visualize.html` in your browser (requires loading `Website_dataset.json` from the same directory)

> **Note**: Due to browser security restrictions, you may need to serve the files from a local web server. You can use Python's built-in server:
> ```bash
> python -m http.server 8000
> ```
> Then open http://localhost:8000/visualize.html