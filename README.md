# Spotify Audio Landscape Dashboard

An interactive exploratory dashboard for analyzing audio features and track popularity across 113,842 Spotify tracks and 114 genres. Built with Streamlit and Plotly.

---

## Files

| File | Description |
|---|---|
| `spotify_dashboard.py` | Main dashboard application |
| `spotify_updated.csv` | Cleaned dataset (113,842 tracks, 20 columns) |
| `requirements.txt` | Python dependencies |

---

## Setup

**1. Clone or download the project files** — all three files must be in the same folder.

**2. Create and activate a virtual environment (recommended)**
```bash
python -m venv spotify_env
source spotify_env/bin/activate        # Mac/Linux
spotify_env\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the dashboard**
```bash
streamlit run spotify_dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## Usage

Use the **sidebar** to filter the data:
- **Genres** — select one or more of the 114 available genres
- **Popularity Range** — drag the slider to focus on a popularity tier

All charts, KPI cards, and statistics update automatically when filters change.

The dashboard is organized into six tabs:

| Tab | Content |
|---|---|
| Genre Overview | Popularity rankings, audio feature radar, genre heatmap |
| Feature Deep-Dive | Distribution histograms, feature scatter, box plots |
| Popularity Drivers | Correlation analysis, feature-vs-popularity scatter, tier trends, top 50 tracks |
| Mood & Mode | Major/minor key breakdown, valence analysis, mood quadrant |
| Explicit Content | Explicit track distribution by genre, audio feature comparison |
| Audio Clusters | K-Means + PCA clustering by sonic fingerprint |

---

## Dataset

The dataset (`spotify_updated.csv`) is a preprocessed version of the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) available on Kaggle. If the CSV is not found in the same directory as the script, the dashboard will prompt you to upload it directly in the browser.
