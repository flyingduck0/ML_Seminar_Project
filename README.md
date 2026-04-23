# Project 6: Prediction Markets 
**Seminar Financial Machine Learning (FEM11215) — Erasmus School of Economics**

### Team Members
* Adrian Koves 
* Aprajita Pandhey 
* Kerem Bora Özcan
* Leonardo Luciano Trapani 
* Stéphanie de Rooy 

## Repository Purpose
This repository houses a fully automated, cloud-based data pipeline and frontend dashboard. The system is designed to continuously ingest live prediction market data from Polymarket, align it with traditional financial asset prices, run a suite of statistical and machine learning models, and display the results on a live web interface. 

The entire process operates automatically on a daily schedule without the need for manual execution or a dedicated backend web server.

---

## Deployment & Access 
To ensure transparency and reproducibility, the entire project is deployed natively within GitHub's ecosystem. 

* **Live Dashboard (GitHub Pages):** The frontend graphical interface is hosted publicly via GitHub Pages. Because it utilizes client-side data fetching, it requires no local Python installation or server to run. 
  * **Link:** [Insert your actual GitHub Pages URL here, e.g., https://flyingduck0.github.io/ML_Seminar_Project/]
  * **Alternative Access:** Navigate to the main `Code` tab of this repository. In the right-hand sidebar, locate the **Deployments** section and click the active **github-pages** environment link (or the green checkmark) to open the live dashboard.
* **Automated Daily Pipeline (GitHub Actions):** The data extraction and machine learning models are executed via a continuous integration pipeline (`.github/workflows/daily_update.yml`). Execution logs can be viewed in the **Actions** tab of this repository.

---

## The Automated Process Flow
The architecture relies on a sequential daily loop, orchestrated entirely within GitHub. Here is how the system operates:

1. **The Trigger (Off-Peak Scheduling):** Every day at 03:17 UTC, a GitHub Actions cron job wakes up a temporary Ubuntu cloud server. *Note: The pipeline is intentionally scheduled at an off-peak minute (03:17) rather than the top of the hour (e.g., 00:00). This avoids the platform-wide "thundering herd" bottleneck caused by millions of default hourly cron jobs, ensuring our pipeline executes reliably without being dropped by GitHub's load balancers.*
2. **Data Extraction & Cleaning:** The server executes the data pipeline script. It connects to external APIs to pull the newest probabilities and asset prices, cleans the data, aligns the timestamps, and outputs a master dataset. 
3. **Machine Learning & Analytics:** The server then executes the analysis script. This engine reads the new master dataset, calculates rolling statistics, detects information "shocks," trains out-of-sample models, and computes backtest metrics.
4. **Data Storage:** The analysis script saves its final metrics as a series of simple CSV files inside the `results/` folder. The cloud server commits these updated CSVs back to the repository and shuts down.
5. **The Frontend Display:** The dashboard is a static HTML file hosted on GitHub Pages. Whenever a user navigates to the URL, their web browser downloads the HTML and uses client-side JavaScript to fetch the freshly updated CSVs directly from the `results/` folder, instantly rendering the newest charts and tables.

---

## Component Dictionary
Below is a breakdown of every major file in this repository and the specific function it serves in the pipeline.

### `.github/workflows/daily_update.yml`
* **Function:** The Scheduler / Orchestrator.
* **Description:** A YAML configuration file that tells GitHub Actions when to run the code, what dependencies to install (e.g., pandas, lightgbm, py_clob_client), and exactly which Python scripts to execute in sequence.

### `Pipeline_finished.py`
* **Function:** The Data Engineer.
* **Description:** Handles all API connections and data wrangling. It uses a custom retry-adapter to bypass rate limits when pulling data from Polymarket. It filters out irrelevant markets, handles timezone conversions, fetches traditional asset prices via `yfinance`, and merges everything into a clean, hourly timeline called `Final_Pipeline_Data.csv` with the addition of announcement static data up until the end of 2026.

### `MLanalysis_final.py`
* **Function:** The Analytics Engine.
* **Description:** Processes the cleaned data to generate all empirical findings. It performs the following sequential tasks:
    1. Computes rolling Z-scores to flag anomalous price movements ("shocks").
    2. Runs baseline Ordinary Least Squares (OLS) and Distributed Lag Models (DLM).
    3. Identifies optimal transmission lags using an Impulse Response Function (IRF).
    4. Trains a LightGBM model using a walk-forward expanding window for out-of-sample directional forecasting.
    5. Calculates pre-announcement activity ratios and runs trading strategy backtests.

### `index.html`
* **Function:** The User Interface.
* **Description:** A static frontend dashboard built with HTML, CSS, and Vanilla JavaScript. Instead of relying on a Python web framework, it uses PapaParse to read the output CSVs directly from the repository. It dynamically generates the interface, including live signal trackers, a transmission lag heatmap, interactive feature-importance bars, and event-study activity ratios, among other things.

### `results/` (Directory)
* **Function:** The Database.
* **Description:** A folder that stores the outputs generated by `MLanalysis_final.py` (e.g., `results_backtest.csv`, `current_signals.csv`). It serves as the bridge between the Python backend and the HTML frontend.

---

## References

Koves, A., Pandhey, A., Özcan, K. B., Trapani, L. L., & de Rooy, S. (2026). *Project 6: Prediction markets* [Unpublished manuscript]. Erasmus School of Economics.
