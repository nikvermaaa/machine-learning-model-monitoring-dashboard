# End-to-End Machine Learning Model Monitoring Pipeline

This project monitors pipeline designed to automatically detect data drift and model degradation in real-time. It leverages Evidently AI for statistical evaluation and SQLite for secure metric logging, all visualized through a custom Streamlit dashboard. It serves as a fully functional, plug-and-play template ready to be connected to live production databases.

## Tech Stack
* **Monitoring & Evaluation:** Evidently AI, Scikit-learn
* **Dashboard & Visualization:** Streamlit, Plotly, Pandas
* **Database:** SQLite

## Local Installation and Setup
To run this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ml-monitoring-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Simulate real-world traffic:**
   This generates synthetic usage data and logs it.
   ```bash
   python simulate_traffic.py
   ```

4. **Run the drift detection engine:**
   This analyzes the predictions against the baseline and populates the database.
   ```bash
   python detect_drift.py
   ```

5. **Launch the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

---

## How it Works with LIVE Data (Dropping the Simulator)

In the real world, you completely delete `simulate_traffic.py`. You don't need to generate fake data because real data is flowing in automatically.

Here is how the architecture changes for a live company:

* **The Live App (The Source):** Every time a user interacts with the live app (e.g., clicks "Add to Cart" on Flipkart), the app's backend asks the ML model for a prediction.
* **The Logging Database:** The backend takes the user's features (age, time on site) and the model's prediction, and writes them directly into a live database (like PostgreSQL, AWS RDS, or Snowflake). This replaces your local `predictions.db`.
* **The Automated Detective (Your Script):** You set up a scheduler (like Apache Airflow, AWS EventBridge, or a simple Linux CRON job) to run your `detect_drift.py` script automatically every night at midnight.
* **The Daily Shift:** * At midnight, your script wakes up.
  * It connects to the live company database.
  * It pulls all the predictions made that day.
  * It compares them against the original `reference.csv`.
  * It calculates the new drift scores and updates the dashboard.
* **The Dashboard:** The server just sits there, reading the latest data. When the engineers log in with their coffee in the morning, the charts have updated with yesterday's real-world traffic.

---

## How you can use this

1. **Swap the Reference Data:** They delete your `reference.csv` (the Adult Income dataset) and upload their own company's baseline CSV file.
2. **Change the Database Connection:** In `detect_drift.py`, they delete the SQLite connection and replace it with their company's live database credentials:

   ```python
   # Old: conn = sqlite3.connect("logs/predictions.db")
   # New: 
   conn = psycopg2.connect(
       host="company-database.aws.com",
       database="live_predictions",
       user="admin",
       password="secure_password"
   )
   ```

3. **Adjust the Dashboard Columns:** In `dashboard/app.py`, they would just change the names of the columns to match whatever features their specific model uses (e.g., changing "age" and "hours-per-week" to "cart_value" and "session_time").
