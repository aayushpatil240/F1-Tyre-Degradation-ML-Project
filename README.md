 **Tyre Degradation Prediction & Strategy Analysis (F1)**
 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 **Introduction**

I am a Computer Science student with a strong passion for motorsport, especially Formula 1.
I aim to work in performance engineering / motorsport data analytics, and this project is my attempt to combine software engineering, machine learning, and race strategy.

This project focuses on one of the most important problems in Formula 1:

Tyre degradation and race strategy decisions

Using real F1 race data, this project predicts tyre performance, classifies tyre usage styles, and presents the results in a race-engineer style dashboard.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 **Project Overview**

This is an ML-based tyre degradation analysis tool built using real Formula 1 race data.

The system:

1. User can select a driver and tyre compound

2. Displays an actual vs predicted tyre degradation graph

3. Predicts how tyre performance drops as laps increase

4. Classifies the stint into Conservative / Balanced / Aggressive strategy

5. Generates race engineer style notes explaining:

6. Tyre management quality

7. Pace and degradation behavior

8. Strengths and weaknesses of the stint

9. Clear improvement recommendations

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Tech Stack**

1. Python

2. FastF1 (real F1 race data)

3. Pandas & NumPy (data processing)

4. Scikit-learn (machine learning)

5. Matplotlib (visualization)

6. Streamlit (interactive dashboard)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Machine Learning Concepts Used**

1️⃣ Supervised Learning — Tyre Degradation Prediction

We trained a regression model to predict how much lap time is lost as tyres age.

Algorithm used: Random Forest Regressor

Why:

Tyre degradation is non-linear

Handles noise and real race variability well

Performs much better than linear models

What it does:

Learns the relationship between lap number, tyre compound, and pace loss.

2️⃣ Unsupervised Learning — Strategy Classification

We used clustering to discover different tyre usage styles without predefined labels.

Algorithm used: KMeans Clustering

Why:

No strategy labels exist beforehand

Lets patterns emerge naturally from the data

What it does:

Groups tyre stints into Conservative, Balanced, and Aggressive strategies based on degradation behavior.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**App Flow**

1. Load real F1 race lap data

2. Split laps into tyre stints

3. Calculate tyre degradation features

4. Train ML models on the data

5. Predict degradation for selected stints

6. Classify stint strategy using clustering

7. Convert results into race-engineer style notes

8. Display everything in an interactive dashboard

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Features**

1. Actual vs predicted tyre degradation curves

2. ML-based strategy classification

3. Comparison with other stints on the same compound

4. Race engineer style performance notes

5. Interactive driver & compound selection
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Screen Shots**
![WhatsApp Image 2026-01-31 at 2 27 08 PM](https://github.com/user-attachments/assets/07314b5f-9aa4-44b8-bf54-b54acacc646f)
![WhatsApp Image 2026-01-31 at 2 27 42 PM](https://github.com/user-attachments/assets/bdfe8a78-fec6-4133-8128-0d5ede227cc3)
