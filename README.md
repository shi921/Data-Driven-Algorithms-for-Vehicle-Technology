# Data-Driven-Algorithms-for-Vehicle-Technology
A structured repository containing my study materials, notes, and implementation code for the course "Data-Driven Algorithms for Vehicle Systems". This repository documents my learning journey through concepts like parameter estimation, classification, regression, forecasting, and Monte Carlo methods, with a focus on electric vehicle (EV) routing applications.

## 🔑 Key Topics Covered

1. **Fundamentals**
   - Python tools for vehicle data processing (pandas, matplotlib, scipy)
   - EV routing basics and transportation database (TRDB) integration
   - API usage for real-time traffic data retrieval

2. **Statistical Foundations**
   - Probability distributions for vehicle parameters
   - Hypothesis testing (Chi-square, ANOVA, Kruskal-Wallis)
   - Data preprocessing and sensor noise reduction

3. **Parameter Estimation**
   - Least Squares (LS) and Recursive Least Squares (RLS)
   - Robust estimation for handling outliers
   - Multi-model estimation for parameter突变 (abrupt changes)

4. **Classification Algorithms**
   - Rule-based systems and fuzzy logic
   - Naive Bayes Classifier (NBC)
   - Hidden Markov Models (HMM) for temporal traffic phase classification

5. **Regression & Forecasting**
   - Linear and nonlinear regression models (SVM, Decision Trees, ANN)
   - Time series forecasting (ARIMA)
   - Deterministic vs. stochastic prediction evaluation (CRPS metric)

6. **Monte Carlo Methods**
   - Uncertainty propagation in EV energy consumption
   - Reachability calculation for EV routing
   - Practical implementation of Monte Carlo simulations


## 💻 Technical Requirements

- Python 3.8+
- Required libraries:
  ```
  pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```


## 📝 Notes on Usage

- All code is tested with sample vehicle datasets (simulated traffic data, EV sensor readings)
- Notes follow a "concept-explanation-formula-implementation" structure
- Visualizations include comparative analysis of algorithms (e.g., BER for classification, CRPS for forecasting)


## 📌 Acknowledgments

Course materials based on lectures and exercises from **"Data-Driven Algorithms for Vehicle Systems"**, focusing on practical applications in electric vehicle routing and energy management.


## 📄 License

This repository is for educational purposes only. All course materials and original concepts belong to their respective creators.
