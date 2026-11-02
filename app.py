import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class SalesPredictionEngine:
    """Core machine learning engine for data preprocessing, training, and evaluation."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.y_pred = None
        self.metrics = {}

    def load_and_preprocess(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except Exception as e:
            raise ValueError(f"Could not read CSV file: {e}")

        # Check for target column 'Sales_Revenue' and feature columns
        required_cols = {'TV_Budget', 'Radio_Budget', 'Social_Media_Budget', 'Sales_Revenue'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        # Handle missing values by replacing NaN with column median
        for col in required_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(self.df[col].median())

        return self.df

    def train_model(self):
        X = self.df[['TV_Budget', 'Radio_Budget', 'Social_Media_Budget']]
        y = self.df['Sales_Revenue']

        # Train-Test Split (80% Train, 20% Test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        # Fit Multiple Linear Regression Model
        self.model.fit(X_train_scaled, self.y_train)

        # Make Predictions
        self.y_pred = self.model.predict(X_test_scaled)

        # Evaluate Performance Metrics
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)

        self.metrics = {
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2_Score": round(r2, 4),
            "Intercept": round(self.model.intercept_, 2),
            "Coeff_TV": round(self.model.coef_[0], 2),
            "Coeff_Radio": round(self.model.coef_[1], 2),
            "Coeff_Social": round(self.model.coef_[2], 2)
        }
        return self.metrics

    def predict_single(self, tv, radio, social):
        input_data = np.array([[tv, radio, social]])
        input_scaled = self.scaler.transform(input_data)
        prediction = self.model.predict(input_scaled)
        return max(0.0, round(prediction[0], 2))

class PredictiveModelingGUI:
    """Desktop GUI interface using Tkinter and Matplotlib integration."""
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Revenue Predictive Modeling Tool - Sahil Rana (26BSC90001)")
        self.root.geometry("1100x750")
        self.root.configure(bg="#F4F6F9")

        self.engine = None
        self.canvas = None

        self._build_header()
        self._build_control_panel()
        self._build_metrics_panel()
        self._build_prediction_panel()
        self._build_plot_area()

    def _build_header(self):
        h_frame = tk.Frame(self.root, bg="#1B365D", height=60)
        h_frame.pack(fill="x", side="top")
        
        lbl_title = tk.Label(h_frame, text="Sales Revenue Predictive Modeling Tool", 
                             font=("Helvetica", 15, "bold"), fg="white", bg="#1B365D")
        lbl_title.pack(side="left", padx=20, pady=12)

        lbl_sub = tk.Label(h_frame, text="Course: Python Programming | Student: Sahil Rana (26BSC90001)", 
                          font=("Helvetica", 10), fg="#A0C4FF", bg="#1B365D")
        lbl_sub.pack(side="right", padx=20, pady=12)

    def _build_control_panel(self):
        c_frame = tk.Frame(self.root, bg="#E2E8F0", pady=10)
        c_frame.pack(fill="x", side="top")

        btn_load = tk.Button(c_frame, text="📁 Load Dataset & Train Model", command=self.load_and_train,
                             font=("Helvetica", 11, "bold"), bg="#2B547E", fg="white", 
                             activebackground="#1B365D", activeforeground="white", padx=15, pady=5, bd=0)
        btn_load.pack(side="left", padx=20)

        self.lbl_status = tk.Label(c_frame, text="No model trained. Load sample_advertising_sales.csv to begin.", 
                                  font=("Helvetica", 10, "italic"), fg="#4A5568", bg="#E2E8F0")
        self.lbl_status.pack(side="left", padx=10)

    def _build_metrics_panel(self):
        self.m_frame = tk.Frame(self.root, bg="#F4F6F9", pady=10)
        self.m_frame.pack(fill="x", side="top", padx=20)

        self.metric_widgets = {}
        titles = ["R² Score (Accuracy)", "RMSE ($ Error)", "MSE", "TV Coeff", "Radio Coeff", "Social Coeff"]

        for title in titles:
            card = tk.Frame(self.m_frame, bg="white", highlightbackground="#E2E8F0", highlightthickness=1)
            card.pack(side="left", expand=True, fill="both", padx=4)

            tk.Label(card, text=title, font=("Helvetica", 8, "bold"), fg="#718096", bg="white").pack(anchor="w", padx=8, pady=(6, 2))
            v_lbl = tk.Label(card, text="--", font=("Helvetica", 11, "bold"), fg="#1B365D", bg="white")
            v_lbl.pack(anchor="w", padx=8, pady=(0, 6))
            self.metric_widgets[title] = v_lbl

    def _build_prediction_panel(self):
        p_frame = tk.LabelFrame(self.root, text=" Live Revenue Estimator ", font=("Helvetica", 10, "bold"), 
                                bg="white", fg="#1B365D", padx=15, pady=8)
        p_frame.pack(fill="x", side="top", padx=20, pady=(0, 10))

        tk.Label(p_frame, text="TV Budget ($):", font=("Helvetica", 9), bg="white").grid(row=0, column=0, padx=5, pady=5)
        self.ent_tv = tk.Entry(p_frame, width=10, font=("Helvetica", 9))
        self.ent_tv.insert(0, "150")
        self.ent_tv.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(p_frame, text="Radio Budget ($):", font=("Helvetica", 9), bg="white").grid(row=0, column=2, padx=5, pady=5)
        self.ent_radio = tk.Entry(p_frame, width=10, font=("Helvetica", 9))
        self.ent_radio.insert(0, "25")
        self.ent_radio.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(p_frame, text="Social Media Budget ($):", font=("Helvetica", 9), bg="white").grid(row=0, column=4, padx=5, pady=5)
        self.ent_social = tk.Entry(p_frame, width=10, font=("Helvetica", 9))
        self.ent_social.insert(0, "40")
        self.ent_social.grid(row=0, column=5, padx=5, pady=5)

        btn_predict = tk.Button(p_frame, text="⚡ Predict Revenue", command=self.predict_custom, 
                                font=("Helvetica", 9, "bold"), bg="#276749", fg="white", bd=0, padx=10, pady=3)
        btn_predict.grid(row=0, column=6, padx=15, pady=5)

        self.lbl_prediction = tk.Label(p_frame, text="Predicted Revenue: --", font=("Helvetica", 10, "bold"), fg="#1B365D", bg="white")
        self.lbl_prediction.grid(row=0, column=7, padx=10, pady=5)

    def _build_plot_area(self):
        self.plot_container = tk.Frame(self.root, bg="white", highlightbackground="#E2E8F0", highlightthickness=1)
        self.plot_container.pack(fill="both", expand=True, padx=20, pady=(0, 15))

    def load_and_train(self):
        fpath = filedialog.askopenfilename(title="Select Sales Training Dataset", filetypes=[("CSV Files", "*.csv")])
        if not fpath:
            return

        try:
            self.engine = SalesPredictionEngine(fpath)
            self.engine.load_and_preprocess()
            metrics = self.engine.train_model()

            # Update Metrics UI
            self.metric_widgets["R² Score (Accuracy)"].config(text=str(metrics["R2_Score"]))
            self.metric_widgets["RMSE ($ Error)"].config(text=f"${metrics['RMSE']}")
            self.metric_widgets["MSE"].config(text=str(metrics["MSE"]))
            self.metric_widgets["TV Coeff"].config(text=str(metrics["Coeff_TV"]))
            self.metric_widgets["Radio Coeff"].config(text=str(metrics["Coeff_Radio"]))
            self.metric_widgets["Social Coeff"].config(text=str(metrics["Coeff_Social"]))

            self.render_evaluation_plots()
            self.lbl_status.config(text=f"Model Trained! Rows: {len(self.engine.df)} | R²: {metrics['R2_Score']}", fg="#276749")

        except Exception as e:
            messagebox.showerror("Model Training Error", str(e))

    def predict_custom(self):
        if not self.engine:
            messagebox.showwarning("Warning", "Please load dataset and train the model first!")
            return
        try:
            tv = float(self.ent_tv.get())
            radio = float(self.ent_radio.get())
            social = float(self.ent_social.get())

            pred_val = self.engine.predict_single(tv, radio, social)
            self.lbl_prediction.config(text=f"Predicted Revenue: ${pred_val:,.2f}", fg="#276749")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for advertising budgets.")

    def render_evaluation_plots(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=100)

        # 1. Actual vs Predicted Revenue Scatter Plot
        y_test = self.engine.y_test
        y_pred = self.engine.y_pred
        axes[0, 0].scatter(y_test, y_pred, color='#1B365D', alpha=0.7, edgecolors='w')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_title('Actual vs Predicted Sales Revenue', fontsize=9, fontweight='bold', color='#1B365D')
        axes[0, 0].set_xlabel('Actual Revenue ($)', fontsize=8)
        axes[0, 0].set_ylabel('Predicted Revenue ($)', fontsize=8)

        # 2. Residual Plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, color='#2B547E', alpha=0.7, edgecolors='w')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=1.5)
        axes[0, 1].set_title('Residuals vs Fitted Values', fontsize=9, fontweight='bold', color='#1B365D')
        axes[0, 1].set_xlabel('Predicted Revenue ($)', fontsize=8)
        axes[0, 1].set_ylabel('Residuals ($)', fontsize=8)

        # 3. TV Budget vs Revenue Fit
        sns.regplot(data=self.engine.df, x='TV_Budget', y='Sales_Revenue', ax=axes[1, 0], 
                    scatter_kws={'color':'#4B7BEC', 'alpha':0.5}, line_kws={'color':'#1B365D', 'lw':2})
        axes[1, 0].set_title('TV Budget Impact on Sales', fontsize=9, fontweight='bold', color='#1B365D')
        axes[1, 0].set_xlabel('TV Budget ($)', fontsize=8)
        axes[1, 0].set_ylabel('Sales Revenue ($)', fontsize=8)

        # 4. Residual Errors Distribution
        axes[1, 1].hist(residuals, bins=12, color='#778CA3', edgecolor='white')
        axes[1, 1].set_title('Residual Errors Normal Distribution', fontsize=9, fontweight='bold', color='#1B365D')
        axes[1, 1].set_xlabel('Residual Error ($)', fontsize=8)
        axes[1, 1].set_ylabel('Frequency', fontsize=8)

        plt.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    PredictiveModelingGUI(root)
    root.mainloop()