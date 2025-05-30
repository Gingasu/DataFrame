import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas.plotting import scatter_matrix

housing = None
preprocessor = None
X_train_prepared_df = None
numerical_features_in_pipeline = []
best_xgboost_model = None
X_test_global = None
y_test_global = None
global_rmse = None
global_mae = None
global_r2 = None
global_initial_xgb_rmse = None

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            rooms_ix_ = numerical_features_in_pipeline.index('total_rooms')
            bedrooms_ix_ = numerical_features_in_pipeline.index('total_bedrooms')
            population_ix_ = numerical_features_in_pipeline.index('population')
            households_ix_ = numerical_features_in_pipeline.index('households')

            rooms_per_household = X[:, rooms_ix_] / X[:, households_ix_]
            population_per_household = X[:, population_ix_] / X[:, households_ix_]

            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix_] / X[:, rooms_ix_]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]

        except ValueError as e:
            messagebox.showerror("Error", f"Error in CombinedAttributesAdder: Missing required feature in numerical pipeline ({e}). Ensure 'total_rooms', 'total_bedrooms', 'population', 'households' are in numerical_features.")
            return X
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred in CombinedAttributesAdder: {e}")
            return X

def prepare_data():
    global housing, X_train_prepared_df, preprocessor, numerical_features_in_pipeline, X_test_global, y_test_global
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    try:
        X = housing.drop('median_house_value', axis=1, errors='ignore')
        y = housing['median_house_value']

        X_train, X_test_global, y_train, y_test_global = train_test_split(X, y, test_size=0.2, random_state=42)

        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = ['ocean_proximity']

        if 'longitude' in numerical_features:
            numerical_features.remove('longitude')
        if 'latitude' in numerical_features:
            numerical_features.remove('latitude')

        numerical_features_in_pipeline = numerical_features[:]

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('attribs_adder', CombinedAttributesAdder()),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])

        X_train_prepared = preprocessor.fit_transform(X_train)

        original_categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[0]
        new_categorical_feature_names = [f"ocean_proximity_{cat.replace('<', '').replace('>', '')}" for cat in original_categories]

        feature_names_prepared_numerical = list(numerical_features_in_pipeline) + ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
        feature_names_prepared = feature_names_prepared_numerical + new_categorical_feature_names

        X_train_prepared_df = pd.DataFrame(X_train_prepared, columns=feature_names_prepared, index=X_train.index)
        messagebox.showinfo("Success", "Data preprocessing has been completed.")
        preprocess_button.config(state=tk.DISABLED)
        show_prepared_button.config(state=tk.NORMAL)
        spot_checking_button.config(state=tk.NORMAL)
        tune_xgboost_button.config(state=tk.NORMAL)
    except Exception as e:
        messagebox.showerror("Error", f"Error during data preprocessing: {e}")

exploration_buttons = []
preprocess_button = None
show_prepared_button = None
analyze_button = None
spot_checking_button = None
tune_xgboost_button = None
evaluate_button = None
summary_button = None
root = None

def perform_spot_checking():
    global X_train_prepared_df, housing, global_initial_xgb_rmse
    if X_train_prepared_df is None or housing is None:
        messagebox.showerror("Error", "Please load and prepare the dataset first.")
        return

    messagebox.showinfo("Info", "Starting spot-checking... This may take a moment.")

    X = X_train_prepared_df
    y = housing.loc[X.index, 'median_house_value']

    models = {
        'Linear Regression': LinearRegression(),
        'KNeighbors Regressor': KNeighborsRegressor(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42, n_estimators=100),
        'XGBoost Regressor': xgb.XGBRegressor(random_state=42, n_estimators=100, eval_metric='rmse'),
        'Support Vector Regressor': SVR()
    }

    results = {}
    seed = 42
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    for name, model in models.items():
        try:
            cv_results = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_scores = np.sqrt(cv_results)
            results[name] = {
                'RMSE_Mean': rmse_scores.mean(),
                'RMSE_Std': rmse_scores.std()
            }
            if name == 'XGBoost Regressor':
                global_initial_xgb_rmse = rmse_scores.mean()
            print(f"{name}: Mean RMSE = {results[name]['RMSE_Mean']:.3f} (Std = {results[name]['RMSE_Std']:.3f})")
        except Exception as e:
            results[name] = {'Error': f"Could not run: {e}"}
            print(f"Error running {name}: {e}")

    results_window = tk.Toplevel(root)
    results_window.title("Spot-Checking Results (RMSE)")

    results_text = "Spot-Checking Results (Lower RMSE is Better):\n\n"
    sorted_results = sorted([ (k, v) for k, v in results.items() if 'RMSE_Mean' in v], key=lambda item: item[1]['RMSE_Mean'])

    for name, metrics in sorted_results:
        results_text += f"{name}:\n"
        if 'RMSE_Mean' in metrics:
            results_text += f"  Mean RMSE: {metrics['RMSE_Mean']:.3f}\n"
            results_text += f"  Standard Deviation: {metrics['RMSE_Std']:.3f}\n"
        else:
            results_text += f"  {metrics['Error']}\n"
        results_text += "\n"

    tk.Label(results_window, text=results_text, justify=tk.LEFT).pack(padx=10, pady=10)
    messagebox.showinfo("Spot-Checking Complete", "Spot-checking of algorithms finished. Check the new window for results.")

def tune_xgboost_hyperparameters():
    global X_train_prepared_df, housing, best_xgboost_model, evaluate_button
    if X_train_prepared_df is None or housing is None:
        messagebox.showerror("Error", "Please load and prepare the dataset first.")
        return

    messagebox.showinfo("Info", "Starting XGBoost hyperparameter tuning (GridSearchCV)... This will take some time.")

    X = X_train_prepared_df
    y = housing.loc[X.index, 'median_house_value']

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')

    grid_search = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid,
                               scoring='neg_mean_squared_error',
                               cv=KFold(n_splits=5, shuffle=True, random_state=42),
                               n_jobs=-1,
                               verbose=2)

    try:
        grid_search.fit(X, y)

        best_score_rmse = np.sqrt(-grid_search.best_score_)
        best_params = grid_search.best_params_
        best_xgboost_model = grid_search.best_estimator_

        tuning_results_window = tk.Toplevel(root)
        tuning_results_window.title("XGBoost Hyperparameter Tuning Results")

        results_text = f"XGBoost Hyperparameter Tuning Complete!\n\n"
        results_text += f"Best RMSE (from CV): {best_score_rmse:.3f}\n"
        results_text += f"Best Parameters: {best_params}\n\n"
        results_text += "For more detailed results, check the console output."

        tk.Label(tuning_results_window, text=results_text, justify=tk.LEFT).pack(padx=10, pady=10)
        messagebox.showinfo("Tuning Complete", "XGBoost hyperparameter tuning finished. Check the new window for results and console for GridSearchCV details.")
        evaluate_button.config(state=tk.NORMAL)
    except Exception as e:
        messagebox.showerror("Error", f"Error during XGBoost hyperparameter tuning: {e}")
        print(f"Error during XGBoost hyperparameter tuning: {e}")

def evaluate_model_performance():
    global best_xgboost_model, preprocessor, X_test_global, y_test_global, global_rmse, global_mae, global_r2, summary_button
    if best_xgboost_model is None:
        messagebox.showerror("Error", "Please tune the XGBoost model first.")
        return
    if X_test_global is None or y_test_global is None:
        messagebox.showerror("Error", "Test data not available. Please ensure data was loaded and prepared.")
        return

    messagebox.showinfo("Info", "Evaluating model performance on the test set...")

    try:
        X_test_prepared = preprocessor.transform(X_test_global)
        y_pred = best_xgboost_model.predict(X_test_prepared)

        global_rmse = np.sqrt(mean_squared_error(y_test_global, y_pred))
        global_mae = mean_absolute_error(y_test_global, y_pred)
        global_r2 = r2_score(y_test_global, y_pred)

        evaluation_window = tk.Toplevel(root)
        evaluation_window.title("Model Performance Evaluation")

        results_text = "Model Performance on Test Set:\n\n"
        results_text += f"Root Mean Squared Error (RMSE): {global_rmse:.3f}\n"
        results_text += f"Mean Absolute Error (MAE): {global_mae:.3f}\n"
        results_text += f"R-squared (R²): {global_r2:.3f}\n\n"
        results_text += "Interpretation:\n"
        results_text += f"- RMSE of {global_rmse:.3f} indicates the typical prediction error in USD.\n"
        results_text += f"- MAE of {global_mae:.3f} shows the average absolute error, less sensitive to outliers.\n"
        results_text += f"- R² of {global_r2:.3f} means that {global_r2*100:.1f}% of the variance in median house value is explained by the model."

        tk.Label(evaluation_window, text=results_text, justify=tk.LEFT).pack(padx=10, pady=10)
        messagebox.showinfo("Evaluation Complete", "Model performance evaluation finished. Check the new window for results.")
        summary_button.config(state=tk.NORMAL)
    except Exception as e:
        messagebox.showerror("Error", f"Error during model evaluation: {e}")
        print(f"Error during model evaluation: {e}")

def show_model_summary():
    global global_rmse, global_mae, global_r2, global_initial_xgb_rmse
    if global_rmse is None or global_mae is None or global_r2 is None:
        messagebox.showerror("Error", "Please evaluate the model performance first.")
        return

    summary_window = tk.Toplevel(root)
    summary_window.title("Model Performance Summary & Next Steps")

    summary_text = "## Model Performance Summary\n"
    summary_text += "---\n\n"
    summary_text += "**1. How accurate is the model?**\n"
    summary_text += f"Based on the evaluation on the test set, the model shows good accuracy for predicting house values:\n"
    summary_text += f"- **RMSE (Root Mean Squared Error):** ${global_rmse:.2f}\n"
    summary_text += f"  This represents the typical prediction error of the model, in USD units. A lower RMSE indicates better accuracy.\n"
    summary_text += f"- **MAE (Mean Absolute Error):** ${global_mae:.2f}\n"
    summary_text += f"  This is the average absolute error, less sensitive to outliers. It shows, on average, how much the prediction deviates from the actual value.\n"
    summary_text += f"- **R-squared (R²):** {global_r2:.3f} ({global_r2*100:.1f}%)\n"
    summary_text += f"  This coefficient indicates the proportion of variance in house prices explained by the model. A value of {global_r2*100:.1f}% is a very good indicator of the model's explanatory power. The closer to 1 (or 100%), the better the model explains the variance. A high R² suggests a good fit.\n\n"

    summary_text += "---\n\n"
    summary_text += "**2. How much did the model's performance improve after hyperparameter tuning?**\n"
    if global_initial_xgb_rmse is not None:
        improvement = global_initial_xgb_rmse - global_rmse
        percentage_improvement = (improvement / global_initial_xgb_rmse) * 100 if global_initial_xgb_rmse != 0 else 0
        summary_text += f"Initial XGBoost RMSE (without tuning, from spot-checking): ${global_initial_xgb_rmse:.2f}\n"
        summary_text += f"Optimized XGBoost RMSE (after tuning): ${global_rmse:.2f}\n"
        if improvement > 0:
            summary_text += f"The model's performance improved by approximately **${improvement:.2f}** in RMSE, an improvement of about **{percentage_improvement:.2f}%**.\n"
            summary_text += "This demonstrates that hyperparameter tuning was effective in optimizing the model's ability to make precise predictions."
        else:
            summary_text += f"No significant improvement in RMSE was observed after tuning, or performance even decreased. This might indicate that the default parameters were already good, or that the tuning search space was not extensive/relevant enough."
    else:
        summary_text += "Spot-checking was not performed, or the initial RMSE for XGBoost was not recorded for comparison. Please run spot-checking to obtain a baseline for comparison.\n"

    summary_text += "\n\n---\n\n"
    summary_text += "**3. What can be done to further improve the model's accuracy?**\n"
    summary_text += "- **Advanced Feature Engineering:** Explore more combined attributes or transformations (e.g., logarithmic transformations for highly skewed features, interactions between features). Consider adding external features (e.g., crime rate, proximity to amenities, local economic trends).\n"
    summary_text += "- **More Thorough Hyperparameter Tuning:** Refine the search grid for `GridSearchCV` or use more efficient methods like `RandomizedSearchCV` or Bayesian optimization (with libraries like `Hyperopt` or `Optuna`) to explore a broader hyperparameter space.\n"
    summary_text += "- **Exploring Other Advanced Algorithms:** Test other powerful ensemble models like LightGBM or CatBoost, which can offer superior performance or faster training. For very large datasets, Neural Networks (Deep Learning) can be an option.\n"
    summary_text += "- **Model Ensembling (Stacking/Blending):** Combine predictions from multiple different models to reduce error and improve robustness.\n"
    summary_text += "- **Outlier and Skewness Treatment:** Analyze and explicitly handle outliers and skewed distributions in features, as they can negatively impact model performance.\n"
    summary_text += "- **Collecting More Data:** If feasible, a larger and more diverse dataset can significantly improve model accuracy."

    summary_label = tk.Label(summary_window, text=summary_text, justify=tk.LEFT, wraplength=700)
    summary_label.pack(padx=20, pady=20)

def show_income_house_stats():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    income_stats = housing['median_income'].agg(['max', 'min', 'std', 'median', 'mean'])
    house_value_stats = housing['median_house_value'].agg(['max', 'min', 'std', 'median', 'mean'])
    stats_window = tk.Toplevel(root)
    stats_window.title("Descriptive Statistics")
    tk.Label(stats_window, text=f"Statistics for median_income:\n{income_stats}").pack(padx=10, pady=5)
    tk.Label(stats_window, text=f"\nStatistics for median_house_value:\n{house_value_stats}").pack(padx=10, pady=5)

def show_boxplot(column):
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    if column not in housing.columns:
        messagebox.showerror("Error", f"Column '{column}' not found in the dataset.")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(y=housing[column], ax=ax)
    ax.set_title(f'Box and Whisker Plot for {column.replace("_", " ").title()}')
    display_plot(fig)

def show_histograms():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    columns = ['total_rooms', 'total_bedrooms', 'population', 'households']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, column in enumerate(columns):
        if column not in housing.columns:
            messagebox.showerror("Error", f"Column '{column}' not found in the dataset.")
            return
        row, col = divmod(i, 2)
        sns.histplot(housing[column].dropna(), bins=50, kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Histogram for {column.replace("_", " ").title()}')
    fig.tight_layout()
    display_plot(fig)

def show_correlation_heatmap():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    numeric_cols = housing.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        messagebox.showerror("Error", "No numeric columns found for correlation analysis.")
        return
    correlation_matrix = housing[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix')
    display_plot(fig)

def show_scatter_matrix_plot():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    numeric_cols = housing.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        messagebox.showerror("Error", "Not enough numeric attributes for scatter matrix.")
        return

    scatter_axes = scatter_matrix(housing[numeric_cols], figsize=(10, 8))
    if scatter_axes.size > 0:
        scatter_fig = scatter_axes[0][0].figure
        plt.suptitle('Scatter Matrix', y=1.02)
        display_plot(scatter_fig)
    else:
        messagebox.showerror("Error", "Could not display scatter matrix.")

def show_data_dimensions():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    dimensions = housing.shape
    info_window = tk.Toplevel(root)
    info_window.title("Data Information")
    tk.Label(info_window, text=f"Dataset dimensions: {dimensions}").pack(padx=10, pady=5)

def show_missing_values():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    missing_values = housing.isnull().sum()
    info_window = tk.Toplevel(root)
    info_window.title("Missing Values")
    missing_text = "Number of missing values per column:\n" + missing_values.to_string()
    tk.Label(info_window, text=missing_text).pack(padx=10, pady=5)

def show_correlation_with_target():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    if 'median_house_value' not in housing.columns:
        messagebox.showerror("Error", "Column 'median_house_value' not found.")
        return
    numeric_cols = housing.select_dtypes(include=np.number).columns
    if 'median_house_value' not in numeric_cols:
        messagebox.showerror("Error", "'median_house_value' is not a numeric column.")
        return
    correlation_matrix = housing[numeric_cols].corr(numeric_only=True)
    correlation_with_target = correlation_matrix['median_house_value'].sort_values(ascending=False)
    info_window = tk.Toplevel(root)
    info_window.title("Correlation with Target")
    correlation_text = "Correlation with median_house_value:\n" + correlation_with_target.to_string()
    tk.Label(info_window, text=correlation_text).pack(padx=10, pady=5)

def show_skewness():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    columns = ['total_rooms', 'total_bedrooms', 'population', 'households']
    valid_columns = [col for col in columns if col in housing.columns and pd.api.types.is_numeric_dtype(housing[col])]
    if not valid_columns:
        messagebox.showerror("Error", "No valid numeric columns found for skewness calculation.")
        return
    skewness_values = housing[valid_columns].skew()
    info_window = tk.Toplevel(root)
    info_window.title("Feature Skewness")
    skewness_text = "Skewness for selected columns:\n" + skewness_values.to_string()
    tk.Label(info_window, text=skewness_text).pack(padx=10, pady=5)

def display_plot(fig):
    plot_window = tk.Toplevel(root)
    plot_window.title("Visualization")
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.draw()

def show_prepared_data():
    global X_train_prepared_df
    if X_train_prepared_df is None:
        messagebox.showerror("Error", "Please prepare the data first.")
        return
    prepared_window = tk.Toplevel(root)
    prepared_window.title("Prepared Training Data (First 5 Rows)")
    tree = ttk.Treeview(prepared_window)
    tree["columns"] = list(X_train_prepared_df.columns)
    tree["show"] = "headings"
    for col in X_train_prepared_df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    for index, row in X_train_prepared_df.head().iterrows():
        tree.insert("", tk.END, values=list(row))
    tree.pack(padx=10, pady=10, fill="both", expand=True)

def analyze_data():
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return

    dimensions = housing.shape
    missing_values = housing.isnull().sum()

    correlation_with_target = housing.corr(numeric_only=True)['median_house_value'].abs().sort_values(ascending=False)
    most_influential_feature = correlation_with_target.index[1] if len(correlation_with_target) > 1 else "N/A"
    most_influential_correlation = correlation_with_target.iloc[1] if len(correlation_with_target) > 1 else "N/A"

    skewness = housing[['total_rooms', 'total_bedrooms', 'population', 'households']].skew()

    analysis_window = tk.Toplevel(root)
    analysis_window.title("Data Analysis and Conclusions")

    tk.Label(analysis_window, text=f"Dataset Dimensions: {dimensions}").pack(padx=10, pady=5, anchor="w")
    tk.Label(analysis_window, text="Missing Values per Column:\n" + missing_values.to_string()).pack(padx=10, pady=5, anchor="w")
    tk.Label(analysis_window, text=f"Most Influential Feature on Median House Value: {most_influential_feature} (Correlation: {most_influential_correlation:.2f})").pack(padx=10, pady=5, anchor="w")
    tk.Label(analysis_window, text="Data Skewness:\n" + skewness.to_string()).pack(padx=10, pady=5, anchor="w")

    tk.Label(analysis_window, text="\nFurther Explanations (based on visualizations):").pack(padx=10, pady=5, anchor="w")
    tk.Label(analysis_window, text="- Histograms show data distribution and skewness.").pack(padx=10, pady=5, anchor="w")
    tk.Label(analysis_window, text="- Correlation matrix visualizes linear relationships between features.").pack(padx=10, pady=5, anchor="w")
    tk.Label(analysis_window, text="- Scatter matrix shows pairwise relationships between attributes.").pack(padx=10, pady=5, anchor="w")
    tk.Label(analysis_window, text="- Box and whisker plots highlight distribution and outliers.").pack(padx=10, pady=5, anchor="w")

def load_dataset():
    global housing
    file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if file_path:
        try:
            housing = pd.read_csv(file_path)
            messagebox.showinfo("Success", f"Dataset loaded successfully from: {file_path}")
            enable_buttons()
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    else:
        messagebox.showinfo("Info", "No file selected.")

def enable_buttons():
    for button in exploration_buttons + [analyze_button, preprocess_button]:
        button.config(state=tk.NORMAL)
    spot_checking_button.config(state=tk.DISABLED)
    tune_xgboost_button.config(state=tk.DISABLED)
    evaluate_button.config(state=tk.DISABLED)
    summary_button.config(state=tk.DISABLED)

root = tk.Tk()
root.title("Housing Data Exploration and Model Summary")

load_button = ttk.Button(root, text="Load Dataset", command=load_dataset)
load_button.pack(pady=10, fill="x", padx=10)

button_frame_explore = ttk.LabelFrame(root, text="Data Exploration")
button_frame_explore.pack(padx=10, pady=10, fill="x")

exploration_buttons = [
    ttk.Button(button_frame_explore, text="Show Income & House Value Stats", command=show_income_house_stats, state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Median Income Box Plot", command=lambda: show_boxplot('median_income'), state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Median House Value Box Plot", command=lambda: show_boxplot('median_house_value'), state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Histograms", command=show_histograms, state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Correlation Heatmap", command=show_correlation_heatmap, state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Scatter Matrix", command=show_scatter_matrix_plot, state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Data Dimensions", command=show_data_dimensions, state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Missing Values", command=show_missing_values, state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Correlation with Target", command=show_correlation_with_target, state=tk.DISABLED),
    ttk.Button(button_frame_explore, text="Show Feature Skewness", command=show_skewness, state=tk.DISABLED),
]
for button in exploration_buttons:
    button.pack(pady=5, fill="x")

analyze_button = ttk.Button(button_frame_explore, text="Analyze Data and Draw Conclusions", command=analyze_data, state=tk.DISABLED)
analyze_button.pack(pady=5, fill="x")

preprocess_frame = ttk.LabelFrame(root, text="Data Preprocessing")
preprocess_frame.pack(padx=10, pady=10, fill="x")

preprocess_button = ttk.Button(preprocess_frame, text="Prepare Data", command=prepare_data, state=tk.DISABLED)
preprocess_button.pack(pady=5, fill="x")

show_prepared_button = ttk.Button(preprocess_frame, text="Show Prepared Training Data", command=show_prepared_data, state=tk.DISABLED)
show_prepared_button.pack(pady=5, fill="x")

model_frame = ttk.LabelFrame(root, text="Model Selection & Tuning")
model_frame.pack(padx=10, pady=10, fill="x")

spot_checking_button = ttk.Button(model_frame, text="Perform Spot-Checking", command=perform_spot_checking, state=tk.DISABLED)
spot_checking_button.pack(pady=5, fill="x")

tune_xgboost_button = ttk.Button(model_frame, text="Tune XGBoost Hyperparameters", command=tune_xgboost_hyperparameters, state=tk.DISABLED)
tune_xgboost_button.pack(pady=5, fill="x")

evaluate_button = ttk.Button(model_frame, text="Evaluate Model Performance", command=evaluate_model_performance, state=tk.DISABLED)
evaluate_button.pack(pady=5, fill="x")

summary_button = ttk.Button(model_frame, text="Summarize Model Performance", command=show_model_summary, state=tk.DISABLED)
summary_button.pack(pady=5, fill="x")

root.mainloop()
