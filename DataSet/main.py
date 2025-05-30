import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

housing = None
exploration_buttons = []
preprocess_button = None
show_prepared_button = None
analyze_button = None
root = None
preprocessor = None
X_train_prepared_df = None

def load_dataset():
    global housing
    file_path = filedialog.askopenfilename(title="Select Dataset File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if file_path:
        try:
            housing = pd.read_csv(file_path)
            messagebox.showinfo("Success", f"Dataset loaded successfully from: {file_path}")
            enable_exploration_buttons()
        except FileNotFoundError:
            messagebox.showerror("Error", "File not found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    else:
        messagebox.showinfo("Info", "No file selected.")

def enable_exploration_buttons():
    for button in exploration_buttons + [analyze_button]:
        button.config(state=tk.NORMAL)
    preprocess_button.config(state=tk.NORMAL)

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

def prepare_data():
    global housing, X_train_prepared_df, preprocessor
    if housing is None:
        messagebox.showerror("Error", "Please load the dataset first.")
        return
    try:
        X = housing.drop('median_house_value', axis=1, errors='ignore')
        y = housing['median_house_value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = ['ocean_proximity']
        if 'longitude' in numerical_features:
            numerical_features.remove('longitude')
        if 'latitude' in numerical_features:
            numerical_features.remove('latitude')

        class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
            def __init__(self, add_bedrooms_per_room=True):
                self.add_bedrooms_per_room = add_bedrooms_per_room
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                try:
                    rooms_ix = list(numerical_features).index('total_rooms')
                    bedrooms_ix = list(numerical_features).index('total_bedrooms')
                    population_ix = list(numerical_features).index('population')
                    households_ix = list(numerical_features).index('households')
                except ValueError as e:
                    print(f"Error in CombinedAttributesAdder: Missing numeric attribute: {e}")
                    return X
                rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                population_per_household = X[:, population_ix] / X[:, households_ix]
                if self.add_bedrooms_per_room:
                    bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                    return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
                else:
                    return np.c_[X, rooms_per_household, population_per_household]

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

        original_categories = preprocessor.named_transformers_['cat']['onehot'].categories_[0]
        new_categorical_feature_names = [f"ocean_proximity_{cat.replace('<', '').replace('>', '')}" for cat in original_categories]

        feature_names_prepared_numerical = list(numerical_features) + ['rooms_per_household', 'population_per_household', 'bedrooms_per_room']
        feature_names_prepared = feature_names_prepared_numerical + new_categorical_feature_names

        X_train_prepared_df = pd.DataFrame(X_train_prepared, columns=feature_names_prepared, index=X_train.index)
        messagebox.showinfo("Succes", "Data preprocessing has been completed..")
        preprocess_button.config(state=tk.DISABLED)
        show_prepared_button.config(state=tk.NORMAL)
    except Exception as e:
        messagebox.showerror("Eroare", f"Error during data preprocessing: {e}")

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

root = tk.Tk()
root.title("Housing Data Exploration and Preprocessing")

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

root.mainloop()
