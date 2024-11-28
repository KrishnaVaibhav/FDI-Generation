import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QHBoxLayout, QFrame, QCheckBox, QListWidget, QListWidgetItem, QMessageBox
from PyQt5.QtGui import QIcon
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from xgboost import XGBRegressor as XGB
from lightgbm import LGBMRegressor as LGBM
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.multioutput import MultiOutputRegressor as MOR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

def handle_inf_values(df):
    for col in df.columns:
        df.loc[df[col] == np.inf, col] = df[col].mean()  # Replace inf with mean (adjustable)
        df.loc[df[col] == -np.inf, col] = -df[col].mean()  # Replace -inf with negative of mean
    return df

def train_model(file, selected_columns, method_selection2, n_jobs):
    data = pd.read_csv(file)
    data = handle_inf_values(data.copy())
    X = data[selected_columns]
    y = data.drop(selected_columns, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if method_selection2 == "XGBoost":
        model = MOR(XGB(n_estimators=100, max_depth=5, n_jobs=n_jobs))
        model.fit(X_train, y_train)
    
    if method_selection2 == "LightGBM":
        model = MOR(LGBM(n_estimators=100, max_depth=5, n_jobs=n_jobs))
        model.fit(X_train, y_train)
        
    if method_selection2 == "RandomForest":
        model = MOR(RF(n_estimators=100, max_depth=5, n_jobs=n_jobs))
        model.fit(X_train, y_train)
        
    with open('model.pickle', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, mae, r2

def add_gaussian_noise(df, noise_level):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    noisy_df = df.copy()
    noisy_df[numeric_cols] += np.random.randn(*df[numeric_cols].shape) * noise_level
    return noisy_df

def add_uniform_noise(df, noise_level):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    noisy_df = df.copy()
    noisy_df[numeric_cols] += np.random.uniform(-noise_level, noise_level, size=df[numeric_cols].shape)
    return noisy_df

def smote(file, col, noise_type=None, noise_level=0.1):
    data = pd.read_csv(file)
    X = data.drop(col, axis=1)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.max().max(), inplace=True)
    y = data[col]

    ros = RandomOverSampler(sampling_strategy='not minority')
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_resampled.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_resampled.fillna(X_resampled.max().max(), inplace=True)

    smote = SMOTE(sampling_strategy='minority')
    X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

    n_original = len(y)
    X_synthetic = X_resampled[n_original:]
    y_synthetic = y_resampled[n_original:]

    if noise_type == "Gaussian":
        X_synthetic = add_gaussian_noise(X_synthetic, noise_level)
    elif noise_type == "Uniform":
        X_synthetic = add_uniform_noise(X_synthetic, noise_level)

    synthetic_data = pd.concat([X_synthetic, y_synthetic], axis=1)
    synthetic_data.to_csv("synthetic_data_SMOTE.csv", index=False)

def smote_enn(file, col, noise_type=None, noise_level=0.1):
    data = pd.read_csv(file)
    X = data.drop(col, axis=1)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.max().max(), inplace=True)
    y = data[col]

    smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    n_original = len(y)
    X_synthetic = X_resampled[n_original:]
    y_synthetic = y_resampled[n_original:]

    if noise_type == "Gaussian":
        X_synthetic = add_gaussian_noise(X_synthetic, noise_level)
    elif noise_type == "Uniform":
        X_synthetic = add_uniform_noise(X_synthetic, noise_level)

    synthetic_data = pd.concat([X_synthetic, y_synthetic], axis=1)
    synthetic_data.to_csv("synthetic_data_SMOTE_ENN.csv", index=False)

class FDIDataGenerator(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()

        # Left section layout
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Dataset with FDI data", self))
        self.input_file1 = self.create_load_data_section(left_layout, "Load input data file 1", self.load_file1)
        self.marker_column = QComboBox(self)
        left_layout.addWidget(QLabel("Select marker column"))
        left_layout.addWidget(self.marker_column)
        self.method_selection = QComboBox(self)
        self.method_selection.addItems(["SMOTE", "SMOTE ENN"])
        left_layout.addWidget(QLabel("Select method"))
        left_layout.addWidget(self.method_selection)

        self.noise_checkbox = QCheckBox("Add Noise", self)
        self.noise_checkbox.stateChanged.connect(self.toggle_noise_options)
        left_layout.addWidget(self.noise_checkbox)

        self.noise_type = QComboBox(self)
        self.noise_type.addItems(["Gaussian", "Uniform"])
        self.noise_type.setEnabled(False)
        left_layout.addWidget(QLabel("Select noise type"))
        left_layout.addWidget(self.noise_type)

        self.noise_level = QLineEdit(self)
        self.noise_level.setPlaceholderText("Noise level (e.g., 0.1)")
        self.noise_level.setEnabled(False)
        left_layout.addWidget(QLabel("Set noise level"))
        left_layout.addWidget(self.noise_level)

        generate_button = QPushButton("Generate Synthetic Data", self)
        generate_button.clicked.connect(self.generate_synthetic_data)
        left_layout.addWidget(generate_button)

        # Right section layout (modified)
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Dataset without FDI data", self))
        self.input_file2 = self.create_load_data_section(right_layout, "Load input data file 2", self.load_file2)
        self.method_selection2 = QComboBox(self)
        self.method_selection2.addItems(["RandomForest", "XGBoost", "LightGBM"])
        right_layout.addWidget(QLabel("Select method"))
        right_layout.addWidget(self.method_selection2)

        self.columns2 = QListWidget(self)
        self.columns2.setSelectionMode(QListWidget.MultiSelection)
        self.columns2.itemSelectionChanged.connect(self.display_selected_columns)
        right_layout.addWidget(QLabel("Select Independent variable columns"))
        right_layout.addWidget(self.columns2)

        self.selected_columns_input = QLineEdit(self)
        self.selected_columns_input.setReadOnly(True)
        right_layout.addWidget(QLabel("Selected columns"))
        right_layout.addWidget(self.selected_columns_input)

        right_layout.addWidget(QLabel("Select number of cores"))
        self.all_cores_checkbox = QCheckBox("Use all cores", self)
        self.all_cores_checkbox.stateChanged.connect(self.toggle_cores_input)
        right_layout.addWidget(self.all_cores_checkbox)

        self.n_jobs_input = QLineEdit(self)
        self.n_jobs_input.setPlaceholderText("Number of cores (e.g., 4)")
        right_layout.addWidget(self.n_jobs_input)

        generate_button2 = QPushButton("Train Model", self)
        generate_button2.clicked.connect(self.generate_synthetic_data2)
        right_layout.addWidget(generate_button2)

        # Add borders around each column
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Box)
        left_frame.setLayout(left_layout)

        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.Box)
        right_frame.setLayout(right_layout)

        main_layout.addWidget(left_frame)
        main_layout.addWidget(right_frame)
        self.setLayout(main_layout)
        self.setWindowTitle("FDI Data Generator")
        self.setGeometry(100, 100, 1200, 400)
        self.setWindowIcon(QIcon('./icon.png'))  # Set the path to your icon file here

    def create_load_data_section(self, layout, label_text, load_file_method):
        frame = QFrame()
        frame_layout = QVBoxLayout()

        label = QLabel(label_text)
        label.setStyleSheet("font-weight: bold; text-align: center;")

        frame_layout.addWidget(label)
        input_file = QLineEdit(self)
        input_file.setReadOnly(True)
        browse_button = QPushButton("Browse", self)
        browse_button.clicked.connect(load_file_method)
        file_layout = QHBoxLayout()
        file_layout.addWidget(input_file)
        file_layout.addWidget(browse_button)
        frame_layout.addLayout(file_layout)

        frame.setLayout(frame_layout)
        layout.addWidget(frame)

        return input_file

    def load_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select input data file 1')
        if file_path:
            self.input_file1.setText(file_path)
            print(f"File 1 loaded: {file_path}")
            data = pd.read_csv(file_path)
            columns = data.columns.tolist()
            self.marker_column.clear()
            self.marker_column.addItems(columns)

    def load_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select input data file 2')
        if file_path:
            self.input_file2.setText(file_path)
            print(f"File 2 loaded: {file_path}")
            data = pd.read_csv(file_path)
            columns = data.columns.tolist()
            self.columns2.clear()
            self.columns2.addItems(columns)

    def toggle_noise_options(self):
        is_checked = self.noise_checkbox.isChecked()
        self.noise_type.setEnabled(is_checked)
        self.noise_level.setEnabled(is_checked)

    def toggle_cores_input(self):
        is_checked = self.all_cores_checkbox.isChecked()
        self.n_jobs_input.setEnabled(not is_checked)

    def display_selected_columns(self):
        selected_columns = [item.text() for item in self.columns2.selectedItems()]
        self.selected_columns_input.setText(', '.join(selected_columns) if selected_columns else 'None')

    def generate_synthetic_data(self):
        file_path = self.input_file1.text()
        marker_col = self.marker_column.currentText()
        method = self.method_selection.currentText()
        noise_type = self.noise_type.currentText() if self.noise_checkbox.isChecked() else None
        noise_level = float(self.noise_level.text()) if self.noise_checkbox.isChecked() else 0.1

        if method == "SMOTE":
            smote(file_path, marker_col, noise_type, noise_level)
        elif method == "SMOTE ENN":
            smote_enn(file_path, marker_col, noise_type, noise_level)
        print(f"Synthetic data generated using {method} for marker column {marker_col} with noise type {noise_type} and noise level {noise_level}")

    def generate_synthetic_data2(self):
        file_path = self.input_file2.text()
        method = self.method_selection2.currentText()
        selected_columns = [item.text() for item in self.columns2.selectedItems()]
        n_jobs = -1 if self.all_cores_checkbox.isChecked() else int(self.n_jobs_input.text())
        
        if not selected_columns:
            QMessageBox.warning(self, "Warning", "Please select at least one independent variable column.")
            return
        
        mse, mae, r2 = train_model(file_path, selected_columns, method, n_jobs)
        
        QMessageBox.information(self, "Model Results", f"MSE: {mse}\nMAE: {mae}\nR2: {r2}")
        print(f"Model Trained using {method}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FDIDataGenerator()
    ex.show()
    sys.exit(app.exec_())
