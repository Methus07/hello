import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    entry_path.delete(0, tk.END)
    entry_path.insert(tk.END, file_path)

def load_data_and_classify():
    file_path = entry_path.get()
    df = pd.read_csv(file_path)

    # Assuming Outcome is the target column
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize SVM classifier
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = svm_classifier.predict(X_test)

    # Display accuracy
    accuracy = accuracy_score(y_test, predictions)
    lbl_accuracy.config(text=f"Accuracy: {accuracy:.2f}")

    # Display data visualization
    sns.pairplot(df, hue='Outcome')
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Diabetes Data SVM Classifier")

# File path entry
entry_path = tk.Entry(root, width=50)
entry_path.grid(row=0, column=0, padx=10, pady=10)

# Browse button
btn_browse = tk.Button(root, text="Browse", command=browse_file)
btn_browse.grid(row=0, column=1, padx=10, pady=10)

# Load and classify button
btn_load_classify = tk.Button(root, text="Load Data and Classify", command=load_data_and_classify)
btn_load_classify.grid(row=1, column=0, columnspan=2, pady=10)

# Accuracy label
lbl_accuracy = tk.Label(root, text="")
lbl_accuracy.grid(row=2, column=0, columnspan=2)

root.mainloop()
