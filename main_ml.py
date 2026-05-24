# main_ml.py
import tkinter as tk
from ml_gui.main_window import MainWindowML

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindowML(root)
    root.mainloop()