import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from pathlib import Path
import platform

class ModernFrame(ttk.Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(padding="20")
        self.grid(sticky="nsew")

class InputGroup(ttk.LabelFrame):
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, text=text, padding="10", **kwargs)
        self.configure(style='Modern.TLabelframe')
        
class MLPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VSC-HVDC Predictor")
        self.root.geometry("1280x800")
        self.setup_styles()
        
        # Configure grid weight
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Load the model and scaler
        try:
            with open('xgb_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('feature_ranges.json', 'r') as f:
                self.feature_ranges = json.load(f)
        except FileNotFoundError:
            messagebox.showerror("Error", "Model files not found. Please ensure model files are in the same directory.")
            root.destroy()
            return

        # Create main container with padding
        self.main_container = ModernFrame(root)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # Create header
        header_frame = ttk.Frame(self.main_container)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        ttk.Label(header_frame, text="VSC-HVDC Parameter Prediction", 
                 style='Header.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Separator(header_frame, orient='horizontal').grid(row=1, column=0, sticky="ew", pady=(10, 0))

        # Create paned window for input and output
        paned = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew")
        self.main_container.grid_rowconfigure(1, weight=1)

        # Left panel for inputs
        left_panel = ttk.Frame(paned)
        
        # Create canvas and scrollbar for inputs
        canvas = tk.Canvas(left_panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Grid layout for scroll components
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        left_panel.grid_columnconfigure(0, weight=1)
        left_panel.grid_rowconfigure(0, weight=1)

        # Right panel for output
        right_panel = ttk.Frame(paned)
        
        # Add panels to paned window
        paned.add(left_panel, weight=2)
        paned.add(right_panel, weight=1)

        # Create input fields
        self.input_vars = {}
        self.create_input_sections()

        # Create predict button
        predict_btn = ttk.Button(right_panel, text="Generate Prediction", 
                               command=self.predict, style='Accent.TButton')
        predict_btn.grid(row=0, column=0, pady=20, padx=20, sticky="ew")

        # Create output display area
        output_frame = ttk.LabelFrame(right_panel, text="Prediction Results", padding="10")
        output_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self.output_text = tk.Text(output_frame, wrap=tk.WORD, font=('Segoe UI', 10))
        self.output_text.grid(row=0, column=0, sticky="nsew")
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)

        # Configure mouse wheel scrolling
        self.bind_mousewheel(canvas)

    def setup_styles(self):
        style = ttk.Style()
        
        # Set theme
        if platform.system() == 'Windows':
            style.theme_use('winnative')
        
        # Configure colors
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Group.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Modern.TLabelframe', font=('Segoe UI', 10))
        style.configure('Modern.TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
        
        # Configure button styles
        style.configure('Accent.TButton', font=('Segoe UI', 10))
        
        # Configure entry style
        style.configure('Modern.TEntry', padding=5)

    def bind_mousewheel(self, widget):
        def _on_mousewheel(event):
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        widget.bind_all("<MouseWheel>", _on_mousewheel)

    def create_input_sections(self):
        # Group features by type
        groups = {
            "Power Settings": ["scenario_P1", "scenario_Qg1", "scenario_Qg2"],
            "AC1 Nominal": [f"scenario_AC1_Nom_{i}" for i in range(1, 7)],
            "AC1 Tolerances": {
                "Minimum": [f"scenario_AC1_MinTol_{i}" for i in range(1, 7)],
                "Maximum": [f"scenario_AC1_MaxTol_{i}" for i in range(1, 7)]
            },
            "AC2 Nominal": [f"scenario_AC2_Nom_{i}" for i in range(1, 7)],
            "AC2 Tolerances": {
                "Minimum": [f"scenario_AC2_MinTol_{i}" for i in range(1, 7)],
                "Maximum": [f"scenario_AC2_MaxTol_{i}" for i in range(1, 7)]
            },
            "DC Settings": {
                "Nominal": ["scenario_DC_Nom_1", "scenario_DC_Nom_2"],
                "Minimum": ["scenario_DC_MinTol_1", "scenario_DC_MinTol_2"],
                "Maximum": ["scenario_DC_MaxTol_1", "scenario_DC_MaxTol_2"]
            }
        }

        row = 0
        for group_name, features in groups.items():
            group_frame = InputGroup(self.scrollable_frame, text=group_name)
            group_frame.grid(row=row, column=0, sticky="ew", padx=10, pady=5)
            self.scrollable_frame.grid_columnconfigure(0, weight=1)
            
            if isinstance(features, dict):
                # Handle nested groups
                inner_row = 0
                for subgroup_name, subfeatures in features.items():
                    ttk.Label(group_frame, text=subgroup_name, style='Group.TLabel').grid(
                        row=inner_row, column=0, columnspan=2, pady=(5, 5), sticky="w")
                    inner_row += 1
                    inner_row = self.create_input_fields(subfeatures, group_frame, inner_row)
            else:
                # Handle flat groups
                self.create_input_fields(features, group_frame, 0)
            
            row += 1

    def create_input_fields(self, features, parent, start_row):
        row = start_row
        for feature in features:
            if feature in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature]['min'], self.feature_ranges[feature]['max']
                
                # Create input frame
                input_frame = ttk.Frame(parent)
                input_frame.grid(row=row, column=0, sticky="ew", pady=2)
                input_frame.grid_columnconfigure(1, weight=1)
                
                # Create label
                ttk.Label(input_frame, text=feature.split('scenario_')[1]).grid(
                    row=0, column=0, padx=(5, 10), sticky="w")
                
                # Create entry with default value
                var = tk.StringVar(value=str((min_val + max_val) / 2))
                entry = ttk.Entry(input_frame, textvariable=var, style='Modern.TEntry')
                entry.grid(row=0, column=1, sticky="ew", padx=5)
                
                # Add tooltip with range information
                self.create_tooltip(entry, f"Valid range: {min_val} to {max_val}")
                
                self.input_vars[feature] = var
                row += 1
        
        return row

    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0", padding=5)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            
        widget.bind('<Enter>', show_tooltip)

    def predict(self):
        try:
            # Collect input values
            input_data = {}
            for feature, var in self.input_vars.items():
                try:
                    value = float(var.get())
                    min_val = self.feature_ranges[feature]['min']
                    max_val = self.feature_ranges[feature]['max']
                    if not min_val <= value <= max_val:
                        raise ValueError(f"{feature.split('scenario_')[1]} must be between {min_val} and {max_val}")
                    input_data[feature] = value
                except ValueError as e:
                    messagebox.showerror("Invalid Input", str(e))
                    return

            # Create input DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.model.predict(input_df)
            
            # Display results
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Predicted Values:\n\n")
            
            # Format each prediction
            for i, col in enumerate(["N_vw1", "N_vw2", "Qcab1_interm", "Qcab2_interm",
                                   "Qf1_interm", "Qf2", "Rdc", "Vac1", "Vac2", "Vdc1",
                                   "Xtr1_pu", "Xtr2_pu", "ucab1", "ucab2", "uf1",
                                   "uf2", "w1", "w2"]):
                formatted_value = f"{prediction[0][i]:.4f}"
                self.output_text.insert(tk.END, f"{col}: {formatted_value}\n")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = MLPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()