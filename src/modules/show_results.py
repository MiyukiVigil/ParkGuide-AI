"""Utility to display and visualize training results"""
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "results"

def load_results():
    """Load training metrics and history"""
    metrics_path = RESULTS_DIR / "metrics.json"
    history_path = RESULTS_DIR / "history.json"
    csv_path = RESULTS_DIR / "history.csv"
    
    metrics = None
    history = None
    csv_data = None
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    
    if csv_path.exists():
        with open(csv_path) as f:
            csv_data = list(csv.DictReader(f))
    
    return metrics, history, csv_data

def print_results():
    """Print training results to console"""
    metrics, history, csv_data = load_results()
    
    if metrics is None:
        print("No results found. Please train first.")
        return
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    print(f"\nBest Val Accuracy: {metrics.get('best_val_acc', 'N/A'):.4f}")
    print(f"Best Val Loss: {metrics.get('best_val_loss', 'N/A'):.4f}")
    print(f"Macro F1 Score: {metrics.get('macro_f1', 'N/A'):.4f}")
    
    print("\n" + "-"*60)
    print("PER-CLASS METRICS")
    print("-"*60)
    
    if 'class_metrics' in metrics:
        for class_name, class_info in metrics['class_metrics'].items():
            print(f"\n{class_name}:")
            print(f"  Precision: {class_info.get('precision', 'N/A'):.4f}")
            print(f"  Recall:    {class_info.get('recall', 'N/A'):.4f}")
            print(f"  F1 Score:  {class_info.get('f1', 'N/A'):.4f}")
    
    print("\n" + "="*60)

def show_plots():
    """Display saved plots"""
    loss_plot = RESULTS_DIR / "loss_curve.png"
    acc_plot = RESULTS_DIR / "accuracy_curve.png"
    cm_plot = RESULTS_DIR / "confusion_matrix.png"
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    if loss_plot.exists():
        img = plt.imread(loss_plot)
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('Loss Curve')
    
    if acc_plot.exists():
        img = plt.imread(acc_plot)
        axes[1].imshow(img)
        axes[1].axis('off')
        axes[1].set_title('Accuracy Curve')
    
    if cm_plot.exists():
        img = plt.imread(cm_plot)
        axes[2].imshow(img)
        axes[2].axis('off')
        axes[2].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()

def generate_report(output_file="training_report.txt"):
    """Generate text report of results"""
    metrics, history, csv_data = load_results()
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PLANT INTERACTION DETECTION - TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        
        if metrics:
            f.write("FINAL RESULTS\n")
            f.write("-"*60 + "\n")
            f.write(f"Best Val Accuracy: {metrics.get('best_val_acc', 'N/A'):.4f}\n")
            f.write(f"Best Val Loss: {metrics.get('best_val_loss', 'N/A'):.4f}\n")
            f.write(f"Macro F1 Score: {metrics.get('macro_f1', 'N/A'):.4f}\n\n")
            
            f.write("PER-CLASS METRICS\n")
            f.write("-"*60 + "\n")
            if 'class_metrics' in metrics:
                for class_name, class_info in metrics['class_metrics'].items():
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {class_info.get('precision', 'N/A'):.4f}\n")
                    f.write(f"  Recall:    {class_info.get('recall', 'N/A'):.4f}\n")
                    f.write(f"  F1 Score:  {class_info.get('f1', 'N/A'):.4f}\n")
    
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    print_results()
