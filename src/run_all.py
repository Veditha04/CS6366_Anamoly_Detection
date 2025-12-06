"""
Main script to run the complete MVTec anomaly detection pipeline
"""

import os
import sys
import argparse
import time
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Run MVTec Anomaly Detection Pipeline')
    parser.add_argument('--train-baseline', action='store_true', help='Train baseline model')
    parser.add_argument('--train-multiscale', action='store_true', help='Train multi-scale model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate both models')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline (train both + evaluate)')
    parser.add_argument('--quick', action='store_true', help='(Currently informational only) Intended: run with fewer epochs')
    
    args = parser.parse_args()
    
    # Set project paths relative to this file
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SRC_DIR = os.path.join(PROJECT_ROOT, "src")
    
    # Add src to Python path
    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)
    
    # Start timing
    start_time = time.time()
    
    print_header("MVTEC ANOMALY DETECTION PIPELINE")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Arguments: {vars(args)}")
    print()
    
    # If no arguments provided, run complete pipeline
    if not any([args.train_baseline, args.train_multiscale, args.evaluate, args.all]):
        print("No arguments provided. Running complete pipeline...")
        args.all = True
    
    try:
        # ===== TRAIN BASELINE =====
        if args.all or args.train_baseline:
            print_header("TRAINING BASELINE AUTOENCODER")
            from train_baseline_enhanced import main as train_baseline
            # NOTE: args.quick is currently informational; epochs are fixed in train_baseline_enhanced.py
            train_baseline()
        
        # ===== TRAIN MULTI-SCALE =====
        if args.all or args.train_multiscale:
            print_header("TRAINING MULTI-SCALE AUTOENCODER")
            from train_multiscale_enhanced import main as train_multiscale
            # NOTE: args.quick is currently informational; epochs are fixed in train_multiscale_enhanced.py
            train_multiscale()
        
        # ===== EVALUATE MODELS =====
        if args.all or args.evaluate:
            print_header("EVALUATING MODELS")
            
            models_dir = os.path.join(PROJECT_ROOT, "models")
            baseline_model = os.path.join(models_dir, "baseline_ae_best_enhanced.pth")
            multiscale_model = os.path.join(models_dir, "multiscale_ae_best_enhanced.pth")
            
            if not os.path.exists(baseline_model):
                print(f"Warning: Baseline model not found at {baseline_model}")
                print("Training baseline model first...")
                from train_baseline_enhanced import main as train_baseline
                train_baseline()
            
            if not os.path.exists(multiscale_model):
                print(f"Warning: Multi-scale model not found at {multiscale_model}")
                print("Training multi-scale model first...")
                from train_multiscale_enhanced import main as train_multiscale
                train_multiscale()
            
            from evaluate_models import main as evaluate_models
            evaluate_models()
        
        # ===== FINAL SUMMARY =====
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # Show where results are stored
        print("\n" + "="*70)
        print(" RESULTS LOCATION")
        print("="*70)
        print(f"Models directory:")
        print(f"  {os.path.join(PROJECT_ROOT, 'models')}")
        print(f"    - baseline_ae_best_enhanced.pth")
        print(f"    - multiscale_ae_best_enhanced.pth")
        print(f"\nTraining plots:")
        print(f"  Baseline   : {os.path.join(PROJECT_ROOT, 'results', 'baseline')}")
        print(f"  Multi-scale: {os.path.join(PROJECT_ROOT, 'results', 'multiscale')}")
        print(f"\nComparison & metrics:")
        print(f"  {os.path.join(PROJECT_ROOT, 'results', 'comparison')}")
        
    except Exception as e:
        print(f"\n{'!'*70}")
        print(" ERROR DURING EXECUTION")
        print(f"{'!'*70}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        
        import traceback
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
