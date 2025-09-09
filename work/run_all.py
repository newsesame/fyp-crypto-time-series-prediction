#!/usr/bin/env python3
"""
Refactored main program: demonstrates how to use the new directory structure
"""

import torch
import pandas as pd

def run_regression_comparison():
    """Execute regression task comparison"""
    print("=" * 60)
    print("Executing Regression Task Comparison (Transformer vs GRU)")
    print("=" * 60)
    
    try:
        from regression.compare import evaluate_models
        results = evaluate_models()
        print("\nRegression Task Results:")
        print("\nModel Information:")
        print(results['model_info'].to_string(index=False))
        print("\nModel Performance:")
        print(results['metrics'].to_string(index=False))
        return results
    except Exception as e:
        print(f"Regression task execution failed: {e}")
        return None

def run_classification_comparison():
    """Execute classification task comparison"""
    print("\n" + "=" * 60)
    print("Executing Classification Task Comparison (Transformer vs GRU)")
    print("=" * 60)
    
    try:
        from classification.compare_classification import evaluate_classification_models
        results = evaluate_classification_models()
        print("\nClassification Task Results:")
        print("\nModel Information:")
        print(results['model_info'].to_string(index=False))
        print("\nModel Performance:")
        print(results['metrics'].to_string(index=False))
        return results
    except Exception as e:
        print(f"Classification task execution failed: {e}")
        return None

def main():
    """Main program"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Execute regression task
    regression_results = run_regression_comparison()
    
    # Execute classification task
    classification_results = run_classification_comparison()
    
    print("\n" + "=" * 60)
    print("All tasks completed")
    print("=" * 60)
    
    return {
        'regression': regression_results,
        'classification': classification_results
    }

if __name__ == '__main__':
    main()
