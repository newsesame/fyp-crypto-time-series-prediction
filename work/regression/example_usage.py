#!/usr/bin/env python3
"""Example usage of the regression comparison and plotting functionality

This script demonstrates how to use the enhanced compare.py module
to run model comparisons and generate various types of plots.
"""

from compare import (
    evaluate_models, main, run_with_plots, run_without_plots,
    plot_gru_vs_real, plot_transformer_vs_real, plot_all_models_comparison
)


def example_basic_comparison():
    """Example: Basic model comparison without plots"""
    print("=" * 60)
    print("Example 1: Basic Model Comparison (No Plots)")
    print("=" * 60)
    
    # Run comparison without plots
    results = evaluate_models(return_raw_results=False)
    
    print("Model Performance Summary:")
    print(results['metrics'].to_string(index=False))
    
    return results


def example_comparison_with_plots():
    """Example: Model comparison with plots (display only)"""
    print("\n" + "=" * 60)
    print("Example 2: Model Comparison with Plots (Display Only)")
    print("=" * 60)
    
    # Run comparison with plots but don't save them
    main(generate_plots=True, save_plots=False)


def example_comparison_with_saved_plots():
    """Example: Model comparison with plots saved to files"""
    print("\n" + "=" * 60)
    print("Example 3: Model Comparison with Saved Plots")
    print("=" * 60)
    
    # Run comparison with plots and save them
    main(generate_plots=True, save_plots=True, output_dir="example_images")


def example_custom_plotting():
    """Example: Custom plotting with individual functions"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Plotting with Individual Functions")
    print("=" * 60)
    
    # Get raw results for custom plotting
    results = evaluate_models(return_raw_results=True)
    raw_results = results['raw_results']
    
    # Generate individual plots
    print("Generating GRU vs Real plot...")
    plot_gru_vs_real(raw_results['gru'], save_path="custom_gru_vs_real.png")
    
    print("Generating Transformer vs Real plot...")
    plot_transformer_vs_real(raw_results['transformer'], save_path="custom_transformer_vs_real.png")
    
    print("Generating All Models Comparison plot...")
    plot_all_models_comparison(
        raw_results['gru'], 
        raw_results['transformer'],
        save_path="custom_all_models_comparison.png"
    )


def example_convenience_functions():
    """Example: Using convenience functions"""
    print("\n" + "=" * 60)
    print("Example 5: Using Convenience Functions")
    print("=" * 60)
    
    print("Running comparison with plots (saved to 'convenience_images' folder)...")
    run_with_plots(output_dir="convenience_images")
    
    print("\nRunning comparison without plots...")
    run_without_plots()


def main_example():
    """Main example function that runs all examples"""
    print("Bitcoin Price Prediction - Plotting Examples")
    print("=" * 80)
    
    try:
        # Example 1: Basic comparison
        example_basic_comparison()
        
        # Example 2: Comparison with plots (display only)
        # Uncomment the line below to see plots displayed
        # example_comparison_with_plots()
        
        # Example 3: Comparison with saved plots
        example_comparison_with_saved_plots()
        
        # Example 4: Custom plotting
        example_custom_plotting()
        
        # Example 5: Convenience functions
        example_convenience_functions()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("Check the generated image folders for saved plots.")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main_example()
