#!/usr/bin/env python3
"""
Model evaluation script for goalkeeper detection
Provides detailed metrics and analysis of model performance
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def evaluate_on_dataset(
    model_path: str = "models/goalkeeper_model",
    dataset_name: str = "goalkeeper_detection",
    output_file: str = "evaluation_results.json"
) -> Dict:
    """
    Evaluate model on a Prodigy dataset
    
    Args:
        model_path: Path to the trained model
        dataset_name: Name of the Prodigy dataset
        output_file: File to save evaluation results
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {dataset_name}")
    
    # Export dataset for evaluation
    temp_export = "temp_eval_data.jsonl"
    export_cmd = [
        "uv", "run", "python", "-m", "prodigy",
        "db-out", dataset_name, temp_export
    ]
    
    try:
        subprocess.run(export_cmd, capture_output=True)
        
        # Run evaluation
        eval_cmd = [
            "uv", "run", "python", "-m", "prodigy",
            "image-classification.evaluate",
            dataset_name,
            model_path,
            "--label", "GOALKEEPER"
        ]
        
        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse evaluation output
            output_lines = result.stdout.strip().split('\n')
            
            # Extract metrics from output
            metrics = {
                "model": model_path,
                "dataset": dataset_name,
                "raw_output": result.stdout
            }
            
            # Parse specific metrics if available
            for line in output_lines:
                if "Accuracy:" in line:
                    metrics["accuracy"] = float(line.split(":")[-1].strip().rstrip('%')) / 100
                elif "Precision:" in line:
                    metrics["precision"] = float(line.split(":")[-1].strip().rstrip('%')) / 100
                elif "Recall:" in line:
                    metrics["recall"] = float(line.split(":")[-1].strip().rstrip('%')) / 100
                elif "F1:" in line:
                    metrics["f1_score"] = float(line.split(":")[-1].strip().rstrip('%')) / 100
            
            # Save results
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\nEvaluation results saved to: {output_file}")
            
            # Clean up
            Path(temp_export).unlink(missing_ok=True)
            
            return metrics
            
        else:
            print(f"Evaluation error: {result.stderr}")
            return {"error": result.stderr}
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {"error": str(e)}

def cross_validate(
    dataset_name: str = "goalkeeper_detection",
    output_dir: str = "cv_models",
    n_folds: int = 5,
    batch_size: int = 16,
    num_epochs: int = 20
) -> List[Dict]:
    """
    Perform k-fold cross-validation
    
    Args:
        dataset_name: Name of the Prodigy dataset
        output_dir: Directory to save CV models
        n_folds: Number of folds
        batch_size: Training batch size
        num_epochs: Number of epochs per fold
    
    Returns:
        List of evaluation results for each fold
    """
    
    print(f"Starting {n_folds}-fold cross-validation")
    print(f"Dataset: {dataset_name}")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    cv_results = []
    
    for fold in range(n_folds):
        print(f"\nTraining fold {fold + 1}/{n_folds}")
        
        model_name = f"cv_model_fold_{fold + 1}"
        model_path = output_path / model_name
        
        # Train model for this fold
        train_cmd = [
            "uv", "run", "python", "-m", "prodigy",
            "train",
            str(model_path),
            "--image-classification", dataset_name,
            "--label-scheme", "binary",
            "--batch-size", str(batch_size),
            "--n-iter", str(num_epochs),
            "--eval-split", f"0.{n_folds}",
            "--cv-fold", str(fold)
        ]
        
        try:
            subprocess.run(train_cmd, capture_output=True)
            
            # Evaluate this fold
            fold_results = evaluate_on_dataset(
                str(model_path),
                dataset_name,
                f"cv_fold_{fold + 1}_results.json"
            )
            
            fold_results["fold"] = fold + 1
            cv_results.append(fold_results)
            
        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")
            cv_results.append({"fold": fold + 1, "error": str(e)})
    
    # Calculate average metrics
    valid_results = [r for r in cv_results if "error" not in r]
    
    if valid_results:
        avg_metrics = {
            "n_folds": n_folds,
            "valid_folds": len(valid_results),
            "avg_accuracy": sum(r.get("accuracy", 0) for r in valid_results) / len(valid_results),
            "avg_precision": sum(r.get("precision", 0) for r in valid_results) / len(valid_results),
            "avg_recall": sum(r.get("recall", 0) for r in valid_results) / len(valid_results),
            "avg_f1": sum(r.get("f1_score", 0) for r in valid_results) / len(valid_results),
            "fold_results": cv_results
        }
        
        # Save CV results
        cv_output_file = "cross_validation_results.json"
        with open(cv_output_file, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        print(f"\nCross-validation complete!")
        print(f"Results saved to: {cv_output_file}")
        print(f"\nAverage metrics across {len(valid_results)} folds:")
        print(f"- Accuracy: {avg_metrics['avg_accuracy']:.3f}")
        print(f"- Precision: {avg_metrics['avg_precision']:.3f}")
        print(f"- Recall: {avg_metrics['avg_recall']:.3f}")
        print(f"- F1 Score: {avg_metrics['avg_f1']:.3f}")
        
        return cv_results
    
    else:
        print("No valid results from cross-validation")
        return cv_results

def analyze_errors(
    model_path: str = "models/goalkeeper_model",
    dataset_name: str = "goalkeeper_detection",
    output_dir: str = "error_analysis"
) -> Dict:
    """
    Analyze model errors and save misclassified examples
    
    Args:
        model_path: Path to the model
        dataset_name: Name of the dataset
        output_dir: Directory to save error analysis
    
    Returns:
        Dictionary with error analysis results
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Analyzing model errors...")
    
    # Create a review interface for errors
    review_cmd = [
        "uv", "run", "python", "-m", "prodigy",
        "review",
        f"{dataset_name}_errors",
        dataset_name,
        "--label", "GOALKEEPER",
        "--auto-accept"
    ]
    
    try:
        # Export predictions for analysis
        pred_file = output_path / "predictions_for_analysis.jsonl"
        
        # Get predictions on the dataset
        predict_cmd = [
            "uv", "run", "python", "-m", "prodigy",
            "image-classification.correct",
            f"{dataset_name}_predictions",
            model_path,
            dataset_name,
            "--label", "GOALKEEPER"
        ]
        
        subprocess.run(predict_cmd, capture_output=True)
        
        # Export predictions
        export_cmd = [
            "uv", "run", "python", "-m", "prodigy",
            "db-out",
            f"{dataset_name}_predictions",
            str(pred_file)
        ]
        
        subprocess.run(export_cmd, capture_output=True)
        
        # Analyze errors
        errors = {
            "false_positives": [],
            "false_negatives": [],
            "correct": []
        }
        
        if pred_file.exists():
            with open(pred_file, 'r') as f:
                for line in f:
                    example = json.loads(line)
                    
                    # Check if prediction matches annotation
                    predicted = example.get("accept", False)
                    actual = example.get("answer") == "accept"
                    
                    if predicted and not actual:
                        errors["false_positives"].append(example.get("meta", {}).get("file", "unknown"))
                    elif not predicted and actual:
                        errors["false_negatives"].append(example.get("meta", {}).get("file", "unknown"))
                    else:
                        errors["correct"].append(example.get("meta", {}).get("file", "unknown"))
        
        # Save error analysis
        error_report = {
            "total_examples": len(errors["correct"]) + len(errors["false_positives"]) + len(errors["false_negatives"]),
            "correct": len(errors["correct"]),
            "false_positives": len(errors["false_positives"]),
            "false_negatives": len(errors["false_negatives"]),
            "error_rate": (len(errors["false_positives"]) + len(errors["false_negatives"])) / max(1, len(errors["correct"]) + len(errors["false_positives"]) + len(errors["false_negatives"])),
            "false_positive_files": errors["false_positives"][:10],  # First 10
            "false_negative_files": errors["false_negatives"][:10]   # First 10
        }
        
        report_file = output_path / "error_analysis_report.json"
        with open(report_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        print(f"\nError analysis complete!")
        print(f"Report saved to: {report_file}")
        print(f"\nSummary:")
        print(f"- Total examples: {error_report['total_examples']}")
        print(f"- Correct: {error_report['correct']}")
        print(f"- False positives: {error_report['false_positives']}")
        print(f"- False negatives: {error_report['false_negatives']}")
        print(f"- Error rate: {error_report['error_rate']:.2%}")
        
        return error_report
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"error": str(e)}

def generate_performance_report(
    model_path: str = "models/goalkeeper_model",
    dataset_name: str = "goalkeeper_detection",
    output_file: str = "model_performance_report.md"
):
    """
    Generate a comprehensive performance report
    
    Args:
        model_path: Path to the model
        dataset_name: Dataset name
        output_file: Output report file
    """
    
    print("Generating comprehensive performance report...")
    
    # Run evaluation
    eval_results = evaluate_on_dataset(model_path, dataset_name)
    
    # Run error analysis
    error_results = analyze_errors(model_path, dataset_name)
    
    # Generate report
    report = f"""# Goalkeeper Detection Model Performance Report

## Model Information
- **Model Path**: {model_path}
- **Dataset**: {dataset_name}

## Performance Metrics
"""
    
    if "error" not in eval_results:
        report += f"""
- **Accuracy**: {eval_results.get('accuracy', 'N/A'):.2%}
- **Precision**: {eval_results.get('precision', 'N/A'):.2%}
- **Recall**: {eval_results.get('recall', 'N/A'):.2%}
- **F1 Score**: {eval_results.get('f1_score', 'N/A'):.2%}
"""
    
    if "error" not in error_results:
        report += f"""
## Error Analysis
- **Total Examples**: {error_results['total_examples']}
- **Correctly Classified**: {error_results['correct']}
- **False Positives**: {error_results['false_positives']}
- **False Negatives**: {error_results['false_negatives']}
- **Error Rate**: {error_results['error_rate']:.2%}
"""
    
    report += """
## Recommendations
1. Review false positive examples to understand when the model incorrectly detects goalkeepers
2. Review false negative examples to understand when the model misses goalkeeper indicators
3. Consider collecting more training data for edge cases
4. Experiment with different preprocessing parameters if accuracy is low
"""
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\nPerformance report saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate goalkeeper detection model")
    parser.add_argument("--model", default="models/goalkeeper_model", help="Path to model")
    parser.add_argument("--dataset", default="goalkeeper_detection", help="Dataset name")
    parser.add_argument("--cross-validate", action="store_true", help="Run cross-validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--analyze-errors", action="store_true", help="Analyze model errors")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    
    args = parser.parse_args()
    
    if args.cross_validate:
        cross_validate(args.dataset, n_folds=args.folds)
    elif args.analyze_errors:
        analyze_errors(args.model, args.dataset)
    elif args.report:
        generate_performance_report(args.model, args.dataset)
    else:
        # Standard evaluation
        results = evaluate_on_dataset(args.model, args.dataset)
        
        if "error" not in results:
            print("\nEvaluation Results:")
            print(f"- Accuracy: {results.get('accuracy', 'N/A')}")
            print(f"- Precision: {results.get('precision', 'N/A')}")
            print(f"- Recall: {results.get('recall', 'N/A')}")
            print(f"- F1 Score: {results.get('f1_score', 'N/A')}")