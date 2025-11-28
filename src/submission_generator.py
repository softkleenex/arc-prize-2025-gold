"""
ARC Prize 2025 - Submission File Generator

This module generates submission files in the required format.
"""

import json
from pathlib import Path
from typing import Dict, List
from data_loader import ARCDataLoader
from baseline_solver import BaselineSolver, AdvancedSolver


class SubmissionGenerator:
    """Generate submission files for ARC Prize 2025."""

    def __init__(self, solver=None):
        """
        Initialize the submission generator.

        Args:
            solver: Solver instance (defaults to BaselineSolver)
        """
        self.solver = solver if solver else BaselineSolver()
        self.loader = ARCDataLoader()

    def generate_submission(self, split: str = "test",
                           output_file: str = "submissions/submission.json",
                           max_tasks: int = None) -> Dict:
        """
        Generate a submission file for evaluation.

        Args:
            split: Dataset split ('test' for final submission, 'evaluation' for validation)
            output_file: Path to save submission file
            max_tasks: Maximum number of tasks to process (None for all)

        Returns:
            Submission dictionary
        """
        print(f"Generating submission for {split} set...")

        # Load challenges
        challenges = self.loader.load_challenges(split)
        task_ids = list(challenges.keys())[:max_tasks] if max_tasks else list(challenges.keys())

        submission = {}

        for i, task_id in enumerate(task_ids):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(task_ids)} tasks...")

            task = challenges[task_id]

            # Get predictions from solver
            predictions = self.solver.solve(task, max_attempts=2)

            # Format: {"task_id": [{"attempt_1": [[...]], "attempt_2": [[...]]}]}
            # Each task has a list of test predictions
            task_submission = []
            for test_idx, test_predictions in enumerate(predictions):
                # Add both attempts for this test
                test_dict = {}
                for attempt_idx, attempt in enumerate(test_predictions[:2]):
                    attempt_key = f"attempt_{attempt_idx + 1}"
                    test_dict[attempt_key] = attempt

                task_submission.append(test_dict)

            submission[task_id] = task_submission

        print(f"\n✓ Generated predictions for {len(submission)} tasks")

        # Save submission file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)

        print(f"✓ Saved submission to: {output_file}")

        # Print file size
        file_size = output_path.stat().st_size
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

        return submission

    def validate_submission(self, submission_file: str, split: str = "evaluation") -> float:
        """
        Validate a submission file against known solutions.

        Args:
            submission_file: Path to submission file
            split: Dataset split to validate against

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if split == "test":
            print("⚠ Cannot validate against test set (solutions not available)")
            return None

        print(f"Validating submission against {split} set...")

        # Load submission
        with open(submission_file, 'r') as f:
            submission = json.load(f)

        # Load solutions
        solutions = self.loader.load_solutions(split)

        total_tests = 0
        correct_tests = 0

        for task_id, task_submission in submission.items():
            if task_id not in solutions:
                continue

            expected_solutions = solutions[task_id]

            # Handle multiple test examples
            if isinstance(expected_solutions[0], list) and isinstance(expected_solutions[0][0], list):
                # Multiple test examples
                n_tests = len(expected_solutions)
            else:
                # Single test example
                expected_solutions = [expected_solutions]
                n_tests = 1

            # task_submission is a list of test predictions
            for test_idx in range(min(n_tests, len(task_submission))):
                expected = expected_solutions[test_idx]
                attempts_dict = task_submission[test_idx]

                # Check both attempts (Pass@2)
                attempt_1 = attempts_dict.get("attempt_1", [])
                attempt_2 = attempts_dict.get("attempt_2", [])

                is_correct = (attempt_1 == expected) or (attempt_2 == expected)

                total_tests += 1
                if is_correct:
                    correct_tests += 1

        accuracy = correct_tests / total_tests if total_tests > 0 else 0.0

        print(f"\n{'='*70}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"Correct: {correct_tests}/{total_tests}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"{'='*70}\n")

        return accuracy


def create_baseline_submission():
    """Create a baseline submission using the baseline solver."""
    print("="*70)
    print("CREATING BASELINE SUBMISSION")
    print("="*70)
    print("\nUsing: BaselineSolver with simple transformations\n")

    solver = BaselineSolver()
    generator = SubmissionGenerator(solver)

    # First, validate on evaluation set
    print("\n" + "─"*70)
    print("STEP 1: Validate on Evaluation Set")
    print("─"*70)

    eval_submission = generator.generate_submission(
        split="evaluation",
        output_file="submissions/evaluation_baseline.json"
    )

    accuracy = generator.validate_submission(
        "submissions/evaluation_baseline.json",
        split="evaluation"
    )

    # Then, generate test submission
    print("\n" + "─"*70)
    print("STEP 2: Generate Test Submission")
    print("─"*70)

    test_submission = generator.generate_submission(
        split="test",
        output_file="submissions/test_baseline.json"
    )

    print("\n" + "="*70)
    print("SUBMISSION GENERATION COMPLETE")
    print("="*70)
    print("\nFiles created:")
    print("  1. submissions/evaluation_baseline.json (for validation)")
    print("  2. submissions/test_baseline.json (for Kaggle submission)")
    print("\nNext steps:")
    print("  1. Submit test_baseline.json to Kaggle")
    print("  2. Check public leaderboard score")
    print("  3. Improve solver based on results")
    print("\nSubmit command:")
    print("  kaggle competitions submit -c arc-prize-2025 \\")
    print("    -f submissions/test_baseline.json \\")
    print("    -m \"Baseline submission with simple transformations\"")
    print()


def create_advanced_submission():
    """Create a submission using the advanced solver."""
    print("="*70)
    print("CREATING ADVANCED SUBMISSION")
    print("="*70)
    print("\nUsing: AdvancedSolver with additional heuristics\n")

    solver = AdvancedSolver()
    generator = SubmissionGenerator(solver)

    # Validate on evaluation set
    eval_submission = generator.generate_submission(
        split="evaluation",
        output_file="submissions/evaluation_advanced.json"
    )

    accuracy = generator.validate_submission(
        "submissions/evaluation_advanced.json",
        split="evaluation"
    )

    # Generate test submission
    test_submission = generator.generate_submission(
        split="test",
        output_file="submissions/test_advanced.json"
    )

    print("\n✓ Advanced submission generated!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "advanced":
        create_advanced_submission()
    else:
        create_baseline_submission()
