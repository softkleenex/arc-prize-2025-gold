"""Submit kernel to competition using Kaggle API."""

from kaggle.api.kaggle_api_extended import KaggleApi
import time

api = KaggleApi()
api.authenticate()

print("="*70)
print("Submitting kernel to ARC Prize 2025")
print("="*70)

kernel_slug = "softkleenex/arc-prize-2025-baseline-submission"
competition = "arc-prize-2025"

try:
    print(f"\nKernel: {kernel_slug}")
    print(f"Competition: {competition}")
    print("\nAttempting to submit...")

    # Submit the kernel to competition
    result = api.competition_submit_cli(
        file_name=None,
        message="Baseline submission: Simple rule-based transformations",
        competition=competition,
        kernel=kernel_slug,
        kernel_version=1
    )

    print("\n✓ Submission successful!")
    print(f"Result: {result}")

except AttributeError:
    print("\n⚠ Direct API submission not available.")
    print("\nPlease submit manually:")
    print(f"1. Visit: https://www.kaggle.com/code/{kernel_slug}")
    print(f"2. Click 'Submit to Competition' button")
    print(f"3. Confirm submission")

except Exception as e:
    print(f"\n✗ Error during submission: {e}")
    print(f"\nError type: {type(e).__name__}")

    print("\n" + "="*70)
    print("MANUAL SUBMISSION REQUIRED")
    print("="*70)
    print(f"\n1. Open: https://www.kaggle.com/code/{kernel_slug}")
    print(f"2. Click the '...' menu (three dots)")
    print(f"3. Select 'Submit to Competition'")
    print(f"4. Choose version 1")
    print(f"5. Confirm submission")

print("\n" + "="*70)
print("Checking submission status...")
print("="*70)

time.sleep(2)

try:
    submissions = api.competition_submissions(competition)
    if submissions:
        print(f"\nFound {len(submissions)} submission(s):")
        for sub in submissions[:5]:
            print(f"  - {sub.fileName}: {sub.status} (Score: {sub.publicScore})")
    else:
        print("\nNo submissions found yet.")
        print("If you just submitted, please wait a few minutes and check again.")
except Exception as e:
    print(f"\nCould not retrieve submissions: {e}")

print("\n" + "="*70)
print("Next steps:")
print("="*70)
print(f"1. Check submission status at:")
print(f"   https://www.kaggle.com/competitions/{competition}/submissions")
print(f"2. View leaderboard at:")
print(f"   https://www.kaggle.com/competitions/{competition}/leaderboard")
print()
