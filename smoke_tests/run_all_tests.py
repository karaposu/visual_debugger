#!/usr/bin/env python
"""
Run all smoke tests and provide a comprehensive report.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_test(test_file):
    """Run a single test file and return results"""
    print(f"\n{'='*80}")
    print(f"Running {test_file.name}...")
    print('='*80)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def main():
    """Run all tests and provide summary"""
    print("\n" + "="*80)
    print("VISUAL DEBUGGER SMOKE TEST SUITE")
    print("="*80)
    print("\nRunning all smoke tests to validate the implementation...")
    
    # Get all test files
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))
    test_files = [f for f in test_files if f.name != "run_all_tests.py"]
    
    if not test_files:
        print("❌ No test files found!")
        return 1
    
    print(f"\nFound {len(test_files)} test files:")
    for f in test_files:
        print(f"  • {f.name}")
    
    # Run each test
    results = {}
    for test_file in test_files:
        passed = run_test(test_file)
        results[test_file.name] = passed
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUITE SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\nTest Results ({passed_count}/{total_count} passed):")
    print("-" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        # Extract test number and focus area from filename
        parts = test_name.replace("test_", "").replace(".py", "").split("_", 1)
        test_num = parts[0]
        test_area = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
        
        print(f"  Test {test_num}: {test_area:<40} {status}")
    
    print("-" * 60)
    
    if passed_count == total_count:
        print("\n🎉 SUCCESS! All tests passed!")
        print("\nThe Visual Debugger implementation is fully functional:")
        print("  ✓ Type-specific annotations working")
        print("  ✓ Annotation processor rendering correctly")
        print("  ✓ Image composition functioning")
        print("  ✓ Info panels displaying properly")
        print("  ✓ Full integration operational")
        return 0
    else:
        failed_count = total_count - passed_count
        print(f"\n⚠️  {failed_count} test(s) failed.")
        print("\nFailed tests:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  • {test_name}")
        print("\nPlease review the individual test outputs above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())