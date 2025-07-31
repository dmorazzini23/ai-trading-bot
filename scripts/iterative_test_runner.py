#!/usr/bin/env python3
"""
Iterative Test Runner for AI Trading Bot
Enables systematic test execution, failure analysis, and debugging.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestResult:
    """Represents the result of a test run."""
    
    def __init__(self, timestamp: str, passed: int, failed: int, skipped: int, 
                 total: int, duration: float, failures: List[str] = None):
        self.timestamp = timestamp
        self.passed = passed
        self.failed = failed
        self.skipped = skipped
        self.total = total
        self.duration = duration
        self.failures = failures or []

class IterativeTestRunner:
    """Manages iterative test execution and analysis."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results_history = []
        self.env_script = self.project_root / "scripts" / "configure_test_env.py"
        
    def setup_environment(self) -> bool:
        """Set up test environment before running tests."""
        try:
            logger.info("🔧 Setting up test environment...")
            
            # Run environment configuration script
            if self.env_script.exists():
                result = subprocess.run(
                    [sys.executable, str(self.env_script)],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root)
                )
                
                if result.returncode != 0:
                    logger.warning(f"Environment setup script failed: {result.stderr}")
                else:
                    logger.info("✅ Environment configuration completed")
            
            # Set environment variables programmatically
            test_env_vars = {
                "PYTEST_RUNNING": "1",
                "TESTING": "1",
                "ALPACA_API_KEY": "PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "ALPACA_SECRET_KEY": "SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD",
                "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
                "WEBHOOK_SECRET": "test-webhook-secret-for-testing",
                "FLASK_PORT": "9000",
                "BOT_MODE": "balanced",
                "DOLLAR_RISK_LIMIT": "0.02",
                "BUY_THRESHOLD": "0.5",
                "TRADE_LOG_FILE": "test_trades.csv",
                "SEED": "42",
                "RATE_LIMIT_BUDGET": "190",
            }
            
            for key, value in test_env_vars.items():
                os.environ[key] = value
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def run_tests(self, test_pattern: str = "", max_failures: int = 10,
                  timeout: int = 300, parallel: bool = False) -> TestResult:
        """Run tests with specified parameters."""
        
        timestamp = datetime.now().isoformat()
        logger.info(f"🧪 Starting test run at {timestamp}")
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            "--disable-warnings",
            "--tb=short",
            f"--maxfail={max_failures}",
            "-v"
        ]
        
        if parallel:
            cmd.extend(["-n", "auto"])
            
        if test_pattern:
            cmd.append(test_pattern)
        
        # Set environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root),
                env=env
            )
            
            duration = time.time() - start_time
            
            # Parse test results
            passed, failed, skipped, total = self._parse_pytest_output(result.stdout)
            failures = self._extract_failures(result.stdout)
            
            test_result = TestResult(
                timestamp=timestamp,
                passed=passed,
                failed=failed,
                skipped=skipped,
                total=total,
                duration=duration,
                failures=failures
            )
            
            self.results_history.append(test_result)
            
            # Log results
            if failed == 0:
                logger.info(f"✅ All tests passed! ({passed}/{total} passed in {duration:.1f}s)")
            else:
                logger.warning(f"❌ {failed} tests failed ({passed}/{total} passed in {duration:.1f}s)")
                
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Test run timed out after {timeout} seconds")
            return TestResult(timestamp, 0, 1, 0, 1, timeout, ["Test run timeout"])
        except Exception as e:
            logger.error(f"❌ Test run failed: {e}")
            return TestResult(timestamp, 0, 1, 0, 1, 0, [str(e)])
    
    def _parse_pytest_output(self, output: str) -> Tuple[int, int, int, int]:
        """Parse pytest output to extract test counts."""
        lines = output.split('\n')
        
        # Look for the summary line
        for line in lines:
            if 'passed' in line and ('failed' in line or 'skipped' in line or 'error' in line):
                # Parse different formats
                passed = failed = skipped = 0
                
                if 'passed' in line:
                    try:
                        passed = int(line.split('passed')[0].strip().split()[-1])
                    except (ValueError, IndexError):
                        pass
                
                if 'failed' in line:
                    try:
                        failed = int(line.split('failed')[0].split(',')[-1].strip().split()[-1])
                    except (ValueError, IndexError):
                        pass
                
                if 'skipped' in line:
                    try:
                        skipped = int(line.split('skipped')[0].split(',')[-1].strip().split()[-1])
                    except (ValueError, IndexError):
                        pass
                
                total = passed + failed + skipped
                return passed, failed, skipped, total
        
        # Fallback: count individual test results
        passed = len([l for l in lines if ' PASSED ' in l])
        failed = len([l for l in lines if ' FAILED ' in l])
        skipped = len([l for l in lines if ' SKIPPED ' in l])
        total = passed + failed + skipped
        
        return passed, failed, skipped, total
    
    def _extract_failures(self, output: str) -> List[str]:
        """Extract failure descriptions from pytest output."""
        failures = []
        lines = output.split('\n')
        
        in_failure = False
        current_failure = []
        
        for line in lines:
            if 'FAILURES' in line or 'ERRORS' in line:
                in_failure = True
                continue
            elif line.startswith('=') and ('failed' in line or 'passed' in line):
                in_failure = False
                break
            elif in_failure and line.strip():
                if line.startswith('_'):
                    if current_failure:
                        failures.append('\n'.join(current_failure))
                        current_failure = []
                    current_failure.append(line)
                elif current_failure:
                    current_failure.append(line)
        
        if current_failure:
            failures.append('\n'.join(current_failure))
        
        return failures
    
    def run_iterative_tests(self, max_iterations: int = 5, target_pattern: str = "") -> bool:
        """Run tests iteratively until all pass or max iterations reached."""
        
        logger.info(f"🔄 Starting iterative test run (max {max_iterations} iterations)")
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"📋 Iteration {iteration}/{max_iterations}")
            
            # Setup environment
            if not self.setup_environment():
                logger.error("❌ Environment setup failed, skipping iteration")
                continue
            
            # Run tests
            result = self.run_tests(target_pattern, max_failures=5)
            
            # Check if all tests passed
            if result.failed == 0:
                logger.info(f"🎉 All tests passed in iteration {iteration}!")
                self.save_results()
                return True
            
            # Analyze failures
            logger.info(f"📊 Iteration {iteration} results:")
            logger.info(f"   Passed: {result.passed}")
            logger.info(f"   Failed: {result.failed}")
            logger.info(f"   Skipped: {result.skipped}")
            logger.info(f"   Duration: {result.duration:.1f}s")
            
            if result.failures:
                logger.info("🔍 Failure summary:")
                for i, failure in enumerate(result.failures[:3], 1):  # Show first 3 failures
                    logger.info(f"   {i}. {failure.split()[0] if failure.split() else 'Unknown'}")
            
            # Wait before next iteration
            if iteration < max_iterations:
                logger.info("⏳ Waiting before next iteration...")
                time.sleep(2)
        
        logger.warning(f"❌ Failed to achieve all tests passing after {max_iterations} iterations")
        self.save_results()
        return False
    
    def save_results(self) -> None:
        """Save test results to file."""
        results_file = self.project_root / "test_results.json"
        
        results_data = []
        for result in self.results_history:
            results_data.append({
                "timestamp": result.timestamp,
                "passed": result.passed,
                "failed": result.failed,
                "skipped": result.skipped,
                "total": result.total,
                "duration": result.duration,
                "failures": result.failures
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Test results saved to {results_file}")
    
    def run_specific_failing_tests(self) -> bool:
        """Run the originally failing tests to verify fixes."""
        
        logger.info("🎯 Running originally failing tests...")
        
        failing_tests = [
            "tests/test_run_overlap.py::test_run_all_trades_overlap",
            "tests/test_strategy_allocator_smoke.py::test_allocator", 
            "tests/test_nameerror_integration.py::test_bot_engine_import_no_nameerror"
        ]
        
        all_passed = True
        
        for test in failing_tests:
            logger.info(f"🔍 Running {test}")
            
            if not self.setup_environment():
                logger.error("❌ Environment setup failed")
                return False
            
            result = self.run_tests(test, max_failures=1)
            
            if result.failed == 0:
                logger.info(f"✅ {test} PASSED")
            else:
                logger.error(f"❌ {test} FAILED")
                all_passed = False
        
        return all_passed

def main():
    """Main function for iterative test runner."""
    
    # Get project root
    project_root = "/home/runner/work/ai-trading-bot/ai-trading-bot"
    
    # Create runner
    runner = IterativeTestRunner(project_root)
    
    # Command line argument handling
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "failing":
            # Run originally failing tests
            success = runner.run_specific_failing_tests()
            sys.exit(0 if success else 1)
        elif command == "all":
            # Run all tests iteratively
            success = runner.run_iterative_tests(max_iterations=3, target_pattern="")
            sys.exit(0 if success else 1)
        elif command == "fast":
            # Run a subset of fast tests
            runner.setup_environment()
            result = runner.run_tests("tests/test_*smoke*.py tests/test_run_overlap.py", max_failures=5)
            sys.exit(0 if result.failed == 0 else 1)
        else:
            # Run specific test pattern
            runner.setup_environment()
            result = runner.run_tests(command, max_failures=5)
            sys.exit(0 if result.failed == 0 else 1)
    else:
        # Default: run originally failing tests
        success = runner.run_specific_failing_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()