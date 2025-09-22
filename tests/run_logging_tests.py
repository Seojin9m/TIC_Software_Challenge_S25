#!/usr/bin/env python3
"""
Comprehensive test runner for the Logging module unit tests.

This script sets up all necessary mocks and runs the complete test suite
for the TMMC_Wrapper.Logging module.

Usage:
    python run_logging_tests.py
    
    Options:
    -v, --verbose    Run tests with verbose output
    -q, --quiet      Run tests with minimal output
"""

import sys
import os
import argparse
from unittest.mock import MagicMock

def setup_mocks():
    """Set up comprehensive mocks for all external dependencies."""
    
    # ROS2 core dependencies
    sys.modules['rclpy'] = MagicMock()
    sys.modules['rclpy.node'] = MagicMock()
    sys.modules['rclpy.serialization'] = MagicMock()
    sys.modules['rclpy.qos'] = MagicMock()
    sys.modules['rclpy.action'] = MagicMock()
    sys.modules['rclpy.task'] = MagicMock()
    
    # ROS2 bag dependencies
    sys.modules['rosbag2_py'] = MagicMock()
    sys.modules['rosidl_runtime_py'] = MagicMock()
    sys.modules['rosidl_runtime_py.utilities'] = MagicMock()
    
    # ROS2 message types
    sys.modules['sensor_msgs'] = MagicMock()
    sys.modules['sensor_msgs.msg'] = MagicMock()
    sys.modules['geometry_msgs'] = MagicMock()
    sys.modules['geometry_msgs.msg'] = MagicMock()
    sys.modules['irobot_create_msgs'] = MagicMock()
    sys.modules['irobot_create_msgs.action'] = MagicMock()
    sys.modules['irobot_create_msgs.srv'] = MagicMock()
    
    # TF2 dependencies
    sys.modules['tf2_ros'] = MagicMock()
    sys.modules['tf2_ros.buffer'] = MagicMock()
    sys.modules['tf2_ros.transform_listener'] = MagicMock()
    
    # Computer vision and other dependencies
    sys.modules['cv2'] = MagicMock()
    sys.modules['numpy'] = MagicMock()
    sys.modules['apriltag'] = MagicMock()
    sys.modules['ultralytics'] = MagicMock()
    sys.modules['pynput'] = MagicMock()
    sys.modules['pynput.keyboard'] = MagicMock()

def main():
    parser = argparse.ArgumentParser(description='Run Logging module unit tests')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Run tests with verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Run tests with minimal output')
    args = parser.parse_args()
    
    # Set up mocks before importing anything
    print("Setting up mocks for external dependencies...")
    setup_mocks()
    
    # Add repo to Python path (parent directory of tests/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    print("Importing test modules...")
    try:
        # Import test modules
        from test_logging import TestLogging, TestLoggingIntegration
        import unittest
        
        print("✓ All modules imported successfully")
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add test cases
        suite.addTest(unittest.makeSuite(TestLogging))
        suite.addTest(unittest.makeSuite(TestLoggingIntegration))
        
        # Determine verbosity
        verbosity = 1  # default
        if args.verbose:
            verbosity = 2
        elif args.quiet:
            verbosity = 0
        
        print(f"\nRunning {suite.countTestCases()} tests...")
        print("=" * 70)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        # Print summary
        print("=" * 70)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        
        if result.failures:
            print("\nFAILURES:")
            for test, trace in result.failures:
                print(f"- {test}: {trace}")
        
        if result.errors:
            print("\nERRORS:")
            for test, trace in result.errors:
                print(f"- {test}: {trace}")
        
        # Exit with appropriate code
        if result.wasSuccessful():
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Failed to run tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()