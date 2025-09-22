# Unit Tests for TMMC_Wrapper.Logging Module

This directory contains comprehensive unit tests for the `Logging.py` module from the TMMC_Wrapper package.

## Test Files

- **`test_logging.py`** - Main test suite with 32 unit tests
- **`run_logging_tests.py`** - Test runner script with dependency mocking
- **`test_setup.py`** - Standalone setup script for mocking dependencies
- **`test_requirements.txt`** - Documentation of test dependencies
- **`README.md`** - This documentation file

## Test Coverage

The test suite provides comprehensive coverage of the `Logging` class including:

### Input Validation Tests
- `configure_logging()` with various invalid inputs (None, non-list, non-string, empty strings)
- Input cleaning and normalization (whitespace stripping, duplicate removal)
- Edge cases and error conditions

### Process Management Tests
- `start_logging()` success and failure scenarios
- `stop_logging()` with graceful shutdown and signal escalation
- Process state checking with `is_logging_active()`
- Resource cleanup and error handling

### File Operations Tests  
- `get_logging_data()` for reading ROS2 bag files
- `delete_logging_data()` for cleanup operations
- Directory validation and error handling

### Utility Method Tests
- `current_logging_dir()` for directory access
- Integration workflow testing

### Error Handling Tests
- Exception scenarios and proper error messages
- Resource cleanup in error conditions
- Timeout handling and process termination

## Running the Tests

### Simple Execution
```bash
python3 run_logging_tests.py
```

### Verbose Output
```bash
python3 run_logging_tests.py -v
```

### Quiet Mode
```bash
python3 run_logging_tests.py -q
```

## Test Architecture

### Dependency Mocking
The tests use comprehensive mocking to avoid requiring actual ROS2 installation:
- `rosbag2_py` - ROS2 bag handling
- `rclpy` - ROS2 Python client library
- `sensor_msgs`, `geometry_msgs` - ROS2 message types
- `cv2`, `numpy` - Computer vision dependencies
- Other TMMC_Wrapper module dependencies

### Test Structure
- **TestLogging** - Main unit test class (30 tests)
- **TestLoggingIntegration** - Integration workflow tests (2 tests)
- Extensive use of `unittest.mock` for isolating the code under test
- Proper setup and teardown for each test case

### Validation Approach
The tests validate:
1. **Functional correctness** - Methods behave as expected
2. **Error handling** - Proper exceptions for invalid inputs
3. **Resource management** - Cleanup and process handling
4. **Integration flows** - Complete workflows work end-to-end

## Test Results

All 32 tests pass successfully:
- 0 failures
- 0 errors  
- 0 skipped

The test suite demonstrates that the enhanced `Logging.py` module properly handles:
- ✅ Input validation and sanitization
- ✅ Robust process management with graceful shutdown
- ✅ Error handling and resource cleanup
- ✅ Security improvements (no shell injection)
- ✅ Logging and observability
- ✅ Complete workflow integration

## Implementation Notes

The tests were designed to validate the significant improvements made to the `Logging.py` module, including:

1. **Enhanced Input Validation**: Comprehensive type checking, null validation, and topic sanitization
2. **Robust Process Management**: Graceful process termination with SIGINT → SIGTERM → SIGKILL escalation
3. **Security Improvements**: Subprocess execution without shell=True
4. **Resource Management**: Proper cleanup of file handles and processes
5. **Error Handling**: Comprehensive exception handling with logging
6. **Observability**: Structured logging for debugging and monitoring

These tests ensure the module is production-ready and handles edge cases gracefully.