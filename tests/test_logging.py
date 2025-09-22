"""
Unit tests for TMMC_Wrapper.Logging module.

This test suite provides comprehensive coverage of the Logging class,
including input validation, process management, error handling, and
resource cleanup functionality.
"""

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock, call
import tempfile
import os
import subprocess
import shutil
import signal
import logging
from typing import Any, Optional

# Mock the ROS dependencies since they won't be available in test environment
import sys
from unittest.mock import MagicMock

# Mock ROS2 modules
sys.modules['rosbag2_py'] = MagicMock()
sys.modules['rosidl_runtime_py'] = MagicMock()
sys.modules['rosidl_runtime_py.utilities'] = MagicMock()
sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.serialization'] = MagicMock()

# Import the modules under test after mocking dependencies
from TMMC_Wrapper.Logging import Logging
from TMMC_Wrapper.Robot import Robot


class TestLogging(unittest.TestCase):
    """Test suite for the Logging class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.mock_robot = Mock(spec=Robot)
        self.logging_instance = Logging(self.mock_robot)
        
        # Set up logging to capture log messages for testing
        self.log_capture = []
        self.test_handler = logging.Handler()
        self.test_handler.emit = lambda record: self.log_capture.append(record)
        
        # Get the logger used by the Logging class
        self.logger = logging.getLogger('TMMC_Wrapper.Logging')
        self.logger.addHandler(self.test_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def tearDown(self):
        """Clean up after each test method."""
        self.logger.removeHandler(self.test_handler)
        self.log_capture.clear()

    def test_init(self):
        """Test Logging class initialization."""
        robot = Mock(spec=Robot)
        logging_obj = Logging(robot)
        self.assertEqual(logging_obj.robot, robot)

    def test_configure_logging_valid_topics(self):
        """Test configure_logging with valid topic list."""
        topics = ["/topic1", "/topic2", "/topic3"]
        self.logging_instance.configure_logging(topics)
        
        self.assertEqual(self.mock_robot.logging_topics, topics)
        # Check that logging message was recorded
        self.assertTrue(any("Configured logging topics" in str(record.getMessage()) 
                          for record in self.log_capture))

    def test_configure_logging_none_input(self):
        """Test configure_logging with None input raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.logging_instance.configure_logging(None)
        self.assertEqual(str(context.exception), "topics must not be None")

    def test_configure_logging_non_list_input(self):
        """Test configure_logging with non-list input raises TypeError."""
        with self.assertRaises(TypeError) as context:
            self.logging_instance.configure_logging("not_a_list")
        self.assertEqual(str(context.exception), "topics must be a list of strings")

    def test_configure_logging_non_string_topics(self):
        """Test configure_logging with non-string topics raises TypeError."""
        topics = ["/topic1", 123, "/topic3"]
        with self.assertRaises(TypeError) as context:
            self.logging_instance.configure_logging(topics)
        self.assertEqual(str(context.exception), "each topic must be a string")

    def test_configure_logging_empty_string_topics(self):
        """Test configure_logging with empty string topics raises ValueError."""
        topics = ["/topic1", "  ", "/topic3"]
        with self.assertRaises(ValueError) as context:
            self.logging_instance.configure_logging(topics)
        self.assertEqual(str(context.exception), "topic entries must be non-empty strings")

    def test_configure_logging_strips_whitespace(self):
        """Test configure_logging strips whitespace from topics."""
        topics = ["  /topic1  ", "\t/topic2\n", "/topic3"]
        self.logging_instance.configure_logging(topics)
        
        expected = ["/topic1", "/topic2", "/topic3"]
        self.assertEqual(self.mock_robot.logging_topics, expected)

    def test_configure_logging_removes_duplicates(self):
        """Test configure_logging removes duplicate topics while preserving order."""
        topics = ["/topic1", "/topic2", "/topic1", "/topic3", "/topic2"]
        self.logging_instance.configure_logging(topics)
        
        expected = ["/topic1", "/topic2", "/topic3"]
        self.assertEqual(self.mock_robot.logging_topics, expected)

    def test_configure_logging_empty_after_cleaning(self):
        """Test configure_logging raises error when no valid topics remain."""
        topics = ["", "   ", "\t\n"]
        with self.assertRaises(ValueError) as context:
            self.logging_instance.configure_logging(topics)
        # The actual implementation raises the first error it encounters (empty string)
        self.assertEqual(str(context.exception), "topic entries must be non-empty strings")

    @patch('shutil.which')
    @patch('subprocess.Popen')
    @patch('builtins.open', mock_open())
    @patch('time.time')
    @patch('os.path.isdir')
    @patch('time.sleep')
    def test_start_logging_success(self, mock_sleep, mock_isdir, mock_time, mock_popen, mock_which):
        """Test successful start_logging execution."""
        # Setup mocks
        mock_which.return_value = '/usr/bin/ros2'
        mock_time.return_value = 1234567890
        mock_isdir.return_value = True
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Configure topics first
        self.mock_robot.logging_topics = ["/topic1", "/topic2"]
        
        # Call start_logging
        self.logging_instance.start_logging()
        
        # Verify results
        self.assertEqual(self.mock_robot.logging_dir, '/tmp/notebook_bag_1234567890')
        self.assertEqual(self.mock_robot.logging_process_log_path, '/tmp/ros2_bag_1234567890.log')
        self.assertEqual(self.mock_robot.logging_instance, mock_process)
        
        # Verify subprocess was called with correct arguments
        expected_cmd = [
            'ros2', 'bag', 'record',
            '-s', 'mcap',
            '--output', '/tmp/notebook_bag_1234567890',
            '/topic1', '/topic2'
        ]
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        self.assertEqual(call_args[0][0], expected_cmd)

    def test_start_logging_already_active(self):
        """Test start_logging raises error when logging is already active."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        self.mock_robot.logging_instance = mock_process
        
        with self.assertRaises(RuntimeError) as context:
            self.logging_instance.start_logging()
        self.assertEqual(str(context.exception), "logging already active")

    def test_start_logging_no_topics_configured(self):
        """Test start_logging raises error when no topics are configured."""
        with self.assertRaises(ValueError) as context:
            self.logging_instance.start_logging()
        self.assertEqual(str(context.exception), "logging topics are not configured; call configure_logging first")

    @patch('shutil.which')
    def test_start_logging_ros2_not_found(self, mock_which):
        """Test start_logging raises error when ros2 executable not found."""
        mock_which.return_value = None
        self.mock_robot.logging_topics = ["/topic1"]
        
        with self.assertRaises(EnvironmentError) as context:
            self.logging_instance.start_logging()
        self.assertEqual(str(context.exception), "'ros2' executable not found in PATH")

    @patch('shutil.which')
    @patch('subprocess.Popen')
    @patch('builtins.open', side_effect=IOError("Failed to open file"))
    @patch('time.time')
    def test_start_logging_file_error(self, mock_time, mock_open, mock_popen, mock_which):
        """Test start_logging handles file opening errors."""
        mock_which.return_value = '/usr/bin/ros2'
        mock_time.return_value = 1234567890
        self.mock_robot.logging_topics = ["/topic1"]
        
        with self.assertRaises(IOError):
            self.logging_instance.start_logging()

    @patch('os.killpg')
    @patch('os.getpgid')
    def test_stop_logging_success(self, mock_getpgid, mock_killpg):
        """Test successful stop_logging execution."""
        # Setup mock process
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        mock_process.wait.return_value = None
        self.mock_robot.logging_instance = mock_process
        self.mock_robot.logging_dir = "/tmp/test_bag"
        
        # Setup mock file handle
        mock_fh = Mock()
        self.mock_robot._logging_process_log_fh = mock_fh
        
        mock_getpgid.return_value = 12345
        
        # Call stop_logging
        result = self.logging_instance.stop_logging()
        
        # Verify results
        self.assertEqual(result, "/tmp/test_bag")
        self.assertIsNone(self.mock_robot.logging_instance)
        mock_killpg.assert_called_once_with(12345, signal.SIGINT)
        mock_process.wait.assert_called_once_with(timeout=10)
        mock_fh.close.assert_called_once()

    def test_stop_logging_no_active_process(self):
        """Test stop_logging raises error when no active process."""
        self.mock_robot.logging_instance = None
        
        with self.assertRaises(RuntimeError) as context:
            self.logging_instance.stop_logging()
        self.assertEqual(str(context.exception), "no active logging process to stop")

    @patch('os.killpg')
    @patch('os.getpgid')
    def test_stop_logging_escalation_to_sigterm(self, mock_getpgid, mock_killpg):
        """Test stop_logging escalates to SIGTERM when SIGINT timeout occurs."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        # First wait() call raises TimeoutExpired, second succeeds
        mock_process.wait.side_effect = [subprocess.TimeoutExpired(None, 10), None]
        self.mock_robot.logging_instance = mock_process
        self.mock_robot.logging_dir = "/tmp/test_bag"
        
        mock_getpgid.return_value = 12345
        
        self.logging_instance.stop_logging()
        
        # Verify both SIGINT and SIGTERM were sent
        expected_calls = [
            call(12345, signal.SIGINT),
            call(12345, signal.SIGTERM)
        ]
        mock_killpg.assert_has_calls(expected_calls)

    @patch('os.killpg')
    @patch('os.getpgid')
    def test_stop_logging_escalation_to_sigkill(self, mock_getpgid, mock_killpg):
        """Test stop_logging escalates to SIGKILL when SIGTERM timeout occurs."""
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.returncode = -9
        # Both wait() calls raise TimeoutExpired, then final wait succeeds
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired(None, 10),  # SIGINT timeout
            subprocess.TimeoutExpired(None, 5),   # SIGTERM timeout
            None                                   # Final wait after SIGKILL
        ]
        self.mock_robot.logging_instance = mock_process
        self.mock_robot.logging_dir = "/tmp/test_bag"
        
        mock_getpgid.return_value = 12345
        
        self.logging_instance.stop_logging()
        
        # Verify all three signals were sent
        expected_calls = [
            call(12345, signal.SIGINT),
            call(12345, signal.SIGTERM),
            call(12345, signal.SIGKILL)
        ]
        mock_killpg.assert_has_calls(expected_calls)

    @patch('os.path.isdir')
    def test_get_logging_data_invalid_input(self, mock_isdir):
        """Test get_logging_data with invalid inputs."""
        # Test None input
        with self.assertRaises(ValueError):
            self.logging_instance.get_logging_data(None)
        
        # Test empty string
        with self.assertRaises(ValueError):
            self.logging_instance.get_logging_data("")
        
        # Test non-string input
        with self.assertRaises(ValueError):
            self.logging_instance.get_logging_data(123)

    @patch('os.path.isdir')
    def test_get_logging_data_directory_not_exist(self, mock_isdir):
        """Test get_logging_data with non-existent directory."""
        mock_isdir.return_value = False
        
        with self.assertRaises(FileNotFoundError) as context:
            self.logging_instance.get_logging_data("/nonexistent/path")
        
        self.assertIn("logging directory does not exist", str(context.exception))

    @patch('os.path.isdir')
    @patch('rosbag2_py.SequentialReader')
    def test_get_logging_data_success(self, mock_reader_class, mock_isdir):
        """Test successful get_logging_data execution."""
        mock_isdir.return_value = True
        
        # Setup mock reader
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        
        # Mock topic types
        mock_topic_type = Mock()
        mock_topic_type.name = "/test_topic"
        mock_topic_type.type = "test_msgs/TestMessage"
        mock_reader.get_all_topics_and_types.return_value = [mock_topic_type]
        
        # Mock message reading
        mock_reader.has_next.side_effect = [True, True, False]  # Two messages then done
        mock_reader.read_next.side_effect = [
            ("/test_topic", b"data1", 1000),
            ("/test_topic", b"data2", 2000)
        ]
        
        # Mock message deserialization through the mocked modules
        import rosidl_runtime_py.utilities
        import rclpy.serialization
        
        rosidl_runtime_py.utilities.get_message.return_value = Mock()
        rclpy.serialization.deserialize_message.return_value = "mock_message"
        
        result = self.logging_instance.get_logging_data("/test/path")
        
        # Verify basic structure - we expect a dict with the topic as key
        self.assertIsInstance(result, dict)
        self.assertIn("/test_topic", result)
        self.assertEqual(len(result["/test_topic"]), 2)  # Two messages
        
        # Verify message structure (timestamp, message) tuples
        for timestamp, message in result["/test_topic"]:
            self.assertIsInstance(timestamp, int)
            # Message will be the mocked return value

    @patch('os.path.exists')
    def test_delete_logging_data_invalid_input(self, mock_exists):
        """Test delete_logging_data with invalid inputs."""
        with self.assertRaises(ValueError):
            self.logging_instance.delete_logging_data(None)
        
        with self.assertRaises(ValueError):
            self.logging_instance.delete_logging_data("")
        
        with self.assertRaises(ValueError):
            self.logging_instance.delete_logging_data(123)

    @patch('os.path.exists')
    def test_delete_logging_data_directory_not_exist(self, mock_exists):
        """Test delete_logging_data when directory doesn't exist."""
        mock_exists.return_value = False
        
        # Should not raise exception, just log warning
        self.logging_instance.delete_logging_data("/nonexistent/path")
        
        # Check that warning was logged
        self.assertTrue(any("logging directory does not exist" in str(record.getMessage()) 
                          for record in self.log_capture))

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_delete_logging_data_success(self, mock_rmtree, mock_exists):
        """Test successful delete_logging_data execution."""
        mock_exists.return_value = True
        
        self.logging_instance.delete_logging_data("/test/path")
        
        mock_rmtree.assert_called_once_with("/test/path")

    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_delete_logging_data_error(self, mock_rmtree, mock_exists):
        """Test delete_logging_data handles removal errors."""
        mock_exists.return_value = True
        mock_rmtree.side_effect = OSError("Permission denied")
        
        with self.assertRaises(OSError):
            self.logging_instance.delete_logging_data("/test/path")

    def test_is_logging_active_no_instance(self):
        """Test is_logging_active when no logging instance exists."""
        self.assertFalse(self.logging_instance.is_logging_active())

    def test_is_logging_active_instance_none(self):
        """Test is_logging_active when logging instance is None."""
        self.mock_robot.logging_instance = None
        self.assertFalse(self.logging_instance.is_logging_active())

    def test_is_logging_active_process_running(self):
        """Test is_logging_active when process is running."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        self.mock_robot.logging_instance = mock_process
        
        self.assertTrue(self.logging_instance.is_logging_active())

    def test_is_logging_active_process_finished(self):
        """Test is_logging_active when process has finished."""
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Process finished
        self.mock_robot.logging_instance = mock_process
        
        self.assertFalse(self.logging_instance.is_logging_active())

    def test_current_logging_dir_no_dir(self):
        """Test current_logging_dir when no directory is set."""
        result = self.logging_instance.current_logging_dir()
        self.assertIsNone(result)

    def test_current_logging_dir_with_dir(self):
        """Test current_logging_dir when directory is set."""
        self.mock_robot.logging_dir = "/test/logging/dir"
        result = self.logging_instance.current_logging_dir()
        self.assertEqual(result, "/test/logging/dir")


class TestLoggingIntegration(unittest.TestCase):
    """Integration tests for the Logging class workflow."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.mock_robot = Mock(spec=Robot)
        self.logging_instance = Logging(self.mock_robot)
    
    def test_full_logging_workflow(self):
        """Test complete logging workflow from configure to cleanup."""
        # Step 1: Configure logging
        topics = ["/topic1", "/topic2"]
        self.logging_instance.configure_logging(topics)
        self.assertEqual(self.mock_robot.logging_topics, topics)
        
        # Step 2: Check logging is not active initially
        self.assertFalse(self.logging_instance.is_logging_active())
        self.assertIsNone(self.logging_instance.current_logging_dir())
        
        # Step 3: Start logging (mocked)
        with patch('shutil.which', return_value='/usr/bin/ros2'):
            with patch('subprocess.Popen') as mock_popen:
                with patch('builtins.open', mock_open()):
                    with patch('time.time', return_value=1234567890):
                        with patch('os.path.isdir', return_value=True):
                            with patch('time.sleep'):
                                mock_process = Mock()
                                mock_process.poll.return_value = None
                                mock_process.pid = 12345
                                mock_popen.return_value = mock_process
                                
                                self.logging_instance.start_logging()
                                
                                # Verify logging is now active
                                self.assertTrue(self.logging_instance.is_logging_active())
                                self.assertEqual(self.logging_instance.current_logging_dir(), 
                                               "/tmp/notebook_bag_1234567890")
        
        # Step 4: Stop logging (mocked)
        with patch('os.killpg'):
            with patch('os.getpgid', return_value=12345):
                mock_process.wait.return_value = None
                mock_process.returncode = 0
                
                result_dir = self.logging_instance.stop_logging()
                
                # Verify logging stopped and directory returned
                self.assertEqual(result_dir, "/tmp/notebook_bag_1234567890")
                self.assertFalse(self.logging_instance.is_logging_active())


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    # Run the tests
    unittest.main()