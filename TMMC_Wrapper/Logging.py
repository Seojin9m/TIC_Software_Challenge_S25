import time
import subprocess
import os
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import shutil
import signal
import logging
from .Robot import Robot
from typing import Any, Optional

logger = logging.getLogger(__name__)

class Logging:
    def __init__(self, robot : Robot):
        ''' Initializes a logging instance by storing the provided robot object. '''
        self.robot = robot

    def configure_logging(self, topics : list[str]):
        ''' Sets and validates the logging topics on the robot. '''
        if topics is None:
            raise ValueError("topics must not be None")
        if not isinstance(topics, list):
            raise TypeError("topics must be a list of strings")
        cleaned_topics = []
        for topic in topics:
            if not isinstance(topic, str):
                raise TypeError("each topic must be a string")
            topic_stripped = topic.strip()
            if not topic_stripped:
                raise ValueError("topic entries must be non-empty strings")
            cleaned_topics.append(topic_stripped)
        # Preserve order but remove duplicates
        seen = set()
        deduped_topics = [t for t in cleaned_topics if not (t in seen or seen.add(t))]
        if len(deduped_topics) == 0:
            raise ValueError("at least one valid topic is required for logging")
        self.robot.logging_topics = deduped_topics
        logger.info("Configured logging topics: %s", deduped_topics)
          
    def start_logging(self):
        ''' Begins the logging process by starting a rosbag2 recorder for configured topics. '''
        if hasattr(self.robot, 'logging_instance') and self.robot.logging_instance is not None:
            if getattr(self.robot.logging_instance, 'poll', lambda: None)() is None:
                raise RuntimeError("logging already active")
        if not hasattr(self.robot, 'logging_topics') or not self.robot.logging_topics:
            raise ValueError("logging topics are not configured; call configure_logging first")
        if shutil.which('ros2') is None:
            raise EnvironmentError("'ros2' executable not found in PATH")

        timestamp = int(time.time())
        self.robot.logging_dir = '/tmp/notebook_bag_' + str(timestamp)
        self.robot.logging_process_log_path = f"/tmp/ros2_bag_{timestamp}.log"

        cmd = [
            'ros2', 'bag', 'record',
            '-s', 'mcap',
            '--output', self.robot.logging_dir,
            *self.robot.logging_topics,
        ]
        logger.info("Starting ros2 bag record to %s for topics: %s", self.robot.logging_dir, self.robot.logging_topics)
        log_file_handle = open(self.robot.logging_process_log_path, 'w')
        try:
            self.robot.logging_instance = subprocess.Popen(
                cmd,
                stdout=log_file_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                preexec_fn=os.setsid,
            )
            # Keep handle to close later
            self.robot._logging_process_log_fh = log_file_handle
        except Exception:
            log_file_handle.close()
            logger.exception("Failed to start ros2 bag record process")
            raise

        # Wait briefly for recorder to initialize and create output directory
        start_time = time.time()
        readiness_timeout_seconds = 10.0
        while time.time() - start_time < readiness_timeout_seconds:
            if os.path.isdir(self.robot.logging_dir):
                break
            if self.robot.logging_instance.poll() is not None:
                # Process exited early; break to let caller inspect
                break
            time.sleep(0.2)
        else:
            logger.warning("ros2 bag directory not detected within %.1fs: %s", readiness_timeout_seconds, self.robot.logging_dir)
        logger.info("ros2 bag record started with PID %s; logs: %s", self.robot.logging_instance.pid, self.robot.logging_process_log_path)
        
    def stop_logging(self) -> str:
        ''' Terminates the logging process safely and returns the directory where the logs were stored. '''
        if not hasattr(self.robot, 'logging_instance') or self.robot.logging_instance is None:
            raise RuntimeError("no active logging process to stop")
        process = self.robot.logging_instance
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except Exception:
            logger.exception("Failed sending SIGINT to logging process group")
        # Wait gracefully, then escalate if needed
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("logging process did not stop on SIGINT; sending SIGTERM")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("logging process did not stop on SIGTERM; sending SIGKILL")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except Exception:
                    logger.exception("Failed to SIGKILL logging process group")
                process.wait()
        finally:
            # Close and cleanup log file handle if present
            fh: Optional[object] = getattr(self.robot, '_logging_process_log_fh', None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    logger.debug("failed closing process log handle", exc_info=True)
                delattr(self.robot, '_logging_process_log_fh')

        logger.info("Stopped ros2 bag record; exit code: %s", process.returncode)
        self.robot.logging_instance = None
        return self.robot.logging_dir
            
    def get_logging_data(self, logging_dir : str) -> dict[str, list[tuple[int, Any]]]:
        ''' Reads the recorded messages from the ROS2 bag, aggregating the data into a dictionary keyed by topic. '''
        if not logging_dir or not isinstance(logging_dir, str):
            raise ValueError("logging_dir must be a non-empty string")
        if not os.path.isdir(logging_dir):
            raise FileNotFoundError(f"logging directory does not exist: {logging_dir}")

        logger.info("Reading bag data from %s", logging_dir)
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=logging_dir, storage_id='mcap')
            converter_options = rosbag2_py.ConverterOptions('', '')
            reader.open(storage_options, converter_options)
            topic_types = reader.get_all_topics_and_types()
            type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
            log_content: dict[str, list[tuple[int, Any]]] = dict()
            message_counter = 0
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                if topic not in log_content.keys():
                    log_content[topic] = []
                log_content[topic].append((t, msg))
                message_counter += 1
            logger.info("Loaded %d messages across %d topics", message_counter, len(log_content))
            return log_content
        except Exception:
            logger.exception("Failed reading bag from %s", logging_dir)
            raise

    def delete_logging_data(self, logging_dir : str):
        ''' Deletes the entire directory containing the logging data. '''
        if not logging_dir or not isinstance(logging_dir, str):
            raise ValueError("logging_dir must be a non-empty string")
        if not os.path.exists(logging_dir):
            logger.warning("logging directory does not exist; nothing to delete: %s", logging_dir)
            return
        logger.info("Deleting logging directory: %s", logging_dir)
        try:
            shutil.rmtree(logging_dir)
        except Exception:
            logger.exception("Failed to delete logging directory: %s", logging_dir)
            raise

    def is_logging_active(self) -> bool:
        ''' Returns True if the rosbag recorder process appears to be running. '''
        if not hasattr(self.robot, 'logging_instance') or self.robot.logging_instance is None:
            return False
        return self.robot.logging_instance.poll() is None

    def current_logging_dir(self) -> Optional[str]:
        ''' Returns the last configured logging directory, if any. '''
        return getattr(self.robot, 'logging_dir', None)
