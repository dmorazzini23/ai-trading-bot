#!/usr/bin/env python3
"""
Process Management and Service Cleanup Script for AI Trading Bot
Addresses duplicate processes, failed services, and process conflicts.
"""

import logging
import os
import signal
import subprocess
import time
from datetime import UTC, datetime

# AI-AGENT-REF: Process management and service cleanup script

class ProcessManager:
    """Manage trading bot processes and services."""

    def __init__(self):
        self.logger = self._setup_logger()
        self.processes_info = []

    def _setup_logger(self) -> logging.Logger:
        """Setup process manager logger."""
        logger = logging.getLogger('process_manager')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def find_python_processes(self) -> list[dict]:
        """Find all Python processes related to trading bot."""
        processes = []

        try:
            # Use ps to find all Python processes
            result = subprocess.run(
                ['ps', 'aux', '--sort=-rss'],  # Sort by memory usage (descending)
                timeout=30,
                capture_output=True, text=True, check=True
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')

                for line in lines[1:]:  # Skip header
                    if 'python' in line.lower() and line.strip():
                        parts = line.split()
                        if len(parts) >= 11:
                            process_info = {
                                'user': parts[0],
                                'pid': int(parts[1]),
                                'cpu_percent': float(parts[2]),
                                'memory_percent': float(parts[3]),
                                'vsz_kb': int(parts[4]),
                                'rss_kb': int(parts[5]),
                                'memory_mb': int(parts[5]) / 1024,
                                'tty': parts[6],
                                'stat': parts[7],
                                'start': parts[8],
                                'time': parts[9],
                                'command': ' '.join(parts[10:])
                            }

                            # Check if it's trading-related
                            if self._is_trading_process(process_info['command']):
                                processes.append(process_info)

        except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
            self.logger.error(f"Failed to get process list: {e}")
        except ValueError as e:
            self.logger.error(f"Failed to parse process information: {e}")

        self.processes_info = processes
        return processes

    def _is_trading_process(self, command: str) -> bool:
        """Check if a command is related to trading bot."""
        trading_keywords = [
            'bot_engine.py',
            'ai_trading.main',
            'ai_trading/main.py',
            'trading',
            'alpaca',
            'runner.py'
        ]

        command_lower = command.lower()
        return any(keyword.lower() in command_lower for keyword in trading_keywords)

    def find_duplicate_processes(self) -> list[dict]:
        """Find duplicate trading processes."""
        if not self.processes_info:
            self.find_python_processes()

        duplicates = []
        seen_commands = {}

        for process in self.processes_info:
            cmd_key = self._normalize_command(process['command'])

            if cmd_key in seen_commands:
                # This is a duplicate
                duplicates.append({
                    'duplicate_process': process,
                    'original_process': seen_commands[cmd_key],
                    'command_key': cmd_key
                })
            else:
                seen_commands[cmd_key] = process

        return duplicates

    def _normalize_command(self, command: str) -> str:
        """Normalize command for duplicate detection."""
        # Remove common variations
        normalized = command.lower()

        # Remove absolute paths
        if 'ai_trading.main' in normalized or 'bot_engine.py' in normalized:
            if 'ai_trading.main' in normalized:
                return 'ai_trading_main'
            elif 'bot_engine.py' in normalized:
                return 'bot_engine'

        return normalized

    def kill_process(self, pid: int, force: bool = False) -> bool:
        """Kill a process by PID."""
        try:
            signal_to_send = signal.SIGKILL if force else signal.SIGTERM
            os.kill(pid, signal_to_send)

            # Wait a moment and check if process is gone
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if process still exists
                return False  # Process still running
            except OSError:
                return True  # Process is gone

        except OSError as e:
            if e.errno == 3:  # No such process
                return True  # Already gone
            self.logger.error(f"Failed to kill process {pid}: {e}")
            return False

    def cleanup_duplicate_processes(self, dry_run: bool = True) -> dict:
        """Clean up duplicate trading processes."""
        duplicates = self.find_duplicate_processes()

        cleanup_report = {
            'duplicates_found': len(duplicates),
            'processes_killed': [],
            'failed_kills': [],
            'dry_run': dry_run
        }

        if not duplicates:
            self.logger.info("No duplicate processes found")
            return cleanup_report

        for duplicate_info in duplicates:
            duplicate_proc = duplicate_info['duplicate_process']
            original_proc = duplicate_info['original_process']

            # Choose which process to keep (prefer lower memory usage or earlier start)
            if duplicate_proc['memory_mb'] > original_proc['memory_mb']:
                # Kill the duplicate (higher memory usage)
                target_pid = duplicate_proc['pid']
                target_proc = duplicate_proc
            else:
                # Kill the original, keep the duplicate
                target_pid = original_proc['pid']
                target_proc = original_proc

            self.logger.info(
                f"Target for termination: PID {target_pid} "
                f"({target_proc['memory_mb']:.1f}MB) - {target_proc['command'][:80]}..."
            )

            if not dry_run:
                if self.kill_process(target_pid):
                    cleanup_report['processes_killed'].append(target_proc)
                    self.logger.info(f"Successfully terminated process {target_pid}")
                else:
                    cleanup_report['failed_kills'].append(target_proc)
                    self.logger.error(f"Failed to terminate process {target_pid}")

        return cleanup_report

    def ensure_single_instance(self, pidfile_path: str = "/tmp/ai_trading_bot.pid") -> bool:
        """
        Ensure only one instance of the trading bot is running.
        
        Args:
            pidfile_path: Path to the PID file for instance tracking
            
        Returns:
            True if this instance can proceed, False if another instance is running
        """
        import atexit
        import fcntl

        try:
            # Try to open the PID file exclusively
            self.pidfile = open(pidfile_path, 'w')

            # Try to acquire an exclusive lock
            fcntl.flock(self.pidfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID to the file
            self.pidfile.write(str(os.getpid()))
            self.pidfile.flush()

            # Register cleanup function
            atexit.register(self._cleanup_pidfile)

            self.logger.info(f"Successfully acquired single instance lock (PID: {os.getpid()})")
            return True

        except OSError as e:
            if hasattr(self, 'pidfile'):
                self.pidfile.close()

            # Check if there's actually a running process
            if os.path.exists(pidfile_path):
                try:
                    with open(pidfile_path) as f:
                        existing_pid = int(f.read().strip())

                    # Check if the process is still running
                    try:
                        os.kill(existing_pid, 0)  # Signal 0 just checks if process exists
                        self.logger.error(f"Another trading bot instance is already running (PID: {existing_pid})")
                        return False
                    except OSError:
                        # Process doesn't exist, remove stale PID file
                        self.logger.info(f"Removing stale PID file for non-existent process {existing_pid}")
                        os.unlink(pidfile_path)
                        # Try again
                        return self.ensure_single_instance(pidfile_path)

                except (OSError, ValueError):
                    # Corrupted PID file, remove it and try again
                    self.logger.warning(f"Removing corrupted PID file: {pidfile_path}")
                    try:
                        os.unlink(pidfile_path)
                    except OSError:
                        pass
                    return self.ensure_single_instance(pidfile_path)

            self.logger.error(f"Failed to acquire single instance lock: {e}")
            return False

    def _cleanup_pidfile(self):
        """Clean up PID file on exit."""
        try:
            if hasattr(self, 'pidfile'):
                self.pidfile.close()
                # Note: PID file will be automatically removed when process exits
                # since we hold an exclusive lock on it
        except Exception as e:
            self.logger.debug(f"Error cleaning up PID file: {e}")

    def check_service_status(self) -> dict:
        """Check status of trading-related systemd services."""
        services = [
            'ai-trading-bot.service',
            'ai-trading-scheduler.service',
            'ai-trading-server.service',
            'dashboard.service'
        ]

        service_status = {}

        for service in services:
            try:
                result = subprocess.run(
                    ['systemctl', 'is-active', service],
                    timeout=30,
                    capture_output=True, text=True, check=True
                )

                status = result.stdout.strip()
                service_status[service] = {
                    'status': status,
                    'active': status == 'active',
                    'return_code': result.returncode
                }

                # Get more detailed info for failed services
                if status != 'active':
                    detail_result = subprocess.run(
                        ['systemctl', 'status', service, '--no-pager', '-l'],
                        timeout=30,
                        capture_output=True, text=True, check=True
                    )
                    service_status[service]['details'] = detail_result.stdout

            except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
                service_status[service] = {
                    'status': 'error',
                    'error': str(e),
                    'active': False
                }

        return service_status

    def fix_file_permissions(self, paths: list[str], target_user: str = 'aiuser') -> dict:
        """Fix file ownership and permissions."""
        permission_report = {
            'paths_checked': len(paths),
            'paths_fixed': [],
            'failed_fixes': []
        }

        for path in paths:
            try:
                if os.path.exists(path):
                    # Check current ownership
                    stat_info = os.stat(path)
                    current_uid = stat_info.st_uid
                    current_gid = stat_info.st_gid

                    # Get target user info
                    try:
                        import pwd
                        target_pwd = pwd.getpwnam(target_user)
                        target_uid = target_pwd.pw_uid
                        target_gid = target_pwd.pw_gid

                        if current_uid != target_uid or current_gid != target_gid:
                            # Need to change ownership
                            subprocess.run(
                                ['sudo', 'chown', f'{target_user}:{target_user}', path],
                                timeout=30,
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            permission_report['paths_fixed'].append(path)
                            self.logger.info(f"Fixed ownership of {path}")

                    except (KeyError, subprocess.SubprocessError) as e:
                        permission_report['failed_fixes'].append({
                            'path': path,
                            'error': str(e)
                        })
                        self.logger.error(f"Failed to fix permissions for {path}: {e}")

            except OSError as e:
                permission_report['failed_fixes'].append({
                    'path': path,
                    'error': str(e)
                })

        return permission_report

    def generate_process_report(self) -> dict:
        """Generate comprehensive process management report."""
        processes = self.find_python_processes()
        duplicates = self.find_duplicate_processes()
        service_status = self.check_service_status()

        # Calculate memory usage summary
        total_memory_mb = sum(p['memory_mb'] for p in processes)
        max_memory_process = max(processes, key=lambda p: p['memory_mb']) if processes else None

        report = {
            'timestamp': datetime.now(UTC).isoformat(),
            'process_summary': {
                'total_python_processes': len(processes),
                'trading_processes': len(processes),
                'total_memory_mb': total_memory_mb,
                'duplicate_processes': len(duplicates)
            },
            'processes': processes,
            'duplicates': duplicates,
            'service_status': service_status,
            'highest_memory_process': max_memory_process,
            'recommendations': self._generate_recommendations(processes, duplicates, service_status)
        }

        return report

    def _generate_recommendations(self, processes: list[dict], duplicates: list[dict],
                                 service_status: dict) -> list[str]:
        """Generate recommendations based on process analysis."""
        recommendations = []

        # Memory recommendations
        high_memory_processes = [p for p in processes if p['memory_mb'] > 500]
        if high_memory_processes:
            recommendations.append(
                f"HIGH PRIORITY: {len(high_memory_processes)} processes using >500MB memory. "
                "Consider memory optimization or process restart."
            )

        # Duplicate process recommendations
        if duplicates:
            recommendations.append(
                f"MEDIUM PRIORITY: {len(duplicates)} duplicate processes detected. "
                "Run cleanup to eliminate conflicts."
            )

        # Service recommendations
        failed_services = [name for name, info in service_status.items()
                          if not info.get('active', False)]
        if failed_services:
            recommendations.append(
                f"MEDIUM PRIORITY: {len(failed_services)} failed services: {', '.join(failed_services)}. "
                "Check service configuration and restart."
            )

        # Performance recommendations
        total_memory = sum(p['memory_mb'] for p in processes)
        if total_memory > 1000:
            recommendations.append(
                f"LOW PRIORITY: Total Python process memory usage: {total_memory:.1f}MB. "
                "Monitor for memory leaks."
            )

        return recommendations

    # AI-AGENT-REF: Add process lock mechanism to prevent multiple instances
    def acquire_process_lock(self, lock_file: str = "/tmp/ai_trading_bot.lock") -> bool:
        """
        Acquire a process lock to prevent multiple instances.
        
        Parameters
        ----------
        lock_file : str
            Path to the lock file
            
        Returns
        -------
        bool
            True if lock acquired successfully, False if another instance is running
        """
        try:
            import atexit

            # Check if lock file exists and contains a valid PID
            if os.path.exists(lock_file):
                try:
                    with open(lock_file) as f:
                        existing_pid = int(f.read().strip())

                    # Check if process with this PID is still running
                    try:
                        os.kill(existing_pid, 0)
                        # Process is still running
                        self.logger.warning(f"Another trading bot instance is running (PID: {existing_pid})")
                        return False
                    except OSError:
                        # Process is dead, remove stale lock file
                        os.remove(lock_file)
                        self.logger.info(f"Removed stale lock file for PID {existing_pid}")
                except (ValueError, OSError) as e:
                    # Invalid lock file, remove it
                    self.logger.warning(f"Invalid lock file {lock_file}: {e}, removing")
                    try:
                        os.remove(lock_file)
                    except OSError:
                        pass

            # Create new lock file
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))

            # Register cleanup function
            def cleanup_lock():
                try:
                    if os.path.exists(lock_file):
                        with open(lock_file) as f:
                            if int(f.read().strip()) == os.getpid():
                                os.remove(lock_file)
                                self.logger.info("Process lock released")
                except (OSError, ValueError):
                    pass

            atexit.register(cleanup_lock)
            self.logger.info(f"Process lock acquired (PID: {os.getpid()})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to acquire process lock: {e}")
            return False

    def check_multiple_instances(self) -> dict:
        """
        Check for multiple trading bot instances and provide recommendations.
        
        Returns
        -------
        Dict
            Report with instance count and recommendations
        """
        processes = self.find_python_processes()
        trading_processes = [p for p in processes if self._is_trading_process(p['command'])]

        report = {
            'total_instances': len(trading_processes),
            'processes': trading_processes,
            'multiple_instances': len(trading_processes) > 1,
            'recommendations': []
        }

        if len(trading_processes) > 1:
            report['recommendations'].append(
                f"CRITICAL: {len(trading_processes)} trading bot instances detected. "
                "This can cause race conditions and duplicate orders."
            )
            report['recommendations'].append(
                "Kill all but one instance immediately to prevent trading conflicts."
            )
        elif len(trading_processes) == 0:
            report['recommendations'].append(
                "No trading bot instances detected. System is ready to start."
            )
        else:
            report['recommendations'].append(
                "Single trading bot instance detected. Normal operation."
            )

        return report


def main():
    """Main process management function."""
    logging.info("AI Trading Bot - Process Management Tool")
    logging.info(str("=" * 50))

    manager = ProcessManager()

    # Generate comprehensive report
    report = manager.generate_process_report()

    logging.info("\nPROCESS SUMMARY:")
    logging.info(str(f"- Total Python processes: {report['process_summary']['total_python_processes']}"))
    logging.info(str(f"- Total memory usage: {report['process_summary']['total_memory_mb']:.1f}MB"))
    logging.info(str(f"- Duplicate processes: {report['process_summary']['duplicate_processes']}"))

    if report['processes']:
        logging.info("\nACTIVE TRADING PROCESSES:")
        for proc in report['processes']:
            logging.info(str(f"- PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {proc['command'][:80]}..."))

    if report['duplicates']:
        logging.info("\nDUPLICATE PROCESSES DETECTED:")
        for dup in report['duplicates']:
            orig = dup['original_process']
            dupl = dup['duplicate_process']
            logging.info(f"- Original: PID {orig['pid']} ({orig['memory_mb']:.1f}MB)")
            logging.info(f"- Duplicate: PID {dupl['pid']} ({dupl['memory_mb']:.1f}MB)")

    logging.info("\nSERVICE STATUS:")
    for service, status in report['service_status'].items():
        status_str = "✓ ACTIVE" if status['active'] else "✗ FAILED"
        logging.info(f"- {service}: {status_str}")

    logging.info("\nRECOMMENDATIONS:")
    for i, rec in enumerate(report['recommendations'], 1):
        logging.info(f"{i}. {rec}")

    # Save report
    timestamp = datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_file = f"process_report_{timestamp}.json"

    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    logging.info(f"\nReport saved to: {report_file}")

    # Interactive cleanup option
    if report['duplicates']:
        response = input(f"\nFound {len(report['duplicates'])} duplicate processes. Clean up? (y/N): ")
        if response.lower() == 'y':
            logging.info("Performing cleanup (dry run first)...")
            dry_run_result = manager.cleanup_duplicate_processes(dry_run=True)
            logging.info(f"Dry run: Would terminate {len(dry_run_result['processes_killed'])} processes")

            confirm = input("Proceed with actual cleanup? (y/N): ")
            if confirm.lower() == 'y':
                cleanup_result = manager.cleanup_duplicate_processes(dry_run=False)
                logging.info(f"Cleanup complete: {len(cleanup_result['processes_killed'])} processes terminated")

    return report


if __name__ == "__main__":
    main()
