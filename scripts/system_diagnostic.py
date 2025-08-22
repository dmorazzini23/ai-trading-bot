#!/usr/bin/env python3
"""
Comprehensive System Diagnostic Script for AI Trading Bot
Analyzes memory usage, performance bottlenecks, and system health.
Uses only built-in Python modules for maximum compatibility.
"""

import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

# AI-AGENT-REF: System diagnostic script for memory and performance analysis

class SystemDiagnostic:
    """Comprehensive system diagnostic and performance analyzer."""

    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup diagnostic logger."""
        logger = logging.getLogger('system_diagnostic')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def get_memory_info(self) -> dict:
        """Analyze current memory usage patterns."""
        memory_info = {}

        # Process memory information
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_info['process_memory'] = {
                'max_rss_mb': usage.ru_maxrss / 1024 / 1024,  # Convert to MB
                'user_time': usage.ru_utime,
                'system_time': usage.ru_stime,
                'page_faults': usage.ru_majflt,
                'voluntary_context_switches': usage.ru_nvcsw,
                'involuntary_context_switches': usage.ru_nivcsw
            }
        except ImportError:
            memory_info['process_memory'] = {'error': 'resource module not available'}

        # System memory information
        try:
            # Read /proc/meminfo for detailed memory information
            with open('/proc/meminfo') as f:
                meminfo = {}
                for line in f:
                    key, value = line.split(':', 1)
                    meminfo[key.strip()] = value.strip()

                memory_info['system_memory'] = {
                    'total_mb': int(meminfo['MemTotal'].split()[0]) / 1024,
                    'available_mb': int(meminfo['MemAvailable'].split()[0]) / 1024,
                    'free_mb': int(meminfo['MemFree'].split()[0]) / 1024,
                    'buffers_mb': int(meminfo['Buffers'].split()[0]) / 1024,
                    'cached_mb': int(meminfo['Cached'].split()[0]) / 1024,
                    'swap_total_mb': int(meminfo['SwapTotal'].split()[0]) / 1024,
                    'swap_free_mb': int(meminfo['SwapFree'].split()[0]) / 1024,
                    'swap_used_mb': (int(meminfo['SwapTotal'].split()[0]) -
                                   int(meminfo['SwapFree'].split()[0])) / 1024
                }
        except (FileNotFoundError, KeyError, ValueError) as e:
            memory_info['system_memory'] = {'error': str(e)}

        return memory_info

    def check_python_processes(self) -> list[dict]:
        """Identify all Python processes and their resource usage."""
        processes = []

        try:
            # Use ps to find Python processes
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=30, check=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines[1:]:  # Skip header
                    if 'python' in line.lower() and line.strip():
                        parts = line.split()
                        if len(parts) >= 11:
                            processes.append({
                                'user': parts[0],
                                'pid': parts[1],
                                'cpu_percent': parts[2],
                                'memory_percent': parts[3],
                                'memory_mb': float(parts[5]) / 1024,  # Convert KB to MB
                                'command': ' '.join(parts[10:])
                            })
        except (subprocess.SubprocessError, ValueError) as e:
            processes.append({'error': str(e)})

        return processes

    def analyze_garbage_collection(self) -> dict:
        """Analyze garbage collection statistics and performance."""
        gc_info = {
            'enabled': gc.isenabled(),
            'counts': gc.get_count(),
            'thresholds': gc.get_threshold(),
            'stats': gc.get_stats() if hasattr(gc, 'get_stats') else None
        }

        # Force garbage collection and measure time
        start_time = time.time()
        collected = gc.collect()
        gc_time = time.time() - start_time

        gc_info['collection_test'] = {
            'objects_collected': collected,
            'collection_time_ms': gc_time * 1000
        }

        # Get referrers for large objects
        try:
            all_objects = gc.get_objects()
            gc_info['object_counts'] = {
                'total_objects': len(all_objects),
                'dict_objects': len([obj for obj in all_objects if isinstance(obj, dict)]),
                'list_objects': len([obj for obj in all_objects if isinstance(obj, list)]),
                'function_objects': len([obj for obj in all_objects if callable(obj)])
            }
        except (ValueError, TypeError) as e:
            gc_info['object_counts'] = {'error': str(e)}

        return gc_info

    def check_file_handles(self) -> dict:
        """Check open file handles and potential leaks."""
        file_info = {}

        try:
            # Count open file descriptors
            proc_fd = f'/proc/{os.getpid()}/fd'
            if os.path.exists(proc_fd):
                fd_count = len(os.listdir(proc_fd))
                file_info['open_file_descriptors'] = fd_count

                # List some file descriptors
                file_info['sample_fds'] = []
                for fd in os.listdir(proc_fd)[:10]:  # Show first 10
                    try:
                        link = os.readlink(os.path.join(proc_fd, fd))
                        file_info['sample_fds'].append(f'{fd}: {link}')
                    except OSError:
                        continue
        except (ValueError, TypeError) as e:
            file_info['error'] = str(e)

        return file_info

    def check_thread_usage(self) -> dict:
        """Analyze thread usage and potential issues."""
        thread_info = {
            'active_threads': threading.active_count(),
            'current_thread': threading.current_thread().name,
            'main_thread_alive': threading.main_thread().is_alive()
        }

        # List all threads
        threads = []
        for thread in threading.enumerate():
            threads.append({
                'name': thread.name,
                'daemon': thread.daemon,
                'alive': thread.is_alive(),
                'ident': thread.ident
            })

        thread_info['all_threads'] = threads

        return thread_info

    def check_modules_loaded(self) -> dict:
        """Check loaded modules and their memory impact."""
        module_info = {
            'total_modules': len(sys.modules),
            'builtin_modules': len(sys.builtin_module_names),
        }

        # Categorize modules
        large_modules = []
        trading_modules = []

        for name, module in sys.modules.items():
            if module is None:
                continue

            # Check for trading-related modules
            if any(keyword in name.lower() for keyword in
                   ['trading', 'alpaca', 'pandas', 'numpy', 'sklearn', 'ai_trading']):
                trading_modules.append(name)

            # Try to estimate module size (this is approximate)
            try:
                if hasattr(module, '__dict__'):
                    attrs = len(module.__dict__)
                    if attrs > 100:  # Arbitrary threshold for "large"
                        large_modules.append((name, attrs))
            except (AttributeError, ImportError):
                continue

        module_info['trading_modules'] = trading_modules
        module_info['large_modules'] = sorted(large_modules, key=lambda x: x[1], reverse=True)[:20]

        return module_info

    def check_environment_variables(self) -> dict:
        """Check environment variables for potential issues."""
        env_info = {}

        # Check Python-specific environment variables
        python_env_vars = [
            'PYTHONPATH', 'PYTHONHOME', 'PYTHON_ENABLE_STACK_TRAMPOLINES',
            'PYTEST_RUNNING', 'TRADING_ENV'
        ]

        for var in python_env_vars:
            env_info[var] = os.environ.get(var, 'Not set')

        # Check memory-related environment variables
        memory_env_vars = ['MALLOC_TRIM_THRESHOLD_', 'MALLOC_MMAP_THRESHOLD_']
        for var in memory_env_vars:
            env_info[var] = os.environ.get(var, 'Not set')

        return env_info

    def check_disk_usage(self) -> dict:
        """Check disk usage and temporary file accumulation."""
        disk_info = {}

        try:
            import shutil

            # Check current directory usage
            total, used, free = shutil.disk_usage('.')
            disk_info['current_dir'] = {
                'total_gb': total / (1024**3),
                'used_gb': used / (1024**3),
                'free_gb': free / (1024**3),
                'usage_percent': (used / total) * 100
            }

            # Check for large files in current directory
            large_files = []
            for root, dirs, files in os.walk('.'):
                # Skip .git and other version control directories
                dirs[:] = [d for d in dirs if not d.startswith('.git')]

                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        size = os.path.getsize(filepath)
                        if size > 10 * 1024 * 1024:  # Files larger than 10MB
                            large_files.append((filepath, size / (1024**2)))  # Size in MB
                    except OSError:
                        continue

            disk_info['large_files'] = sorted(large_files, key=lambda x: x[1], reverse=True)[:10]

        except (ValueError, TypeError) as e:
            disk_info['error'] = str(e)

        return disk_info

    def run_full_diagnostic(self) -> dict:
        """Run complete system diagnostic."""
        self.logger.info("Starting comprehensive system diagnostic...")

        diagnostic_results = {
            'timestamp': datetime.now(UTC).isoformat(),
            'python_version': sys.version,
            'platform': sys.platform,
            'diagnostic_runtime_seconds': 0
        }

        # Run all diagnostic checks
        checks = [
            ('memory_analysis', self.get_memory_info),
            ('process_analysis', self.check_python_processes),
            ('garbage_collection', self.analyze_garbage_collection),
            ('file_handles', self.check_file_handles),
            ('thread_analysis', self.check_thread_usage),
            ('module_analysis', self.check_modules_loaded),
            ('environment_variables', self.check_environment_variables),
            ('disk_usage', self.check_disk_usage)
        ]

        for check_name, check_func in checks:
            self.logger.info(f"Running {check_name}...")
            try:
                start_time = time.time()
                result = check_func()
                check_time = time.time() - start_time

                diagnostic_results[check_name] = result
                diagnostic_results[f'{check_name}_time_ms'] = check_time * 1000

            except (ValueError, TypeError) as e:
                self.logger.error(f"Error in {check_name}: {str(e)}")
                diagnostic_results[check_name] = {'error': str(e)}

        diagnostic_results['diagnostic_runtime_seconds'] = time.time() - self.start_time

        return diagnostic_results

    def generate_recommendations(self, results: dict) -> list[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Memory recommendations
        if 'memory_analysis' in results:
            memory = results['memory_analysis']

            if 'system_memory' in memory and isinstance(memory['system_memory'], dict):
                sys_mem = memory['system_memory']
                if 'swap_used_mb' in sys_mem and sys_mem['swap_used_mb'] > 100:
                    recommendations.append(
                        f"HIGH PRIORITY: Reduce swap usage ({sys_mem['swap_used_mb']:.1f}MB). "
                        "Consider increasing available RAM or reducing memory usage."
                    )

                if ('total_mb' in sys_mem and 'available_mb' in sys_mem and
                    sys_mem['available_mb'] / sys_mem['total_mb'] < 0.1):
                    recommendations.append(
                        "MEDIUM PRIORITY: Low available memory. "
                        "Consider garbage collection optimization."
                    )

        # Process recommendations
        if 'process_analysis' in results and isinstance(results['process_analysis'], list):
            python_processes = [p for p in results['process_analysis'] if isinstance(p, dict) and 'pid' in p]
            if len(python_processes) > 1:
                recommendations.append(
                    f"MEDIUM PRIORITY: Multiple Python processes detected ({len(python_processes)}). "
                    "Consider consolidating or implementing proper process management."
                )

        # Garbage collection recommendations
        if 'garbage_collection' in results:
            gc_data = results['garbage_collection']
            if ('object_counts' in gc_data and isinstance(gc_data['object_counts'], dict) and
                'total_objects' in gc_data['object_counts'] and
                gc_data['object_counts']['total_objects'] > 100000):
                recommendations.append(
                    "MEDIUM PRIORITY: High object count detected. "
                    "Consider implementing regular garbage collection cycles."
                )

        # Thread recommendations
        if 'thread_analysis' in results:
            thread_data = results['thread_analysis']
            if 'active_threads' in thread_data and thread_data['active_threads'] > 10:
                recommendations.append(
                    f"LOW PRIORITY: High thread count ({thread_data['active_threads']}). "
                    "Monitor for thread leaks."
                )

        # File handle recommendations
        if 'file_handles' in results:
            fh_data = results['file_handles']
            if 'open_file_descriptors' in fh_data and fh_data['open_file_descriptors'] > 100:
                recommendations.append(
                    f"MEDIUM PRIORITY: High file descriptor count ({fh_data['open_file_descriptors']}). "
                    "Check for file handle leaks."
                )

        return recommendations


def main():
    """Main diagnostic function."""
    logging.info("AI Trading Bot - System Diagnostic Tool")
    logging.info(str("=" * 50))

    diagnostic = SystemDiagnostic()
    results = diagnostic.run_full_diagnostic()

    # Generate recommendations
    recommendations = diagnostic.generate_recommendations(results)

    # Output results
    logging.info("\nDIAGNOSTIC RESULTS:")
    logging.info(str(json.dumps(results, indent=2)))

    logging.info("\nRECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        logging.info(f"{i}. {rec}")

    # Save results to file
    timestamp = datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = f"diagnostic_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'recommendations': recommendations
        }, f, indent=2)

    logging.info(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
