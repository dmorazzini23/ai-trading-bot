#!/usr/bin/env python3
"""Trading Execution Debugging CLI Tool

This command-line tool helps diagnose and debug trading execution issues
using the enhanced debugging system.

Usage:
    python debug_cli.py status                    # Show overall status
    python debug_cli.py executions [--limit 10]  # Show recent executions
    python debug_cli.py positions                 # Check position discrepancies  
    python debug_cli.py pnl [symbol]             # Show PnL breakdown
    python debug_cli.py trace [correlation_id]   # Trace execution timeline
    python debug_cli.py health                   # Run health check
"""

import argparse
import os
import sys
from datetime import datetime, timezone

# Set required environment variables for CLI usage
os.environ.setdefault('ALPACA_API_KEY', 'cli_key')
os.environ.setdefault('ALPACA_SECRET_KEY', 'cli_secret')
os.environ.setdefault('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
os.environ.setdefault('WEBHOOK_SECRET', 'cli_webhook')
os.environ.setdefault('FLASK_PORT', '9000')


def cmd_status():
    """Show overall execution system status."""
    print("📊 EXECUTION SYSTEM STATUS")
    print("=" * 40)
    
    try:
        from ai_trading.execution import get_execution_statistics, get_reconciliation_statistics
        
        # Execution statistics
        exec_stats = get_execution_statistics()
        print(f"Active Orders: {exec_stats['active_orders']}")
        print(f"Recent Successes: {exec_stats['recent_successes']}")
        print(f"Recent Failures: {exec_stats['recent_failures']}")
        print(f"Success Rate: {exec_stats['success_rate']:.1%}")
        
        # Position reconciliation
        recon_stats = get_reconciliation_statistics()
        print(f"Position Discrepancies: {recon_stats['current_discrepancies']}")
        print(f"Bot Positions: {recon_stats['bot_positions_count']}")
        
        # Overall health indicator
        if exec_stats['success_rate'] > 0.95 and recon_stats['current_discrepancies'] == 0:
            print("🟢 System Status: HEALTHY")
        elif exec_stats['success_rate'] > 0.8:
            print("🟡 System Status: CAUTION")
        else:
            print("🔴 System Status: ISSUES DETECTED")
            
    except Exception as e:
        print(f"❌ Error getting status: {e}")


def cmd_executions(limit=10):
    """Show recent executions."""
    print(f"📋 RECENT EXECUTIONS (last {limit})")
    print("=" * 40)
    
    try:
        from ai_trading.execution import get_debug_tracker
        
        tracker = get_debug_tracker()
        
        # Show recent successful executions
        successes = tracker.get_recent_executions(limit=limit)
        if successes:
            print(f"✅ SUCCESSFUL EXECUTIONS ({len(successes)}):")
            for exec_data in successes[-limit:]:
                symbol = exec_data.get('symbol', 'Unknown')
                side = exec_data.get('side', 'Unknown')
                qty = exec_data.get('qty', 'Unknown')
                start_time = exec_data.get('start_time', 'Unknown')
                print(f"  {symbol} {side} {qty} shares at {start_time}")
        
        # Show recent failures
        failures = tracker.get_failed_executions(limit=limit)
        if failures:
            print(f"\n❌ FAILED EXECUTIONS ({len(failures)}):")
            for exec_data in failures[-limit:]:
                symbol = exec_data.get('symbol', 'Unknown')
                side = exec_data.get('side', 'Unknown')
                qty = exec_data.get('qty', 'Unknown')
                error = exec_data.get('error', 'Unknown error')
                print(f"  {symbol} {side} {qty} shares - {error}")
        
        # Show active orders
        active = tracker.get_active_orders()
        if active:
            print(f"\n⏳ ACTIVE ORDERS ({len(active)}):")
            for correlation_id, order_data in active.items():
                symbol = order_data.get('symbol', 'Unknown')
                side = order_data.get('side', 'Unknown')
                status = order_data.get('status', 'Unknown')
                print(f"  {symbol} {side} - {status} (ID: {correlation_id[:8]}...)")
        
        if not successes and not failures and not active:
            print("No recent executions found.")
            
    except Exception as e:
        print(f"❌ Error getting executions: {e}")


def cmd_positions():
    """Check position discrepancies."""
    print("🏦 POSITION RECONCILIATION")
    print("=" * 40)
    
    try:
        from ai_trading.execution import (
            get_position_reconciler, force_position_reconciliation,
            get_position_discrepancies
        )
        
        reconciler = get_position_reconciler()
        
        # Show current bot positions
        bot_positions = reconciler.get_bot_positions()
        print(f"🤖 BOT POSITIONS ({len(bot_positions)}):")
        if bot_positions:
            for symbol, qty in bot_positions.items():
                print(f"  {symbol}: {qty} shares")
        else:
            print("  No positions tracked by bot")
        
        # Force reconciliation check
        print(f"\n🔄 Running reconciliation check...")
        discrepancies = force_position_reconciliation()
        
        if discrepancies:
            print(f"\n⚠️  DISCREPANCIES FOUND ({len(discrepancies)}):")
            for disc in discrepancies:
                severity_icon = "🔴" if disc.severity == "high" else "🟡" if disc.severity == "medium" else "🟢"
                print(f"  {severity_icon} {disc.symbol}:")
                print(f"    Bot: {disc.bot_qty} shares")
                print(f"    Broker: {disc.broker_qty} shares")
                print(f"    Difference: {disc.difference}")
                print(f"    Type: {disc.discrepancy_type}")
                print(f"    Severity: {disc.severity}")
        else:
            print(f"\n✅ No discrepancies found - positions are in sync")
            
    except Exception as e:
        print(f"❌ Error checking positions: {e}")


def cmd_pnl(symbol=None):
    """Show PnL breakdown."""
    if symbol:
        print(f"💰 PnL BREAKDOWN FOR {symbol}")
    else:
        print("💰 PORTFOLIO PnL SUMMARY")
    print("=" * 40)
    
    try:
        from ai_trading.execution import (
            get_pnl_attributor, get_symbol_pnl_breakdown, 
            get_portfolio_pnl_summary, explain_recent_pnl_changes
        )
        
        attributor = get_pnl_attributor()
        
        if symbol:
            # Show specific symbol breakdown
            breakdown = get_symbol_pnl_breakdown(symbol)
            
            if breakdown:
                total_realized = 0
                print(f"📈 {symbol} PnL BY SOURCE:")
                for source, amount in breakdown.items():
                    if amount != 0:
                        icon = "💚" if amount > 0 else "💸" if amount < 0 else "⚪"
                        print(f"  {icon} {source}: ${amount:+.2f}")
                        if source != 'unrealized':
                            total_realized += amount
                
                print(f"\n📊 SUMMARY:")
                print(f"  Total Realized: ${total_realized:+.2f}")
                if 'unrealized' in breakdown:
                    print(f"  Unrealized: ${breakdown['unrealized']:+.2f}")
                    print(f"  Net PnL: ${total_realized + breakdown['unrealized']:+.2f}")
                
                # Explain recent changes
                explanation = explain_recent_pnl_changes(symbol, minutes=60)
                if explanation['total_change'] != 0:
                    print(f"\n📝 RECENT CHANGES (last hour):")
                    print(f"  {explanation['explanation']}")
                    print(f"  Total change: ${explanation['total_change']:+.2f}")
            else:
                print(f"No PnL data found for {symbol}")
        else:
            # Show portfolio summary
            summary = get_portfolio_pnl_summary()
            
            print(f"📊 PORTFOLIO SUMMARY:")
            print(f"  Total Realized PnL: ${summary['total_realized_pnl']:+.2f}")
            print(f"  Total Unrealized PnL: ${summary['total_unrealized_pnl']:+.2f}")
            print(f"  Net PnL: ${summary['total_pnl']:+.2f}")
            
            if summary['pnl_by_source']:
                print(f"\n💰 PnL BY SOURCE:")
                for source, amount in summary['pnl_by_source'].items():
                    if amount != 0:
                        icon = "💚" if amount > 0 else "💸"
                        print(f"  {icon} {source}: ${amount:+.2f}")
            
            if summary['today_pnl']:
                print(f"\n📅 TODAY'S PnL:")
                for source, amount in summary['today_pnl'].items():
                    if amount != 0:
                        print(f"  {source}: ${amount:+.2f}")
                        
    except Exception as e:
        print(f"❌ Error getting PnL data: {e}")


def cmd_trace(correlation_id):
    """Trace execution timeline for correlation ID."""
    print(f"🔍 EXECUTION TRACE: {correlation_id}")
    print("=" * 40)
    
    try:
        from ai_trading.execution import get_debug_tracker
        
        tracker = get_debug_tracker()
        timeline = tracker.get_execution_timeline(correlation_id)
        
        if timeline:
            print(f"📅 TIMELINE ({len(timeline)} events):")
            for i, event in enumerate(timeline, 1):
                timestamp = event['timestamp']
                phase = event['phase']
                data = event.get('data', {})
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S.%f')[:-3]  # HH:MM:SS.mmm
                except:
                    time_str = timestamp
                
                print(f"\n  {i:2d}. {time_str} - {phase.upper()}")
                
                # Show relevant data
                if data:
                    for key, value in data.items():
                        if key not in ['timestamp', 'correlation_id']:
                            print(f"      {key}: {value}")
        else:
            print(f"❌ No timeline found for correlation ID: {correlation_id}")
            print("Available correlation IDs:")
            
            # Show available correlation IDs
            active_orders = tracker.get_active_orders()
            recent_executions = tracker.get_recent_executions(limit=5)
            
            all_ids = set()
            all_ids.update(active_orders.keys())
            for exec_data in recent_executions:
                if 'correlation_id' in exec_data:
                    all_ids.add(exec_data['correlation_id'])
            
            for correlation_id in sorted(all_ids):
                print(f"  {correlation_id}")
                
    except Exception as e:
        print(f"❌ Error tracing execution: {e}")


def cmd_health():
    """Run comprehensive health check."""
    print("🏥 SYSTEM HEALTH CHECK")
    print("=" * 40)
    
    try:
        from ai_trading.execution import (
            get_execution_statistics, get_reconciliation_statistics,
            get_pnl_attribution_stats, force_position_reconciliation
        )
        
        issues = []
        
        # Check execution health
        print("🔄 Checking execution system...")
        exec_stats = get_execution_statistics()
        
        if exec_stats['success_rate'] < 0.9:
            issues.append(f"Low execution success rate: {exec_stats['success_rate']:.1%}")
        
        if exec_stats['active_orders'] > 10:
            issues.append(f"High number of active orders: {exec_stats['active_orders']}")
        
        # Check position reconciliation
        print("🏦 Checking position reconciliation...")
        discrepancies = force_position_reconciliation()
        
        high_severity_discrepancies = [d for d in discrepancies if d.severity == 'high']
        if high_severity_discrepancies:
            issues.append(f"High severity position discrepancies: {len(high_severity_discrepancies)}")
        
        # Check PnL attribution
        print("💰 Checking PnL attribution...")
        pnl_stats = get_pnl_attribution_stats()
        
        if pnl_stats.get('total_events', 0) == 0:
            issues.append("No PnL events recorded - may indicate tracking issues")
        
        # Report results
        print(f"\n📋 HEALTH CHECK RESULTS:")
        if issues:
            print(f"⚠️  {len(issues)} ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ No issues detected - system appears healthy")
        
        # Show summary stats
        print(f"\n📊 SYSTEM METRICS:")
        print(f"  Execution success rate: {exec_stats['success_rate']:.1%}")
        print(f"  Active orders: {exec_stats['active_orders']}")
        print(f"  Position discrepancies: {len(discrepancies)}")
        print(f"  PnL events tracked: {pnl_stats.get('total_events', 0)}")
        
    except Exception as e:
        print(f"❌ Error running health check: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Trading Execution Debugging CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show overall execution system status')
    
    # Executions command
    exec_parser = subparsers.add_parser('executions', help='Show recent executions')
    exec_parser.add_argument('--limit', type=int, default=10, help='Limit number of results')
    
    # Positions command
    subparsers.add_parser('positions', help='Check position discrepancies')
    
    # PnL command
    pnl_parser = subparsers.add_parser('pnl', help='Show PnL breakdown')
    pnl_parser.add_argument('symbol', nargs='?', help='Symbol to analyze (optional)')
    
    # Trace command
    trace_parser = subparsers.add_parser('trace', help='Trace execution timeline')
    trace_parser.add_argument('correlation_id', help='Correlation ID to trace')
    
    # Health command
    subparsers.add_parser('health', help='Run comprehensive health check')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'status':
            cmd_status()
        elif args.command == 'executions':
            cmd_executions(args.limit)
        elif args.command == 'positions':
            cmd_positions()
        elif args.command == 'pnl':
            cmd_pnl(args.symbol)
        elif args.command == 'trace':
            cmd_trace(args.correlation_id)
        elif args.command == 'health':
            cmd_health()
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()