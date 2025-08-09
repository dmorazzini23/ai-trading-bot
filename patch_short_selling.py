#!/usr/bin/env python3
"""
Script to complete the short selling logic fix by updating the specific lines
that couldn't be updated with str_replace due to duplicates.
"""


def patch_short_selling_logic():
    """Apply the short selling logic patches."""
    
    # Read the current file
    with open('trade_execution.py', 'r') as f:
        lines = f.readlines()
    
    # Find and update the first occurrence (sync execute_order)
    for i, line in enumerate(lines):
        if i == 1799 and 'if side.lower() == "sell" and existing == 0:' in line:
            # Replace the simple logic with enhanced short selling logic
            lines[i+1] = '            # No existing position - check if this is a valid short selling request\n'
            lines[i+2] = '            if hasattr(self.ctx, \'allow_short_selling\') and self.ctx.allow_short_selling:\n'
            lines[i+3] = '                if self._validate_short_selling(api, symbol, remaining):\n'
            lines[i+4] = '                    self.logger.info("SHORT_SELLING_INITIATED | symbol=%s qty=%d", symbol, remaining)\n'
            lines[i+5] = '                else:\n'
            lines[i+6] = '                    self.logger.info("SHORT_SELLING_REJECTED | symbol=%s qty=%d", symbol, remaining)\n'
            lines[i+7] = '                    return None\n'
            lines[i+8] = '            else:\n'
            lines[i+9] = '                self.logger.info("SKIP_NO_POSITION | no shares to sell, skipping")\n'
            lines[i+10] = '                return None\n'
            print(f"Updated sync execute_order at line {i+1}")
            break
    
    # Find and update the second occurrence (async execute_order_async)
    for i, line in enumerate(lines):
        if i == 2087 and 'if side.lower() == "sell" and existing == 0:' in line:
            # Replace the simple logic with enhanced short selling logic
            lines[i+1] = '            # No existing position - check if this is a valid short selling request\n'
            lines[i+2] = '            if hasattr(self.ctx, \'allow_short_selling\') and self.ctx.allow_short_selling:\n'
            lines[i+3] = '                if self._validate_short_selling(api, symbol, remaining):\n'
            lines[i+4] = '                    self.logger.info("SHORT_SELLING_INITIATED | symbol=%s qty=%d", symbol, remaining)\n'
            lines[i+5] = '                else:\n'
            lines[i+6] = '                    self.logger.info("SHORT_SELLING_REJECTED | symbol=%s qty=%d", symbol, remaining)\n'
            lines[i+7] = '                    return None\n'
            lines[i+8] = '            else:\n'
            lines[i+9] = '                self.logger.info("SKIP_NO_POSITION | no shares to sell, skipping")\n'
            lines[i+10] = '                return None\n'
            print(f"Updated async execute_order_async at line {i+1}")
            break
    
    # Write the updated file
    with open('trade_execution.py', 'w') as f:
        f.writelines(lines)
    
    print("Short selling logic patches applied successfully!")

if __name__ == '__main__':
    patch_short_selling_logic()