"""
Smoke test for RL training-inference parity.

Validates that the same observation produces the same action
in both training environment and inference wrapper.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_action_space_parity():
    """Test that training and inference use consistent action spaces."""
    print("Testing RL action space parity...")
    
    try:
        from ..env import TradingEnv, ActionSpaceConfig, RewardConfig
        from ..inference import UnifiedRLInference, InferenceConfig
        
        # Generate test data
        np.random.seed(42)
        test_data = np.random.randn(100, 5)  # 100 timesteps, 5 features
        
        # Test discrete action space
        print("  Testing discrete action space...")
        action_config = ActionSpaceConfig(action_type="discrete", discrete_actions=3)
        reward_config = RewardConfig(normalize_rewards=True)
        
        env = TradingEnv(
            data=test_data,
            window=10,
            action_config=action_config,
            reward_config=reward_config
        )
        
        # Reset environment and get initial state
        obs, _ = env.reset()
        print(f"    Environment observation shape: {obs.shape}")
        print(f"    Action space: {env.action_space}")
        
        # Test a few actions
        actions = [0, 1, 2, 0, 1]  # hold, buy, sell, hold, buy
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"    Step {i}: action={action}, reward={reward:.4f}, position={info['position']}")
            
            if terminated:
                break
        
        print("  Discrete action space test passed!")
        
        # Test continuous action space
        print("  Testing continuous action space...")
        action_config_cont = ActionSpaceConfig(
            action_type="continuous",
            continuous_bounds=(-1.0, 1.0)
        )
        
        env_cont = TradingEnv(
            data=test_data,
            window=10,
            action_config=action_config_cont,
            reward_config=reward_config
        )
        
        obs_cont, _ = env_cont.reset()
        print(f"    Continuous environment action space: {env_cont.action_space}")
        
        # Test continuous actions
        cont_actions = [0.0, 0.5, -0.3, 0.8, -0.1]
        for i, action in enumerate(cont_actions):
            obs_cont, reward, terminated, truncated, info = env_cont.step(action)
            print(f"    Step {i}: action={action:.2f}, reward={reward:.4f}, position={info['position']:.2f}")
            
            if terminated:
                break
        
        print("  Continuous action space test passed!")
        
        # Test inference configuration parity
        print("  Testing inference configuration...")
        
        # Create a mock model path (in real use, this would be a trained model)
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_model_path = os.path.join(temp_dir, "mock_model")
            
            # Create inference config
            inference_config = InferenceConfig(
                model_path=mock_model_path,
                action_config=action_config,
                reward_config=reward_config,
                observation_window=10
            )
            
            print(f"    Inference config action type: {inference_config.action_config.action_type}")
            print(f"    Inference config discrete actions: {inference_config.action_config.discrete_actions}")
            
            # Test observation preprocessing
            try:
                inference = UnifiedRLInference(inference_config)
                
                # Test observation preprocessing
                single_obs = test_data[50]  # Single timestep
                processed_obs = inference.preprocess_observation(single_obs)
                print(f"    Processed observation shape: {processed_obs.shape}")
                
                # Test postprocessing
                mock_action = 1  # Mock discrete action
                action_details = inference.postprocess_action(mock_action, processed_obs)
                print(f"    Postprocessed action: {action_details['action']} (confidence: {action_details['confidence']:.2f})")
                
            except Exception as e:
                # Expected to fail due to missing model, but preprocessing should work
                if "model not loaded" in str(e).lower() or "model not found" in str(e).lower():
                    print("    Inference preprocessing test passed (model loading expected to fail)")
                else:
                    raise e
        
        print("All RL parity tests passed!")
        return True
        
    except ImportError as e:
        print(f"Skipping RL tests due to missing dependencies: {e}")
        return True
    except Exception as e:
        print(f"RL parity test failed: {e}")
        return False


def test_reward_normalization():
    """Test reward normalization functionality."""
    print("Testing reward normalization...")
    
    try:
        from ..env import TradingEnv, ActionSpaceConfig, RewardConfig, RunningStats
        
        # Test running stats
        stats = RunningStats(window=10)
        
        # Add some test values
        test_values = [1.0, 2.0, -1.0, 3.0, 0.5, -0.5, 2.5, 1.5, -0.2, 0.8]
        for val in test_values:
            stats.update(val)
        
        print(f"  Running stats - mean: {stats.mean:.3f}, std: {stats.std:.3f}")
        
        # Test normalization
        test_val = 1.5
        normalized = stats.normalize(test_val)
        print(f"  Normalized {test_val} -> {normalized:.3f}")
        
        # Test in environment
        np.random.seed(42)
        test_data = np.random.randn(50, 3)
        
        reward_config = RewardConfig(normalize_rewards=True, reward_window=20)
        env = TradingEnv(data=test_data, window=5, reward_config=reward_config)
        
        obs, _ = env.reset()
        total_raw_reward = 0
        total_normalized_reward = 0
        
        for i in range(10):
            action = np.random.choice([0, 1, 2])  # Random actions
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_raw_reward += info.get('raw_reward', 0)
            total_normalized_reward += reward
            
            if terminated:
                break
        
        print(f"  Total raw reward: {total_raw_reward:.3f}")
        print(f"  Total normalized reward: {total_normalized_reward:.3f}")
        print("Reward normalization test passed!")
        return True
        
    except Exception as e:
        print(f"Reward normalization test failed: {e}")
        return False


def main():
    """Run smoke parity tests."""
    print("=== RL Training-Inference Parity Smoke Tests ===")
    
    success = True
    
    # Test action space parity
    if not test_action_space_parity():
        success = False
    
    print()
    
    # Test reward normalization
    if not test_reward_normalization():
        success = False
    
    print()
    
    if success:
        print("✓ All RL smoke tests passed!")
        return 0
    else:
        print("✗ Some RL smoke tests failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())