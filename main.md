# 复现ABS强化学习框架指南

ABS (Agile But Safe) 是一个用于四足机器人的高速无碰撞运动控制框架。下面我将指导你如何复现这个强化学习框架。

## 1. 环境配置

首先确保你已经配置好以下环境：

```bash
# 基础依赖
conda create -n abs python=3.8
conda activate abs
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy scipy matplotlib gym

# Isaac Gym (必须)
git clone https://github.com/NVIDIA/isaacgym
cd isaacgym
pip install -e .
```

## 2. 下载ABS代码

```bash
git clone https://github.com/LeCAR-Lab/ABS.git
cd ABS
pip install -r requirements.txt
```

## 3. 训练敏捷策略 (Agile Policy)

ABS框架包含四个主要模块，我们先从敏捷策略开始：

```python
# agile_policy_train.py
import isaacgym
from abs.policy.agile_policy import AgilePolicy
from abs.envs.agile_env import AgileEnv
from abs.utils.train_utils import setup_training

def train_agile_policy():
    # 初始化环境
    env = AgileEnv(
        num_envs=1280,  # 并行环境数量
        device="cuda:0"  # 使用GPU加速
    )
    
    # 初始化策略
    policy = AgilePolicy(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0]
    )
    
    # 设置训练参数
    cfg = {
        "total_steps": 10000000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "batch_size": 32768
    }
    
    # 开始训练
    trainer = setup_training(env, policy, cfg)
    trainer.learn()
    
    # 保存模型
    trainer.save("agile_policy.pt")

if __name__ == "__main__":
    train_agile_policy()
```

## 4. 训练RA值网络 (Reach-Avoid Value Network)

```python
# ra_value_train.py
from abs.value_nets.ra_value import RAValueNet
from abs.utils.data_utils import collect_agile_trajectories
from abs.utils.train_utils import ValueTrainer

def train_ra_value():
    # 收集敏捷策略的轨迹数据
    trajectories = collect_agile_trajectories(
        policy_path="agile_policy.pt",
        num_episodes=200000
    )
    
    # 初始化RA值网络
    value_net = RAValueNet(
        obs_dim=trajectories[0]['obs'].shape[1],
        hidden_dim=256
    )
    
    # 训练配置
    cfg = {
        "batch_size": 4096,
        "learning_rate": 1e-4,
        "gamma_ra": 0.999999,  # 接近1的高折扣因子
        "epochs": 100
    }
    
    # 训练RA值网络
    trainer = ValueTrainer(value_net, trajectories, cfg)
    trainer.train()
    
    # 保存模型
    trainer.save("ra_value_net.pt")

if __name__ == "__main__":
    train_ra_value()
```

## 5. 训练恢复策略 (Recovery Policy)

```python
# recovery_policy_train.py
from abs.policy.recovery_policy import RecoveryPolicy
from abs.envs.recovery_env import RecoveryEnv
from abs.utils.train_utils import setup_training

def train_recovery_policy():
    # 初始化恢复环境
    env = RecoveryEnv(
        num_envs=1280,
        device="cuda:0"
    )
    
    # 初始化恢复策略
    policy = RecoveryPolicy(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0]
    )
    
    # 训练配置
    cfg = {
        "total_steps": 5000000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "batch_size": 16384
    }
    
    # 开始训练
    trainer = setup_training(env, policy, cfg)
    trainer.learn()
    
    # 保存模型
    trainer.save("recovery_policy.pt")

if __name__ == "__main__":
    train_recovery_policy()
```

## 6. 训练射线预测网络 (Ray-Prediction Network)

```python
# ray_prediction_train.py
import torch
from abs.perception.ray_predictor import RayPredictor
from abs.utils.data_utils import load_depth_ray_dataset
from torch.utils.data import DataLoader

def train_ray_predictor():
    # 加载数据集
    dataset = load_depth_ray_dataset("data/depth_ray_pairs.h5")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 初始化模型
    model = RayPredictor(
        input_shape=(160, 90),  # 下采样后的深度图像尺寸
        num_rays=11             # 预测的射线数量
    )
    
    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # 训练循环
    for epoch in range(100):
        for depth, rays in loader:
            pred = model(depth)
            loss = criterion(pred, rays.log())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # 保存模型
    torch.save(model.state_dict(), "ray_predictor.pt")

if __name__ == "__main__":
    train_ray_predictor()
```

## 7. 部署ABS系统

训练完所有组件后，可以部署完整的ABS系统：

```python
# abs_deploy.py
import torch
from abs.system import ABS
from abs.perception.depth_camera import DepthCamera
from abs.robot_interface import RobotInterface

class ABSDeployer:
    def __init__(self):
        # 初始化各组件
        self.agile_policy = AgilePolicy.load("agile_policy.pt")
        self.ra_value_net = RAValueNet.load("ra_value_net.pt")
        self.recovery_policy = RecoveryPolicy.load("recovery_policy.pt")
        self.ray_predictor = RayPredictor.load("ray_predictor.pt")
        
        # 初始化传感器和机器人接口
        self.camera = DepthCamera()
        self.robot = RobotInterface()
        
        # ABS系统参数
        self.V_threshold = -0.05  # RA值阈值
    
    def run(self):
        while True:
            # 获取观测
            depth_img = self.camera.get_image()
            rays = self.ray_predictor(depth_img)
            proprio = self.robot.get_proprioception()
            
            # 组合敏捷策略的观测
            agile_obs = self._build_agile_obs(proprio, rays)
            
            # 计算RA值
            ra_obs = self._build_ra_obs(proprio, rays)
            V = self.ra_value_net(ra_obs)
            
            if V < self.V_threshold:
                # 使用敏捷策略
                action = self.agile_policy(agile_obs)
            else:
                # 使用恢复策略
                twist_cmd = self._optimize_twist(ra_obs)
                recovery_obs = self._build_recovery_obs(proprio, twist_cmd)
                action = self.recovery_policy(recovery_obs)
            
            # 执行动作
            self.robot.execute_action(action)
    
    def _optimize_twist(self, ra_obs):
        # 实现论文中的twist优化算法
        # 详见论文第5.3节
        pass

if __name__ == "__main__":
    deployer = ABSDeployer()
    deployer.run()
```

## 8. 关键实现细节

1. **敏捷策略训练**:
   - 使用目标到达(task-reaching)而非速度跟踪(velocity-tracking)的奖励设计
   - 包含位置跟踪奖励、航向奖励和敏捷奖励
   - 使用课程学习逐渐增加难度

2. **RA值网络**:
   - 基于Hamilton-Jacobi可达性理论
   - 使用折扣RA Bellman方程进行训练
   - 输入为低维观测(基座状态、目标命令和射线距离)

3. **恢复策略**:
   - 专为快速跟踪twist命令设计
   - 在敏捷策略可能失败时接管控制

4. **射线预测网络**:
   - 将深度图像映射到11条射线距离
   - 使用数据增强提高泛化能力

## 9. 训练建议

1. 先在少量环境(如128个)上测试代码是否正确
2. 逐步增加并行环境数量(论文使用1280个)
3. 监控训练曲线，确保各奖励项平衡增长
4. 敏捷策略训练约需4小时(在RTX 4090上)
5. 恢复策略训练更快，约10分钟可收敛

如需更详细的实现或遇到问题，可以参考论文中的附录或查看项目GitHub仓库中的完整代码。