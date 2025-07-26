"""
setup_environment.py - FinGPT量化策略环境配置脚本
自动检查和配置运行环境
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


class EnvironmentSetup:
    """环境配置器"""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.success_steps = []
        
        print("FinGPT量化策略环境配置器")
        print("="*50)
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("\n1. 检查Python版本...")
        
        if self.python_version >= (3, 8):
            print(f"✅ Python版本: {sys.version}")
            self.success_steps.append("Python版本检查")
            return True
        else:
            error = f"❌ Python版本过低: {sys.version}，需要Python 3.8+"
            print(error)
            self.errors.append(error)
            return False
    
    def check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        print("\n2. 检查GPU可用性...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"✅ 检测到GPU: {gpu_name}")
                print(f"✅ GPU显存: {gpu_memory:.1f} GB")
                print(f"✅ CUDA版本: {torch.version.cuda}")
                
                if gpu_memory >= 16:
                    print("✅ 显存充足，可以运行FinGPT 13B模型")
                    self.success_steps.append("GPU检查")
                    return True
                else:
                    warning = f"⚠️  显存可能不足({gpu_memory:.1f}GB)，建议16GB+，但可以尝试8bit量化"
                    print(warning)
                    self.warnings.append(warning)
                    self.success_steps.append("GPU检查")
                    return True
            else:
                warning = "⚠️  未检测到CUDA GPU，将使用CPU（性能极差）"
                print(warning)
                self.warnings.append(warning)
                return False
                
        except ImportError:
            error = "❌ PyTorch未安装，无法检查GPU"
            print(error)
            self.errors.append(error)
            return False
    
    def check_disk_space(self) -> bool:
        """检查磁盘空间"""
        print("\n3. 检查磁盘空间...")
        
        try:
            free_space = shutil.disk_usage('.').free / 1024**3
            required_space = 30  # 30GB用于模型下载和缓存
            
            if free_space >= required_space:
                print(f"✅ 可用磁盘空间: {free_space:.1f} GB")
                self.success_steps.append("磁盘空间检查")
                return True
            else:
                error = f"❌ 磁盘空间不足: 需要{required_space}GB，仅有{free_space:.1f}GB"
                print(error)
                self.errors.append(error)
                return False
                
        except Exception as e:
            error = f"❌ 无法检查磁盘空间: {e}"
            print(error)
            self.errors.append(error)
            return False
    
    def install_dependencies(self) -> bool:
        """安装Python依赖"""
        print("\n4. 安装Python依赖...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            error = "❌ requirements.txt文件不存在"
            print(error)
            self.errors.append(error)
            return False
        
        try:
            print("正在安装依赖包，这可能需要几分钟...")
            
            # 安装基础依赖
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ 依赖包安装成功")
                self.success_steps.append("依赖安装")
                return True
            else:
                error = f"❌ 依赖安装失败: {result.stderr}"
                print(error)
                self.errors.append(error)
                return False
                
        except subprocess.TimeoutExpired:
            error = "❌ 依赖安装超时"
            print(error)
            self.errors.append(error)
            return False
        except Exception as e:
            error = f"❌ 依赖安装异常: {e}"
            print(error)
            self.errors.append(error)
            return False
    
    def create_directories(self) -> bool:
        """创建必要的目录"""
        print("\n5. 创建项目目录...")
        
        directories = [
            'logs',
            'cache', 
            'models',
            'data',
            'output',
            'output/backtest',
            'config'
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ 创建目录: {directory}")
            
            self.success_steps.append("目录创建")
            return True
            
        except Exception as e:
            error = f"❌ 目录创建失败: {e}"
            print(error)
            self.errors.append(error)
            return False
    
    def create_env_template(self) -> bool:
        """创建环境变量模板"""
        print("\n6. 创建环境变量模板...")
        
        env_template = """# FinGPT量化策略环境变量配置
# 复制此文件为.env并填入真实的API密钥

# 数据源API密钥
TIINGO_API_KEY=
FINNHUB_API_KEY=

# 交易接口（实盘交易需要）
ALPACA_API_KEY_ID=
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# HuggingFace Token（某些模型需要）
HUGGINGFACE_TOKEN=

# AWS配置（云部署需要）
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_DEFAULT_REGION=us-east-2

# 本地模型路径（可选，如果已下载模型）
LOCAL_FINGPT_MODEL_PATH=/path/to/local/fingpt/model

# 调试设置
DEBUG=False
LOG_LEVEL=INFO
"""
        
        try:
            env_file = self.project_root / ".env.template"
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_template)
            
            print(f"✅ 环境变量模板已创建: {env_file}")
            print("   请复制为.env文件并填入真实API密钥")
            
            self.success_steps.append("环境变量模板")
            return True
            
        except Exception as e:
            error = f"❌ 环境变量模板创建失败: {e}"
            print(error)
            self.errors.append(error)
            return False
    
    def test_model_loading(self) -> bool:
        """测试模型加载（可选）"""
        print("\n7. 测试模型连接（可选）...")
        
        response = input("是否测试FinGPT模型下载和加载？(y/N): ").strip().lower()
        
        if response != 'y':
            print("⏭️  跳过模型测试")
            return True
        
        try:
            print("正在测试FinGPT模型加载，首次运行将下载~27GB模型文件...")
            print("这可能需要10-30分钟，取决于网络速度...")
            
            # 测试导入
            from transformers import LlamaTokenizerFast
            from peft import PeftModel
            
            print("✅ Transformers和PEFT库可用")
            
            # 测试模型路径（不实际加载）
            try:
                tokenizer = LlamaTokenizerFast.from_pretrained(
                    "NousResearch/Llama-2-13b-hf",
                    cache_dir="./models"
                )
                print("✅ 基础模型连接成功")
                self.success_steps.append("模型连接测试")
                return True
                
            except Exception as e:
                warning = f"⚠️  模型连接测试失败: {e}"
                print(warning)
                print("   这可能是网络问题，运行时会自动重试")
                self.warnings.append(warning)
                return True
                
        except ImportError as e:
            error = f"❌ 模型库导入失败: {e}"
            print(error)
            self.errors.append(error)
            return False
        except Exception as e:
            warning = f"⚠️  模型测试异常: {e}"
            print(warning)
            self.warnings.append(warning)
            return True
    
    def create_quick_start_guide(self) -> bool:
        """创建快速开始指南"""
        print("\n8. 生成快速开始指南...")
        
        guide = """# FinGPT量化策略快速开始指南

## 1. 环境配置完成 ✅

恭喜！您的环境已配置完成。

## 2. 配置API密钥

1. 复制 `.env.template` 为 `.env`
2. 填入以下API密钥：
   - TIINGO_API_KEY: 获取股价数据 (https://api.tiingo.com/)
   - FINNHUB_API_KEY: 获取新闻数据 (https://finnhub.io/)
   - ALPACA_API_KEY: 实盘交易接口 (https://alpaca.markets/)

## 3. 运行测试

```bash
# 测试系统集成
python test_fingpt_integration.py

# 如果测试通过，运行完整回测
python run_strategy_backtest.py
```

## 4. 系统架构

- `factor_model_optimized.py`: FinGPT因子提取
- `signal_generator.py`: 交易信号生成
- `portfolio_manager.py`: 投资组合管理
- `backtest_engine.py`: 回测引擎
- `universe_builder.py`: 股票池管理

## 5. 配置调优

编辑 `config.py` 中的参数：
- 信号阈值
- 风控参数
- 模型配置

## 6. AWS部署

推荐实例：g5.2xlarge (A10G 24GB)
参考部署指南运行。

## 问题排查

- GPU显存不足：启用8bit量化
- 模型下载慢：使用本地缓存或S3
- API限制：检查密钥和额度

祝您交易成功！🚀
"""
        
        try:
            guide_file = self.project_root / "QUICK_START.md"
            with open(guide_file, 'w', encoding='utf-8') as f:
                f.write(guide)
            
            print(f"✅ 快速开始指南已创建: {guide_file}")
            self.success_steps.append("快速开始指南")
            return True
            
        except Exception as e:
            error = f"❌ 指南创建失败: {e}"
            print(error)
            self.errors.append(error)
            return False
    
    def run_setup(self) -> bool:
        """运行完整配置"""
        print("开始FinGPT量化策略环境配置...\n")
        
        # 配置步骤
        steps = [
            ("Python版本检查", self.check_python_version),
            ("GPU可用性检查", self.check_gpu_availability),
            ("磁盘空间检查", self.check_disk_space),
            ("依赖包安装", self.install_dependencies),
            ("目录创建", self.create_directories),
            ("环境变量模板", self.create_env_template),
            ("模型连接测试", self.test_model_loading),
            ("快速开始指南", self.create_quick_start_guide)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            try:
                if step_func():
                    success_count += 1
                else:
                    print(f"⚠️  步骤失败: {step_name}")
            except KeyboardInterrupt:
                print("\n用户中断配置")
                return False
            except Exception as e:
                print(f"❌ 步骤异常: {step_name} - {e}")
                self.errors.append(f"{step_name}异常: {e}")
        
        # 生成配置报告
        self.generate_setup_report(success_count, len(steps))
        
        return success_count >= len(steps) - 2  # 允许2个步骤失败
    
    def generate_setup_report(self, success_count: int, total_steps: int):
        """生成配置报告"""
        print("\n" + "="*60)
        print("环境配置报告")
        print("="*60)
        
        print(f"配置进度: {success_count}/{total_steps}")
        
        if self.success_steps:
            print("\n✅ 成功完成:")
            for step in self.success_steps:
                print(f"   - {step}")
        
        if self.warnings:
            print("\n⚠️  警告:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.errors:
            print("\n❌ 错误:")
            for error in self.errors:
                print(f"   - {error}")
        
        if success_count >= total_steps - 2:
            print("\n🎉 环境配置基本完成！")
            print("\n下一步:")
            print("1. 复制 .env.template 为 .env 并填入API密钥")
            print("2. 运行集成测试: python test_fingpt_integration.py")
            print("3. 如果测试通过，运行策略: python run_strategy_backtest.py")
        else:
            print("\n⚠️  环境配置未完全成功，请修复错误后重试")
        
        print("="*60)


def main():
    """主函数"""
    try:
        setup = EnvironmentSetup()
        success = setup.run_setup()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n用户中断配置")
        return 1
    except Exception as e:
        print(f"\n\n配置过程发生异常: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

