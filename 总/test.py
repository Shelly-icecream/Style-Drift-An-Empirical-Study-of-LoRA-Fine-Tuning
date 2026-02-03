import transformers
print(transformers.__version__)  # 应该 >= 4.40.0
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def test_environment():
    """测试环境兼容性"""
    print("🧪 开始环境兼容性测试...")

    # 1. 测试PyTorch
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA版本: {torch.version.cuda}")
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

    # 2. 测试modelscope
    try:
        from modelscope import __version__ as ms_version
        print(f"✅ ModelScope版本: {ms_version}")
    except Exception as e:
        print(f"❌ ModelScope导入失败: {e}")
        return False

    # 3. 测试transformers
    try:
        import transformers
        print(f"✅ Transformers版本: {transformers.__version__}")
    except Exception as e:
        print(f"❌ Transformers导入失败: {e}")
        return False

    # 4. 测试PIL
    try:
        from PIL import Image
        print("✅ PIL导入成功")
    except Exception as e:
        print(f"❌ PIL导入失败: {e}")
        return False

    print("🎉 所有基础依赖测试通过!")
    return True


if __name__ == "__main__":
    test_environment()


