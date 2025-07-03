from string import Template
import os

# 定义样本参数
SAMPLES = {
    "WT": {"sk_value": 1e-6},
    "dPBS": {"sk_value": 2.6e-7},
    "dOCP": {"sk_value": 1.25e-6}
}

# 读取模板文件
with open('template_optimize.py', 'r') as f:
    template = Template(f.read())

# 为每个样本生成脚本
for sample_name, params in SAMPLES.items():
    # 填充占位符
    script_content = template.substitute(
        sample_name=sample_name,
        sk_value=params["sk_value"]
    )
    
    # 写入文件
    output_path = f"optimize_{sample_name}.py"
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"Generated: {output_path}")