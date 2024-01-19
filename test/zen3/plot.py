import json
import matplotlib.pyplot as plt

# 加载 JSON 数据
with open('result.json', 'r') as file:
    data = json.load(file)

# 解析数据
openblas_gflops = []
zen3_gflops = []
sizes = [2**i for i in range(1, 12)]  # 测试大小是 2 的指数
t1 = 2
t2 = 2

for entry in data['benchmarks']:
    name = entry['name']
    iters = entry['iterations']
    cpu_time = entry['cpu_time']
    print(iters)
    print(cpu_time)
    print(' ')
    if 'openblas_gemm' in name:
        openblas_gflops.append(t1 * t1 * t1 * 2 / cpu_time )
        t1 *= 2
    elif 'zen3_gemm' in name:
        zen3_gflops.append(t2 * t2 * t2 * 2 / cpu_time )
        t2 *= 2
    
print(openblas_gflops)
print(zen3_gflops)

# 绘制图表
plt.plot(sizes, openblas_gflops, label='OpenBLAS GEMM')
plt.plot(sizes, zen3_gflops, label='Zen3 GEMM')
plt.xlabel('Size (M=N=K)')
plt.ylabel('GFlops/s')
plt.title('GEMM Performance Comparison')
plt.legend()
plt.savefig("./test.png")
