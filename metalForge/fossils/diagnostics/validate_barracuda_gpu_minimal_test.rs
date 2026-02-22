use barracuda::tensor::Tensor;
use neural_spring::gpu::Gpu;
use neural_spring::rng::Rng;

#[tokio::main]
async fn main() {
    let gpu = Gpu::new().await.unwrap();
    eprintln!("  adapter: {} ({:?}, {:?})", gpu.adapter_name, gpu.device_type, gpu.backend);
    let device = gpu.wgpu_device().clone();

    let mut rng = Rng::new(42);
    let n = 20_usize;
    let d = 8_usize;
    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..d).map(|_| rng.uniform()).collect())
        .collect();
    
    // CPU reference
    let mut cpu = vec![vec![0.0_f64; n]; n];
    for i in 0..n { for j in 0..n { for k in 0..d { cpu[i][j] += data[i][k] * data[j][k]; }}}
    
    let flat: Vec<f32> = data.iter().flat_map(|r| r.iter().map(|&x| x as f32)).collect();
    let x1 = Tensor::from_data(&flat, vec![n, d], device.clone()).unwrap();
    let x2 = Tensor::from_data(&flat, vec![n, d], device.clone()).unwrap();
    let x2t = x2.transpose().unwrap();
    let gram = x1.matmul(&x2t).unwrap();
    let out = gram.to_vec().unwrap();

    // Find zero entries
    eprintln!("Zero entries:");
    for i in 0..n {
        for j in 0..n {
            let gpu_val = out[i * n + j];
            if gpu_val == 0.0 {
                eprintln!("  [{i}][{j}] GPU=0.0 CPU={:.4}", cpu[i][j]);
            }
        }
    }
    
    // Summary by column
    eprintln!("\nZero pattern (0=ok, X=zero):");
    for i in 0..n {
        let mut row_str = String::new();
        for j in 0..n {
            if out[i * n + j] == 0.0 {
                row_str.push('X');
            } else {
                row_str.push('.');
            }
        }
        eprintln!("  row {i:2}: {row_str}");
    }
}
