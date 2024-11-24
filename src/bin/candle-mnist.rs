use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // 1. create tensor
    let input_image = compare_onnx_runtimes::mnist_input_image();
    let input_tensor =
        candle_core::Tensor::from_vec(input_image, &[1, 1, 28, 28], &candle_core::Device::Cpu)?;
    let mut inputs = HashMap::new();
    inputs.insert("Input3".to_string(), input_tensor);

    // 2. load model and initialize session
    let model = candle_onnx::read_file("models/mnist-opt.onnx")?;

    // 3. run inference
    let outputs = candle_onnx::simple_eval(&model, inputs)?;
    println!("{:?}", outputs);

    Ok(())
}
