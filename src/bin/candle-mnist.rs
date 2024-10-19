use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    let model = candle_onnx::read_file("models/mnist-opt.onnx")?;
    let input_image = compare_onnx_runtimes::mnist_input_image();

    let input_tensor =
        candle_core::Tensor::from_vec(input_image, &[1, 1, 28, 28], &candle_core::Device::Cpu)?;

    let mut inputs = HashMap::new();
    inputs.insert("Input3".to_string(), input_tensor);
    let outputs = candle_onnx::simple_eval(&model, inputs)?;
    println!("{:?}", outputs);

    Ok(())
}
