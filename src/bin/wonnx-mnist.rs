use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let input_image = compare_onnx_runtimes::mnist_input_image();

    let input_tensor = ndarray::Array4::from_shape_vec((1, 1, 28, 28), input_image)?;

    let mut inputs = HashMap::new();
    inputs.insert(
        "Input3".to_string(),
        input_tensor.as_slice().unwrap().into(),
    );

    let session = wonnx::Session::from_path("models/mnist-opt.onnx").await?;

    let outputs: HashMap<String, wonnx::utils::OutputTensor> = session.run(&inputs).await?;
    println!("{:?}", outputs);

    Ok(())
}
