use std::collections::HashMap;
use std::time::Instant;

use compare_onnx_runtimes::argmax;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. create tensor
    let input_image = compare_onnx_runtimes::mnist_input_image();
    let input_tensor = ndarray::Array4::from_shape_vec((1, 1, 28, 28), input_image)?;

    let mut inputs = HashMap::new();
    inputs.insert(
        "Input3".to_string(),
        input_tensor.as_slice().unwrap().into(),
    );

    // 2. load model and initialize session
    let session = wonnx::Session::from_path("models/mnist-opt.onnx").await?;

    // 3. run inference
    let start = Instant::now();
    let mut outputs: HashMap<String, wonnx::utils::OutputTensor> = session.run(&inputs).await?;
    let end = start.elapsed();

    // 4. fetch result
    let a: Vec<f32> = outputs.remove("Plus214_Output_0").unwrap().try_into()?;
    let ans = argmax(a.iter());

    println!("answer is {}", ans.unwrap());
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
