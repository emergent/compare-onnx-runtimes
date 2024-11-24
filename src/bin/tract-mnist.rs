use compare_onnx_runtimes::argmax;
use std::time::Instant;
use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    // 1. create tensor
    let input_image = compare_onnx_runtimes::mnist_input_image();
    let input_tensor: Tensor = ndarray::Array4::from_shape_vec((1, 1, 28, 28), input_image)?.into();

    // 2. load model and initialize session
    let model = tract_onnx::onnx()
        .model_for_path("models/mnist-opt.onnx")?
        .with_input_fact(0, f32::fact([1, 1, 28, 28]).into())?
        .incorporate()?
        .into_optimized()?
        .into_runnable()?;

    // 3. run inference
    let start = Instant::now();
    let outputs = model.run(tvec!(input_tensor.into()))?;
    let end = start.elapsed();

    // 4. fetch result
    let ans = argmax(outputs[0].to_array_view()?.iter());

    println!("answer is {}", ans.unwrap());
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
