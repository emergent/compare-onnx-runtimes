use compare_onnx_runtimes::argmax;
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::Tensor;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // 1. create tensor
    let image_data = compare_onnx_runtimes::mnist_input_image();
    let mut image_tensor = Tensor::from_vec(image_data);
    image_tensor.reshape(&[1, 1, 28, 28]);

    // 2. load model and initialize session
    let model = Model::load_file("models/mnist-opt.rten")?;

    // 3. run inference
    let start = Instant::now();
    let outputs = model.run_one(image_tensor.view().into(), None)?;
    let end = start.elapsed();

    // 4. fetch result
    let ans = argmax(outputs.into_float().unwrap().data().unwrap().iter()).unwrap();

    println!("answer is {}", ans);
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
