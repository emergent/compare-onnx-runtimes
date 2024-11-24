use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use compare_onnx_runtimes::{argmax, model::mnist::Model};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = NdArrayDevice::default();

    // 1. create tensor
    let input_image = compare_onnx_runtimes::mnist_input_image();
    let input_tensor =
        tensor::Tensor::<NdArray<f32>, 1>::from_floats(input_image.as_slice(), &device)
            .reshape([1, 1, 28, 28]);

    // 2. load model and initialize session
    let model: Model<NdArray<f32>> = Model::default();

    // 3. run inference
    let start = Instant::now();
    let output = model.forward(input_tensor);
    let end = start.elapsed();

    // 4. fetch result
    println!("Model output: {:?}", output);
    let ans = argmax(output.into_primitive().tensor().array.iter());

    println!("answer is {}", ans.unwrap());
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
