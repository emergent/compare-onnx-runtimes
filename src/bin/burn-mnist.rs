use burn::tensor;
use burn_ndarray::{NdArray, NdArrayDevice};
use compare_onnx_runtimes::model::mnist::Model;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device = NdArrayDevice::default();

    // Create model instance and load weights from target dir default device.
    // (see more load options below in "Loading and Using Models" section)
    let model: Model<NdArray<f32>> = Model::default();

    // Create input tensor (replace with your actual input)
    let input_image = compare_onnx_runtimes::mnist_input_image();
    let mut input_tensor =
        tensor::Tensor::<NdArray<f32>, 1>::from_floats(input_image.as_slice(), &device)
            .reshape([1, 1, 28, 28]);

    // Normalize the input
    input_tensor = ((input_tensor / 255) - 0.1307) / 0.3081;

    // Perform inference
    let start = Instant::now();
    let output = model.forward(input_tensor);
    let end = start.elapsed();

    println!("Model output: {:?}", output);

    let arg_max = output.argmax(1).into_scalar();
    println!("arg_max: {:?}", arg_max);

    /*     let mut argmax = f32::MIN;
       let mut ans = 0;
       for (i, x) in output. {
           println!("{}: {:?}", i, x);

           if *x > argmax {
               argmax = *x;
               ans = i;
           }
       }

    println!("answer is {}", ans);
    */
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    println!("Time taken: {:?}", end);
    Ok(())
}
