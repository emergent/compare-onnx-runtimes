use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::Tensor;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let model = Model::load_file("models/mnist-opt.rten")?;
    let image_data = compare_onnx_runtimes::mnist_input_image();
    let mut image_tensor = Tensor::from_vec(image_data);
    image_tensor.reshape(&[1, 1, 28, 28]);

    let start = Instant::now();
    let outputs = model.run_one(image_tensor.view().into(), None)?;
    let end = start.elapsed();

    println!("{:?}", outputs);

    let mut argmax = f32::MIN;
    let mut ans = 0;
    for (i, x) in outputs
        .into_float()
        .unwrap()
        .data()
        .unwrap()
        .into_iter()
        .enumerate()
    {
        println!("{}: {}", i, x);
        if *x > argmax {
            argmax = *x;
            ans = i;
        }
    }
    println!("answer is {}", ans);
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
