use std::collections::HashMap;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let input_image = compare_onnx_runtimes::mnist_input_image();

    let input_tensor = ndarray::Array4::from_shape_vec((1, 1, 28, 28), input_image)?;

    let mut inputs = HashMap::new();
    inputs.insert(
        "Input3".to_string(),
        input_tensor.as_slice().unwrap().into(),
    );

    let start = Instant::now();
    let session = wonnx::Session::from_path("models/mnist-opt.onnx").await?;
    let end = start.elapsed();

    let outputs: HashMap<String, wonnx::utils::OutputTensor> = session.run(&inputs).await?;
    println!("{:?}", outputs);

    let mut argmax = f32::MIN;
    let mut ans = 0;
    for x in outputs.into_values() {
        let x: Vec<f32> = x.try_into()?;
        for (j, y) in x.into_iter().enumerate() {
            println!("{}: {}", j, y);
            if y > argmax {
                argmax = y;
                ans = j;
            }
        }
    }
    println!("answer is {}", ans);
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
