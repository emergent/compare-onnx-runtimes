use compare_onnx_runtimes::argmax;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // 1. create tensor
    let input_image: Vec<f32> = compare_onnx_runtimes::mnist_input_image();
    let input_tensor =
        ort::Value::from_array(([1usize, 1, 28, 28], input_image.into_boxed_slice()))?;

    // 2. load model and initialize session
    ort::init().with_name("mnist").commit()?;
    let session = ort::Session::builder()?.commit_from_file("models/mnist-opt.onnx")?;

    // 3. run inference
    let start = Instant::now();
    let outputs = session.run(ort::inputs![input_tensor]?)?;
    let end = start.elapsed();

    // 4. fetch result
    let outputs = outputs["Plus214_Output_0"].try_extract_tensor::<f32>()?;
    let ans = argmax(outputs.iter());

    println!("answer is {}", ans.unwrap());
    println!("time: {}.{:09}", end.as_secs(), end.subsec_nanos());

    Ok(())
}
