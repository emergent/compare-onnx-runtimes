use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let input_image: Vec<f32> = compare_onnx_runtimes::mnist_input_image();
    let input_tensor =
        ort::Value::from_array(([1usize, 1, 28, 28], input_image.into_boxed_slice()))?;

    ort::init().with_name("mnist").commit()?;

    let session = ort::Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level1)?
        .with_intra_threads(4)?
        .commit_from_file("models/mnist-opt.onnx")?;

    let start = Instant::now();
    let outputs = session.run(ort::inputs![input_tensor]?)?;
    let end = start.elapsed();

    let outputs = outputs["Plus214_Output_0"].try_extract_tensor::<f32>()?;
    println!("{}", outputs);
    let mut argmax = f32::MIN;
    let mut ans = 0;
    for (i, x) in outputs.into_iter().enumerate() {
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
