const express = require("express");
const app = express();
const { PythonShell } = require("python-shell");
const port = 8000;

let image = "__";
let model_file = "__";
let weight_file = "__";
let mode = "pred";
let device = "cpu";

app.get("/model", (req, res, next) => {
  console.log("result: ");
  let options = {
    mode: "text",
    pythonPath: "python",
    scriptPath: "./python/",
    pythonOptions: ["-u"],
    args: [
      "--image", image,
      "--model_file", model_file,
      "--weight_file", weight_file,
      "--mode", mode,
      "--device", device,
    ],
  };

  PythonShell.run("run.py", options, function (err, result) {
    if (err) throw err;
    // result is an array consisting of messages collected
    //during execution of script.
    console.log("result: ", result.toString());
    res.send(result.toString());
  });
});

app.listen(port, () => console.log(`Server connected to ${port}`));
