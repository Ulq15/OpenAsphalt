// Method 1
const spawn = require("child_process").spawn;

// Run a Python script and return output
function runPythonScript(scriptPath, args) {
  // Use child_process.spawn method from
  // child_process module and assign it to variable
  const pyProg = spawn("python", [scriptPath].concat(args));

  // Collect data from script and print to console
  let data = "";
  pyProg.stdout.on("data", (stdout) => {
    data += stdout.toString();
  });

  // Print errors to console, if any
  pyProg.stderr.on("data", (stderr) => {
    console.log(`stderr: ${stderr}`);
  });

  // When script is finished, print collected data
  pyProg.on("close", (code) => {
    console.log(`child process exited with code ${code}`);
    console.log(data);
  });
}

// Run the Python file
runPythonScript("/path/to/python_file.py", [arg1, arg2]);

// Method 2
function postData(img_path) {
  $.ajax({
    type: "POST",
    url: "/api.py",
    data: { param: img_path },
    async: true,
    success: callbackFunc,
  });
}

function callbackFunc(response) {
  // do something with the response
  console.log(response);
}

postData("path to image");


// Method 3 - Flask API
import axios from 'axios';

function fetchAPI() {
  axios.get('http://localhost:5000/hello')
    .then(response => console.log(response.data))
}

class App extends React.Component {
    componentDidMount() {
      fetchAPI();
    }
  
    render() {
      return (
        // render code here
        0
      );
    }
  }