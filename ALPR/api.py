from run import test
from flask import Flask

app = Flask(__name__)


@app.route('/predict')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()

