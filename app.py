"""
Flask Serving
This file is a sample flask app that can be used to test your model with an REST API.
This app does the following:
    - Look for a Zvector and/or n_samples parameters
    - Returns the output file generated at /output
Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: checkpoint
    - It is loaded from /input
"""
import os
import torch
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from dcgan import DCGAN

ALLOWED_EXTENSIONS = set(['pth'])

MODEL_PATH = '/model'
print('Loading model from path: %s' % MODEL_PATH)
OUTPUT_PATH = "/output/generated.png"

app = Flask('DCGAN-Generator')

#  2 possible parameters - checkpoint, zinput(file.cpth)
# Return an Image
@app.route('/<path:path>', methods=['GET', 'POST'])
def geneator_handler(path):
    zvector = None
    batchSize = 1
    # Upload a serialized Zvector
    if request.method == 'POST':
        # DO things
        # check if the post request has the file part
        if 'file' not in request.files:
            return BadRequest("File not present in request")
        file = request.files['file']
        if file.filename == '':
            return BadRequest("File name is not present in request")
        if not allowed_file(file.filename):
            return BadRequest("Invalid file type")
        filename = secure_filename(file.filename)
        input_filepath = os.path.join('/output', filename)
        file.save(input_filepath)
        # Load a Z vector and Retrieve the N of samples to generate
        zvector = torch.load(input_filepath)
        batchSize = zvector.size()[0]

    checkpoint = request.form.get("ckp") or "netG_epoch_69.pth"
    Generator = DCGAN(netG=os.path.join(MODEL_PATH, checkpoint), zvector=zvector, batchSize=batchSize)
    Generator.build_model()
    Generator.generate()
    return send_file(OUTPUT_PATH, mimetype='image/png')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0')
