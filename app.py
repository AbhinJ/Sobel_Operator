from flask import Flask, render_template, request, send_from_directory, send_file
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import boto3, botocore
from werkzeug.utils import secure_filename
from io import BytesIO
from dotenv import load_dotenv


app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

s3 = boto3.client(
   "s3",
   aws_access_key_id= os.getenv('AWS_ACCESS_KEY'),
   aws_secret_access_key=os.getenv('AWS_SECRET_KEY')
)

def upload_file_to_s3(file, acl="public-read"):
    filename = secure_filename(file.filename)
    try:
        s3.upload_fileobj(
            file,
            os.getenv('AWS_BUCKET_NAME'),
            file.filename,
            ExtraArgs={
                "ACL": acl,
                "ContentType": file.content_type
            }
        )

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Something Happened: ", e)
        return e
    return file.filename

def read_image_from_s3(bucket, key, region_name='us-east-1'):
    
    outfile = BytesIO()
    s3.download_fileobj(bucket, key, outfile)
    outfile.seek(0)
    img = Image.open(outfile)
    return np.array(img)



def convolution(a,b):
  sum = 0
  for i in range(3):
    for j in range(3):
      sum += a[i,j]*b[i,j]
  return sum


@app.route("/")
def main():
    return render_template('index.html')

@app.route("/upload", methods=["POST"])
def upload():

    upload = request.files['file']
    option = request.form['options']
    # print(option)
    # filename = upload.filename
    # print(filename)
    # destination = "/".join([target, filename])
    # upload.save(destination)


    file_name = upload_file_to_s3(upload)
    print(file_name)


    img = read_image_from_s3(os.getenv('AWS_BUCKET_NAME'), file_name)
    global finalImgRow 
    global finalImgCol
    finalImgRow = len(img) - 2
    finalImgCol = len(img[0]) - 2
    global magnitude
    magnitude = np.zeros(shape=(finalImgRow, finalImgCol), dtype = np.uint8)
    kernelX = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    kernelY = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    for i in range(finalImgRow):
        for j in range(finalImgCol):
            if(img.ndim == 3):
                matBlue = img[i:i+3,j:j+3, 0]
                matGreen= img[i:i+3,j:j+3, 1]
                matRed = img[i:i+3,j:j+3, 2]

                gradientXB = convolution(matBlue, kernelX)
                gradientYB = convolution(matBlue, kernelY)

                gradientXG = convolution(matGreen, kernelX)
                gradientYG = convolution(matGreen, kernelY)

                gradientXR = convolution(matRed, kernelX)
                gradientYR = convolution(matRed, kernelY)

                if(option == "Maximum"):
                    gradientX = max(gradientXB, gradientXG, gradientXR)
                    gradientY = max(gradientYB, gradientYG, gradientYR)
                
                if(option == "Minimum"):
                    gradientX = min(gradientXB, gradientXG, gradientXR)
                    gradientY = min(gradientYB, gradientYG, gradientYR)
                    
                if(option == "Average"):
                    gradientX = (gradientXB + gradientXG + gradientXR)/3
                    gradientY = (gradientYB + gradientYG + gradientYR)/3

            if(img.ndim == 2):
                matImg = img[i:i+3, j:j+3]
                gradientX = convolution(matImg, kernelX)
                gradientY = convolution(matImg, kernelY)

            magnitude[i,j] = (gradientX**2 + gradientY**2)**0.5

    magnitude_np = np.array(magnitude)
    hist_magnitude = magnitude_np.flatten()
    
    mean = int(np.mean(hist_magnitude))
    median = int(np.median(hist_magnitude))

    counts, bins = np.histogram(hist_magnitude, bins = 50)
    plt.stairs(counts, bins)
    plt.title('Gradient Magnitude Histogram')
    plt.xlabel('Gradient Magnitude')
    plt.ylabel('Frequency')

    # target = os.path.join(APP_ROOT, 'static/images/chart.png')

    chart_data = BytesIO()
    plt.savefig(chart_data, format='png',bbox_inches='tight')
    chart_data.seek(0)
    s3.upload_fileobj(
            chart_data,
            os.getenv('AWS_BUCKET_NAME'),
            'chart.png',
            ExtraArgs={
                "ACL": 'public-read',
            }
    )

    plt.show()
    plt.close('all')
    image_url = f"https://sobeloperator.s3.amazonaws.com/{file_name}"
    chart_url = f"https://sobeloperator.s3.amazonaws.com/chart.png"
    return render_template("base.html", image_name = image_url, chart_name = chart_url, chart_mean = mean, chart_median = median)


@app.route("/sobel", methods=["POST"])
def sobel():
    # retrieve parameters from html form
    threshold = request.form['threshold']
    # filename = request.form['image']

    # # open and process image
    # target = os.path.join(APP_ROOT, 'static/images')
    # destination = "/".join([target, filename])


    # img = Image.open(destination)
    # img = np.array(img)
    # finalImgRow = len(img) - 2 
    # finalImgCol = len(img[0]) - 2
    convolvedImg = np.zeros(shape=(finalImgRow, finalImgCol), dtype = np.uint8)
    for i in range(finalImgRow):
        for j in range(finalImgCol):

            if(magnitude[i,j]>int(threshold)):
                convolvedImg[i,j] = 255
            else:
                convolvedImg[i,j] = 0

    # save and return image

    img_output = Image.fromarray(convolvedImg)
    # destination = "/".join([target, 'temp.png'])
    # if os.path.isfile(destination):
    #     os.remove(destination)
    output_data = BytesIO()
    img_output.save(output_data , format='png')
    output_data.seek(0)
    return send_file(output_data, mimetype="image/png")

# @app.route('/static/images/<filename>')
# def send_image(filename):
#     return send_from_directory("static/images", filename)