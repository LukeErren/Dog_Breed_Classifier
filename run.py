from flask import Flask
from flask import render_template, request, send_from_directory
import time
import breed

app = Flask(__name__)

# index webpage 
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

# web page that handles image
@app.route('/go', methods=['POST'])
def go():
    if request.method == 'POST':
      try:
        image_file = request.files['fileToUpload']
        image_file.seek(0)
        image_file.save('temp.jpg')
        breed.find_dog_breed_on_humans_and_dogs('temp.jpg')
        result = "<img src='result.png'>"
      except :
        result = "<b>No valid image found</b>" 
    else :
      result = "<b>Please select an image</b>"
          
    result = "<img class='NO-CACHE'  src='result.png?%s'>" % int(time.time())
    return render_template( 'go.html', query=result )

# Display image
@app.route('/result.png')
def image():
    return send_from_directory('.', 'result.png')

def main():
    app.run(host='0.0.0.0', port=3001, debug=False, threaded=False)

if __name__ == '__main__':
    main()